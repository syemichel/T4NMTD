# 多线程
from util.DFA import *
import argparse
import collections
import copy
import csv
import os
import re
from threading import Lock
from stable_baselines3.common.base_class import maybe_make_env, BaseAlgorithm
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.type_aliases import MaybeCallback
import torch as th
from torch.nn import functional as F
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
from pyRDDLGym import RDDLEnv
from LowerModel_exp4 import LowerSAC
from stable_baselines3 import SAC, PPO
from server.ParameterServer import *
from model.UpperModel import DFAPathFinder
from test.Evaluator import SACTestAgent
from server.InitialStateCreator import InitialStateCreator
import time
import ray
from env.GetEnv import *

class T4DMT:
    def __init__(self, env_name, dfa_text, dfa_path, training_time, option_num, log_path):
        self.option_num = option_num
        self.eval_env = GetLowerEnv(name=env_name)
        self.eval_env.reset()
        self.eval_time = 5
        self.dfa_text = dfa_text
        self.training_time = training_time
        self.log_path = log_path
        self.dfa = get_dfa(dfa_text)
        self.env_name = env_name
        self.dfa_path = dfa_path

    def get_add_state_for_dfa(self, dfa, option_num):
        state_index = []
        for i in range(option_num):
            matching_nodes = []
            for node in dfa.nodes():
                if node == "@q2":
                    continue
                for neighbor in dfa[node]:
                    if neighbor == "@q2" or neighbor == node:
                        continue
                    options = extract_true_predicates(dfa[node][neighbor]['formula'], dfa[node][node]['formula'])
                    indexes = [int(o[-1]) for o in options]
                    if i + 1 in indexes:  # 这里的条件可能需要根据实际情况调整
                        matching_nodes.append(node)
                        break  # 找到一个符合条件的邻居即可，无需继续检查其他邻居
            # 构建映射
            state_index.append(matching_nodes)
        return state_index

    def run(self):

        reset_ps = ray.remote(num_cpus=0.5)(ResetStatesServer)
        state_index_list = self.get_add_state_for_dfa(self.dfa, self.option_num)
        reset_ps = reset_ps.remote(state_index_list)

        lower_envs = [GetLowerEnv(name=self.env_name, option_index=i, reset_ps=reset_ps) for i in range(self.option_num)]


        model = SAC("MultiInputPolicy", lower_envs[0], verbose=1, learning_starts=1000,
                      learning_rate=3e-4, batch_size=256, train_freq=1, action_noise=None,
                      buffer_size=100000, device='cpu')

        upper_model_ps = ray.remote(num_cpus=1)(DFAPathFinder)
        upper_model_ps = upper_model_ps.remote(self.dfa, epsilon=0.1, option_num=self.option_num)

        actor_ps = ray.remote(num_cpus=1)(NetworkServer)
        actor_ps = actor_ps.remote(net=model.actor, option_num=self.option_num)
        critic_ps = ray.remote(num_cpus=1)(NetworkServer)
        critic_ps = critic_ps.remote(net=model.critic_target, option_num=self.option_num)

        eval_env = GetLowerEnv(name=self.env_name)
        eval_env = BaseAlgorithm._wrap_env(maybe_make_env(eval_env, 1), 1, True)

        initial_state = eval_env.reset()
        IS_creator = ray.remote(num_cpus=0.5)(InitialStateCreator)
        IS_creator = IS_creator.remote(self.option_num, initial_state, 5, 15, upper_model_ps, self.dfa_text, reset_ps, self.eval_env)
        IS_creator.run.remote()

        evaluator = ray.remote(num_cpus=1)(SACTestAgent)
        evaluator = evaluator.remote(actor_ps, critic_ps, upper_model_ps, reset_ps, self.log_path, self.eval_env,
                                     self.option_num, model.policy, self.dfa_text)

        '''learn = LowerSAC(policy="MultiInputPolicy",
                           env=lower_envs[0],
                           actor_ps=actor_ps,
                           critic_ps=critic_ps,
                           reset_ps=reset_ps,
                           option_index=0,
                           dfa_text=self.dfa_text,
                           upper_model_ps=upper_model_ps,
                           evaluator=evaluator,
                           verbose=1,
                           learning_starts=1000,
                           learning_rate=3e-4,
                           batch_size=256,
                           train_freq=1,
                           device='cpu',
                           option_num=self.option_num,
                           )
        stop = False
        while True and not stop:
            time.sleep(30)
            stop, _ = ray.get(evaluator.evaluate.remote(self.training_time))
        learn.learn()'''

        lower_learners = [ray.remote(num_cpus=2.25)(LowerSAC) for _ in range(self.option_num)]
        lower_learners = [
            learner.remote(policy="MultiInputPolicy",
                           env=lower_envs[i],
                           actor_ps=actor_ps,
                           critic_ps=critic_ps,
                           reset_ps=reset_ps,
                           option_index=i,
                           dfa_text=self.dfa_text,
                           upper_model_ps=upper_model_ps,
                           evaluator=evaluator,
                           verbose=1,
                           learning_starts=1000,
                           learning_rate=3e-4,
                           batch_size=256,
                           train_freq=1,
                           device='cpu',
                           option_num=self.option_num,
                           )
            for i, learner in enumerate(lower_learners)]

        # ray.wait([learner.upload_nets.remote() for learner in lower_learners])
        # [learner.download_nets.remote() for learner in lower_learners]
        [learner.learn.remote() for learner in lower_learners]
        del model
        del lower_envs
        stop = False
        while True and not stop:
            time.sleep(30)
            stop, _ = ray.get(evaluator.evaluate.remote(self.training_time))

        # save lower models
        log_path = self.log_path.rsplit('/', 1)[0]
        for i, model in enumerate(lower_learners):
            print('start save')
            model.save.remote(log_path + '/model' + str(i))
            # ray.wait([model.save.remote('log/task1/sac' + str(i))])
            print(i)
        time.sleep(500)
        print('Training Finish')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-log', type=str, default='log1/task1/task1.csv', help='log path')
    parser.add_argument('-i', type=str, default='inst11', help='inst name')
    parser.add_argument('-r', type=str, default='halfcheetah1', help='inst name')
    parser.add_argument('-o', type=int, default=5, help='process num')
    parser.add_argument('-t', type=int, default=3600, help='training time')
    args = parser.parse_args()

    training_time = args.t
    option_num = args.o
    upper_domain = 'high_level_benchmarks/waterworld/' + args.r + '.rddl'
    lower_domain = 'low_level_benchmarks/waterworld/' + args.r + '.rddl'
    instance = 'low_level_benchmarks/waterworld/' + args.i + '.rddl'

    name = args.r
    with open('../main/dfa_text_config.json') as f:
        text_path = json.load(f)['prefix_mappings']
    text_path = text_path[name]
    with open(text_path, 'r', encoding='utf-8') as file:
        dfa_text = file.read()
    log_path = args.log
    data = [
        ['training_times', 'time', 'mean_reward', 'mean_length'],
    ]
    directory = os.path.dirname(log_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    with open(log_path, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(data)
    #ray.init(address='auto')
    ray.init()
    agent = T4DMT(name, dfa_text, text_path, training_time, option_num, log_path)

    agent.run()
    ray.shutdown()

