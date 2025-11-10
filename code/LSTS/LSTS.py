# latest version
import argparse
import copy
import csv
import os
import random
import sys
import time
from typing import Optional
sys.setrecursionlimit(1000000)
import numpy as np
from stable_baselines3.common.base_class import maybe_make_env, BaseAlgorithm
from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.utils import safe_mean, should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv
from utils import *
from LowerEnv import LowerEnv
from LowerModel import LowerSAC
from pyRDDLGym import RDDLEnv
from utils import TrainingLogger

class PredMod:
    def __init__(self, upper_domain, lower_domain, instance, text, training_time, log_path):
        # classify predicates
        self.log_path = log_path
        self.time_start = time.time()
        self.training_time = training_time
        self.dfa = get_dfa(text)
        self.dfa_trans = DFATransformer(text)
        self.upper_buffer_action = None
        self.active_tasks = set()
        self.learned_tasks = []
        self.discard_tasks = set()

        lower_env = FlattenAction(LowerEnv(domain=lower_domain, instance=instance))
        eval_env = maybe_make_env(copy.deepcopy(lower_env), 1)
        self.eval_env = BaseAlgorithm._wrap_env(eval_env, 1, True)
        self.lower_model = LowerSAC("MultiInputPolicy", lower_env, verbose=1, learning_starts=500,
                  learning_rate=3e-4, batch_size=256, device='cpu', dfa_text=text, train_freq=5)
        self.logger = TrainingLogger(log_interval=100)
        self.max_currentH = 100
        self.sub_task_learn_time = 60
        self.set_active_tasks("@q1")
        self.sample_active_task()

    def update_q_value(self, u, v, average_rewards):
        alpha = 0.2
        self.lower_model.dfa[u][v]['Q_value'] = alpha * average_rewards + (1 - alpha) * self.lower_model.dfa[u][v]['Q_value']

    def set_active_tasks(self, dfa_state):
        out_edges = self.dfa.out_edges(dfa_state, data=False)
        out_edges = [(u, v) for u, v in out_edges if u != v and v != '@q2']
        stop_training = True
        for edge in out_edges:
            if edge not in self.learned_tasks or edge not in self.discard_tasks:
                self.active_tasks.add(edge)
                stop_training = False
        return stop_training

    def sample_active_task(self):
        Q_value = {}
        for task in self.active_tasks:
            u, v = task
            Q_value[task] = self.lower_model.dfa[u][v]['Q_value']
        if random.random() < 0.7:
            active_task = random.choice(list(self.active_tasks))
        else:
            active_task = max(Q_value.keys(), key=lambda k: Q_value[k])
        return active_task

    def set_discarded_tasks(self, goal_dfa_state):
        """直接通过主路径边集合计算需要删除的边
        Args:
            G: nx.DiGraph
            main_edges: 主路径边集合，如 [(v0, v1), (v1, v2), ..., (vk, p)]

        Returns:
            list: 需要删除的边列表 [(src, dst), ...]
        """
        # find edge path
        main_edges = self.find_edge_path(self.learned_tasks, goal_dfa_state)

        # 提取主路径覆盖的关键节点（所有被指向的节点）
        critical_nodes = {v for _, v in main_edges}

        # 生成需要删除的边列表（指向关键节点但不在主路径中的边）
        discard_tasks = [
            (u, v) for u, v in self.dfa.edges()
            if v in critical_nodes and (u, v) not in main_edges
        ]
        self.discard_tasks.update(discard_tasks)

    def find_edge_path(self, edges, target):
        # 构建父节点映射 {子节点: 父节点}
        parent_map = {}
        for u, v in edges:
            parent_map[v] = u

        # 回溯节点路径
        path = []
        current = target
        while current in parent_map:
            path.append(current)
            current = parent_map[current]
        path.append(current)  # 添加起点
        path.reverse()

        # 将节点路径转为边路径
        edge_path = []
        for i in range(len(path) - 1):
            edge = (path[i], path[i + 1])
            edge_path.append(edge)

        return edge_path

    def policy_advance(self, goal_dfa_state):
        self.lower_model._last_obs = self.lower_model.env.reset()
        edge_path = self.find_edge_path(self.learned_tasks, goal_dfa_state)
        dfa_state = '@q' + str(self.lower_model._last_obs['ds'].item() + 1)
        length = 0
        for edge in edge_path:
            u, v = edge
            assert dfa_state == u
            while dfa_state != v or length == 300:
                action, _ = self.lower_model.predict_eval(self.lower_model._last_obs, u, v, deterministic=True)
                new_obs, reward_, done, info = self.lower_model.env.step(action)
                length += 1
                assert not done
                self.lower_model._last_obs = new_obs
                dfa_state = '@q' + str(self.lower_model._last_obs['ds'].item() + 1)
        assert dfa_state == goal_dfa_state
        return True

    def collect_rollouts(
            self,
            env: VecEnv,
            callback: BaseCallback,
            train_freq: TrainFreq,
            replay_buffer: ReplayBuffer,
            action_noise: Optional[ActionNoise] = None,
            learning_starts: int = 0,
            log_interval=100,
            u="",
            v="",
    ):
        # set last obs
        dfa_state = '@q' + str(self.lower_model._last_obs['ds'].item() + 1)
        if dfa_state != u:
            success_advance = False
            while not success_advance:
                try:
                    success_advance = self.policy_advance(u)
                except:
                    pass

        self.lower_model.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        dfa_state = '@q' + str(self.lower_model._last_obs['ds'].item() + 1)
        assert dfa_state == u
        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        callback.on_rollout_start()
        continue_training = True
        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            if self.lower_model.use_sde and self.lower_model.sde_sample_freq > 0 and num_collected_steps % self.lower_model.sde_sample_freq == 0:
                self.lower_model.dfa[u][v]['actor'].reset_noise(env.num_envs)

            # Select action randomly or according to policy
            actions, buffer_actions = self.lower_model._sample_action(learning_starts, action_noise, env.num_envs, u, v)

            # Rescale and perform action
            new_obs, rewards, dones, infos = env.step(actions)
            self.lower_model.currentH += 1
            original_dones = dones.copy()
            # reward shaping
            next_dfa_state = '@q' + str(new_obs['ds'].item() + 1)
            next_values = None

            # set dones
            if (next_dfa_state != dfa_state) or self.lower_model.currentH == self.max_currentH:
                dones = np.array([True])
                infos[0]['terminal_observation'] = new_obs

            # reward shaping
            if dones:
                if next_dfa_state == v or rewards > 1:
                    rewards += 10
                else:
                    rewards = np.array([-10])
            else:
                rewards = np.array([0])

            if self.logger:
                self.logger.record(reward=rewards, done=dones)

            self.lower_model.num_timesteps += env.num_envs
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if not callback.on_step():
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes,
                                     continue_training=False)

            # Retrieve reward and episode length if using Monitor wrapper
            self.lower_model._update_info_buffer(infos, dones)

            self.lower_model._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos, u, v)

            self.lower_model._update_current_progress_remaining(self.lower_model.num_timesteps,
                                                                self.lower_model._total_timesteps)

            self.lower_model._on_step()

            if dones:
                self.lower_model.currentH = 0
                self.lower_model._last_obs = self.lower_model.env.reset()

            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    num_collected_episodes += 1
                    self.lower_model._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

        callback.on_rollout_end()
        return True

    def select_convergence(self, last_average_rewards, average_rewards):
        print(sum(self.logger.if_successes), len(self.logger.if_successes))
        success_rate = sum(self.logger.if_successes) / len(self.logger.if_successes)
        print(success_rate, average_rewards, round(abs(average_rewards - last_average_rewards), 2))
        return success_rate > 0.95 and abs(average_rewards - last_average_rewards) < 1

    def learn(
            self,
            total_timesteps: int = 99999999,
            callback: MaybeCallback = None,
            log_interval: int = 50,
            tb_log_name: str = "run",
            reset_num_timesteps: bool = True,
            progress_bar: bool = False,
    ):
        total_timesteps, callback = self.lower_model._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        now_time = time.time()
        while time.time() - self.time_start <= self.training_time:
            u, v = self.sample_active_task()
            print("sample task is:", u, v)
            while time.time() - now_time < 60 or len(self.logger.episode_rewards) < 3:
                self.collect_rollouts(
                    self.lower_model.env,
                    callback,
                    self.lower_model.train_freq,
                    self.lower_model.replay_buffer,
                    self.lower_model.action_noise,
                    self.lower_model.learning_starts,
                    log_interval,
                    u,
                    v
                )
                if self.lower_model.num_timesteps > 0 and self.lower_model.dfa[u][v]['sample_start']:
                    self.lower_model.train(batch_size=self.lower_model.batch_size,
                                           gradient_steps=self.lower_model.gradient_steps, u=u, v=v)

            average_rewards = self.logger.get_average_rewards()
            self.update_q_value(u, v, average_rewards)
            if self.select_convergence(self.lower_model.dfa[u][v]['average_rewards'], average_rewards):
                try:
                    self.learned_tasks.append((u, v))
                    for i in range(20):
                        self.policy_advance(v)
                    self.active_tasks.remove((u, v))
                    self.set_discarded_tasks(v)
                    stop = self.set_active_tasks(v)
                    if stop:
                        break
                    self.logger.reset()
                except:
                    self.learned_tasks.remove((u, v))
                    print("policy_advance error.")
            self.lower_model.dfa[u][v]['average_rewards'] = average_rewards
            self.evaluate(time.time() - self.time_start)
            self.logger.reset()
            now_time = time.time()


    def sample_random_task(self, dfa_state):
        out_edges = self.dfa.out_edges(dfa_state, data=False)
        out_edges = [(u, v) for u, v in out_edges if u != v and v != '@q2']
        Q_value = {}
        for task in out_edges :
            u, v = task
            Q_value[task] = self.lower_model.dfa[u][v]['Q_value']
        if random.random() < 0.7:
            active_task = random.choice(list(self.active_tasks))
        else:
            active_task = max(Q_value.keys(), key=lambda k: Q_value[k])
        return active_task


    def get_learned_task(self, dfa_state):
        for task in self.learned_tasks:
            if dfa_state == task[0]:
                return task
        return self.sample_random_task(dfa_state)

    def evaluate(self, seconds):
        reward = np.array([0])
        done = False
        obs = self.eval_env.reset()
        length = 0
        while not done:
            dfa_state = '@q' + str(obs['ds'].item() + 1)
            u, v = self.get_learned_task(dfa_state)
            while not done:
                action, _ = self.lower_model.predict_eval(obs, u, v, deterministic=True)
                new_obs, reward_, done, info = self.eval_env.step(actions=action)
                length += 1
                # reward shaping
                dfa_state = '@q' + str(obs['ds'].item() + 1)
                next_dfa_state = '@q' + str(new_obs['ds'].item() + 1)

                obs = new_obs
                # reward shaping
                if done and reward_ < 1:
                    reward += 0
                elif done and reward_ > 1:
                    reward = np.array([100])
                elif next_dfa_state != dfa_state:
                    reward += 10
                    break
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        second = int(seconds % 60)
        hour = str(hours) + "h" + str(minutes) + "min" + str(second) + 's'
        data = [
            [hour, reward.item(), length]
        ]
        with open(self.log_path, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(data)
        print('time:', hours, "h", minutes, "min", second, 's')
        print('eval_reward:', reward.item(), 'eval_length:', length)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-log', type=str, default='task2.csv', help='log path')
    parser.add_argument('-i', type=str, default='inst1', help='inst name')
    parser.add_argument('-r', type=str, default='racecar1', help='inst name')
    parser.add_argument('-c', type=int, default=6, help='process num')
    parser.add_argument('-t', type=int, default=4800, help='training time')
    args = parser.parse_args()
    upper_domain = 'high_level_benchmarks/racecar/' + args.r + '.rddl'
    lower_domain = 'low_level_benchmarks/racecar/' + args.r + '.rddl'
    instance = 'low_level_benchmarks/racecar/' + args.i + '.rddl'
    text_path = 'dfa_text/racecar/' + args.r + '.txt'
    with open(text_path, 'r', encoding='utf-8') as file:
        text = file.read()

    log_path = args.log
    data = [
        ['time', 'mean_reward', 'mean_length'],
    ]
    directory = os.path.dirname(log_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    with open(log_path, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(data)

    model = PredMod(upper_domain, lower_domain, instance, text, training_time=args.t, log_path=log_path)
    model.learn(log_interval=100)
