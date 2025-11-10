# latest version
import argparse
import copy
import csv
import os
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
from UpperModel import UpperSAC
from LowerEnv import LowerEnv
from New_LowerModel import LowerSAC
from pyRDDLGym import RDDLEnv
from utils import TrainingLogger
from HalfCheetahEnv2 import HalfCheetahEnv

class PredMod:
    def __init__(self, upper_domain, lower_domain, instance, text, training_time, log_path):
        # classify predicates
        self.log_path = log_path
        self.time_start = time.time()
        self.training_time = training_time
        self.dfa = get_dfa(text)
        self.dfa_trans = DFATransformer(text)
        self.upper_buffer_action = None

        upper_env = UpperRMEnv(HalfCheetahEnv())
        self.upper_model = UpperSAC("MultiInputPolicy", upper_env, verbose=1, learning_starts=500,
                  learning_rate=3e-4, batch_size=256, train_freq=1, device='cpu')
        lower_env = HalfCheetahEnv()
        eval_env = maybe_make_env(copy.deepcopy(lower_env), 1)
        self.eval_env = BaseAlgorithm._wrap_env(eval_env, 1, True)
        self.lower_model = LowerSAC("MultiInputPolicy", lower_env, verbose=1, learning_starts=500,
                  learning_rate=3e-4, batch_size=256, train_freq=1, device='cpu', dfa_text=text, upper_model=self.upper_model)
        self.logger = TrainingLogger(log_interval=100)
        self.upper_logger = {}
        self.eval_upper_logger = {}
        self.max_currentH = 800


    def get_lower_model_outedge(self, obs, deterministic=False):
        action, _ = self.upper_model.predict(obs, deterministic)
        dfa_state = '@q' + str(obs['ds'].item() + 1)
        out_edges = self.dfa.out_edges(dfa_state, data=True)
        out_edges = [(u, v, data) for u, v, data in out_edges if u != v and v != '@q2']
        select_option = int(action.item() * len(out_edges))
        try:
            u, v, _ = out_edges[select_option]
        except:
            print('a')

        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.upper_model.action_space, spaces.Box):
            scaled_action = self.upper_model.policy.scale_action(action)
            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = self.upper_model.policy.unscale_action(scaled_action)
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = action
            action = buffer_action
        return u, v, action, buffer_action

    def evaluate(self, seconds):
        reward = np.array([0])
        done = False
        obs = self.eval_env.reset()
        length = 0
        while not done:
            u, v, _, _ = self.get_lower_model_outedge(obs, deterministic=True)
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
            [hour, reward, length]
        ]
        with open(self.log_path, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(data)
        print('time:', hours, "h", minutes, "min", second, 's')
        print('eval_reward:', reward, 'eval_length:', length)

    def set_upper_logger(self, logger, dfa_state, current_option_index, if_success):
        next_dfa_state_dict = logger.get(dfa_state, {})
        success_time = next_dfa_state_dict.get(str(current_option_index), [0, 0, 0])
        success_time[1] += 1
        if if_success:
            success_time[0] += 1
        success_time[2] = success_time[0] / success_time[1]
        next_dfa_state_dict[str(current_option_index)] = success_time
        logger[dfa_state] = next_dfa_state_dict

    def collect_rollouts(
            self,
            env: VecEnv,
            callback: BaseCallback,
            train_freq: TrainFreq,
            replay_buffer: ReplayBuffer,
            action_noise: Optional[ActionNoise] = None,
            learning_starts: int = 0,
            log_interval: Optional[int] = 100,
    ):

        # predict out edge
        if not self.lower_model.current_outedge:
            u, v, self.upper_model.upper_action, self.upper_buffer_action = self.get_lower_model_outedge(self.lower_model._last_obs)
            self.lower_model.current_outedge = (u, v)

        u, v = self.lower_model.current_outedge
        self.lower_model.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

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
            dfa_state = '@q' + str(self.lower_model._last_obs['ds'].item() + 1)
            assert dfa_state == u
            next_dfa_state = '@q' + str(new_obs['ds'].item() + 1)
            next_values = None

            # set dones
            if ((next_dfa_state != dfa_state) or self.lower_model.currentH == self.max_currentH) and not original_dones:
                dones = np.array([True])
                infos[0]['terminal_observation'] = new_obs
                self.lower_model.currentH = 0

            # reward shaping
            if original_dones and rewards < 1:
                rewards = np.array([-10])
            elif original_dones and rewards > 1:
                rewards = np.array([100])
            elif next_dfa_state != dfa_state:
                rewards = np.array([10])
            else:
                rewards = np.array([0])
            # reset lower env
            if dones:
                self.lower_model.currentH = 0

            '''# set upper log
            if dones or next_dfa_state in self.upper_model.end_states:
                self.set_upper_logger(self.upper_logger, dfa_state, self.lower_model.current_option_index, next_dfa_state in self.upper_model.end_states)'''

            if self.logger:
                self.logger.record(reward=rewards, done=original_dones)

            # save exp in upper model and train
            if dones or next_dfa_state in self.upper_model.end_states:
                self.upper_model._store_transition(self.upper_model.replay_buffer, self.upper_buffer_action, new_obs, rewards,
                                                   original_dones, infos)
                self.upper_model.num_timesteps += 1
                if self.upper_model.num_timesteps > self.upper_model.learning_starts:
                    self.upper_model.train(batch_size=self.upper_model.batch_size, gradient_steps=self.upper_model.gradient_steps)


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

            # Counterfactual reasoning
            # get props
            new_obs_ = copy.deepcopy(new_obs) if not original_dones else copy.deepcopy(infos[0]['terminal_observation'])
            if original_dones:
                for k, v in new_obs_.items():
                    new_obs_[k] = np.array([[v]])
            props = {}
            for key in new_obs_.keys():
                if key.startswith("p-"):
                    key_ = key.split('-')[1]
                    try:
                        props[key_] = new_obs_[key].item() if not original_dones else new_obs_[key]
                    except:
                        print('a')

            for u1, v1 in self.dfa.edges():
                if u1 == v1 or v1 == "@q2":
                    continue
                formula = self.dfa[u1][v1]['formula']
                if evaluate_logic_formula(props, formula) or original_dones:
                    d = np.array([True])
                else:
                    d = np.array([False])

                if evaluate_logic_formula(props, formula):
                    r = np.array([10])
                    new_obs_['ds'] = np.array([[int(v1[-1])]])
                else:
                    r = np.array([0])
                    v1_ = dfa_step(self.dfa, u1, props)
                    new_obs_['ds'] = np.array([[int(v1_[-1])]])

                obs = copy.deepcopy(self.lower_model._last_obs)
                obs['ds'] = np.array([[int(u1[-1])]])

                self.lower_model._store_transition(replay_buffer, buffer_actions, obs, new_obs_, r, d, infos, u1, v1)

            self.lower_model._last_obs = new_obs

            self.lower_model._update_current_progress_remaining(self.lower_model.num_timesteps, self.lower_model._total_timesteps)

            self.lower_model._on_step()

            if next_dfa_state != dfa_state and not original_dones:
                u_, v_, self.upper_model.upper_action, self.upper_buffer_action = self.get_lower_model_outedge(
                    new_obs)
                self.lower_model.current_outedge = (u_, v_)


            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    num_collected_episodes += 1
                    self.lower_model._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    if log_interval is not None and self.lower_model._episode_num % log_interval == 0 and self.logger is not None:
                        self.logger.get_info(self.lower_model.num_timesteps)

        callback.on_rollout_end()
        return original_dones

    def learn(
            self,
            total_timesteps: int= 99999999,
            callback: MaybeCallback = None,
            log_interval: int = 50,
            tb_log_name: str = "run",
            reset_num_timesteps: bool = True,
            progress_bar: bool = False,
    ):
        total_timesteps, callback = self.upper_model._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )
        total_timesteps, callback = self.lower_model._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )


        now_time = time.time()
        while time.time() - self.time_start <= self.training_time:
            if time.time() - now_time > 60:
                self.evaluate(time.time() - self.time_start)
                now_time = time.time()
                # print(self.upper_logger)
            done = self.collect_rollouts(
                self.lower_model.env,
                callback,
                self.lower_model.train_freq,
                self.lower_model.replay_buffer,
                self.lower_model.action_noise,
                self.lower_model.learning_starts,
                log_interval
            )
            u, v = self.lower_model.current_outedge

            if self.lower_model.num_timesteps > 0 and self.lower_model.dfa[u][v]['sample_start']:
                self.lower_model.train(batch_size=self.upper_model.batch_size,
                                       gradient_steps=self.lower_model.gradient_steps, u=u, v=v)
            if done:
                self.lower_model.current_outedge = None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-log', type=str, default='task2.csv', help='log path')
    parser.add_argument('-i', type=str, default='inst31', help='inst name')
    parser.add_argument('-r', type=str, default='waterworld3', help='inst name')
    parser.add_argument('-c', type=int, default=6, help='process num')
    parser.add_argument('-t', type=int, default=4800, help='training time')
    args = parser.parse_args()
    upper_domain = 'high_level_benchmarks/waterworld/' + args.r + '.rddl'
    lower_domain = 'low_level_benchmarks/waterworld/' + args.r + '.rddl'
    instance = 'low_level_benchmarks/waterworld/' + args.i + '.rddl'
    text_path = 'dfa_text/halfcheetah/halfcheetah2.txt'
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
