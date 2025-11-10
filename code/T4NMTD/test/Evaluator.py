import csv
import time
import argparse
import collections
import copy
import os
import re
import sys
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.base_class import BaseAlgorithm, maybe_make_env

sys.setrecursionlimit(1000000)
import csv
import torch as th
import time
import ray
from util.DFA import *
import pickle
from env.GetEnv import *
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

class SACTestAgent:
    def __init__(self, actor_ps, critic_ps, upper_model_ps, reset_ps, log_path, eval_env, option_num, policy, dfa_text):
        self.actor_ps = actor_ps
        self.critic_ps = critic_ps
        self.reset_ps = reset_ps
        self.eval_time = 1
        self.time_start = time.time()
        # eval_env = MinigridActionEnv(AddDFAStateObs(MiniGridEnv1(render_mode='human')))
        self.eval_env = BaseAlgorithm._wrap_env(maybe_make_env(eval_env, 1), 1, True)
        self.policy = policy
        self.actors = [None for _ in range(option_num)]
        self.log_path = log_path
        self.option_num = option_num
        self.actors = None
        self.critics = None
        self.dfa = get_dfa(dfa_text)
        self.upper_model_ps = upper_model_ps
        self.upper_policy = None
        self.critic = None
        self.min_length = 99999
        self.dfa_trace = [set() for _ in range(option_num)]
        self.stop = False

    def predict(
            self,
            observation: Union[np.ndarray, Dict[str, np.ndarray]],
            state: Optional[Tuple[np.ndarray, ...]] = None,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False,
            index: int = 0,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:

        observation, vectorized_env = self.policy.obs_to_tensor(observation)

        with th.no_grad():
            actions = self.actors[index](observation, deterministic)
        # Convert to numpy, and reshape to the original action shape
        actions = actions.cpu().numpy().reshape((-1, *self.policy.action_space.shape))

        if isinstance(self.policy.action_space, spaces.Box):
            if self.policy.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.policy.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.policy.action_space.low, self.policy.action_space.high)

        # Remove batch dimension if needed
        if not vectorized_env:
            actions = actions.squeeze(axis=0)

        return actions, state

    def get_lower_model_index(self, obs, deterministic=False):
        dfa_state = '@q' + str(obs['ds'].item() + 1)
        next_edge, best_option = self.upper_policy.predict(dfa_state, deterministic=deterministic)
        self.dfa_trace[best_option].add(dfa_state)
        return best_option, [next_edge[1]]

    def evaluate(self, training_time):
        trace = ""
        self.upper_policy = ray.get(self.upper_model_ps.get_policy.remote())
        self.actors = ray.get(self.actor_ps.get_networks.remote())
        # print('evaluate', self.actors[0].mu.bias)
        length = 0
        reward = 0
        for _ in range(self.eval_time):
            done = False
            obs = self.eval_env.reset()
            while not done:
                index, end_states = self.get_lower_model_index(obs)
                trace += str(index)
                while not done:
                    action, _ = self.predict(obs, deterministic=False, index=index)
                    # print(action)
                    new_obs, r, done, info = self.eval_env.step(actions=action)
                    # print(new_obs['ds'], obs['ds'])
                    length += 1

                    # reward shaping
                    dfa_state = '@q' + str(obs['ds'].item() + 1)
                    next_dfa_state = '@q' + str(new_obs['ds'].item() + 1)

                    obs = new_obs
                    if done and r < 1:
                        reward += 0
                    if done and r > 1:
                        reward = 100
                    elif next_dfa_state not in end_states and next_dfa_state != dfa_state:
                        reward += 10
                        break
                    elif next_dfa_state in end_states:
                        reward += 10
                        break
        # set log
        seconds = time.time() - self.time_start
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        second = int(seconds % 60)
        hour = str(hours) + "h" + str(minutes) + "min" + str(second) + 's'
        training_steps = ray.get(self.actor_ps.get_update_times.remote())
        data = [
            [training_steps * 300, hour, reward / self.eval_time, length / self.eval_time]
        ]
        with open(self.log_path, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(data)
        print('time:', hours, "h", minutes, "min", second, 's')
        print('eval_reward:', reward / self.eval_time, 'eval_length:', length / self.eval_time)
        print(trace, self.dfa_trace)
        self.reset_ps.set_dfa_trace.remote(self.dfa_trace)
        self.dfa_trace = [set() for _ in range(self.option_num)]

        if seconds > training_time:
            self.stop = True
            return True, False
        else:
            return False, False

    def select_training_stop(self):
        return self.stop
