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
from stable_baselines3.common.utils import should_collect_more_steps, polyak_update, get_parameters_by_name, \
    obs_as_tensor
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

class SACTestAgent:
    def __init__(self, policy_ps, upper_policy_ps, reset_ps, log_path, eval_env, option_num, policy, dfa_text, distances):
        self.policy_ps = policy_ps
        self.reset_ps = reset_ps
        self.eval_time = 10
        self.time_start = time.time()
        # eval_env = MinigridActionEnv(AddDFAStateObs(MiniGridEnv1(render_mode='human')))
        self.eval_env = BaseAlgorithm._wrap_env(maybe_make_env(eval_env, 1), 1, True)
        self.policy = policy
        self.policies = [None for _ in range(option_num)]
        self.log_path = log_path
        self.option_num = option_num
        self.actors = None
        self.critics = None
        self.dfa = get_dfa(dfa_text)
        self.upper_policy_ps = upper_policy_ps
        self.upper_policy = None
        self.critic = None
        self.min_length = 99999
        self.dfa_trace = [set() for _ in range(option_num)]
        self.stop = False
        self.distances = distances

    def get_lower_model_index(self, obs, deterministic=False):
        dfa_state = '@q' + str(obs['ds'].item() + 1)
        next_edge, best_option = self.upper_policy.predict(dfa_state, deterministic=deterministic)
        self.dfa_trace[best_option].add(dfa_state)
        return best_option, [next_edge[1]]

    def evaluate(self, training_time):
        trace = ""
        self.upper_policy = ray.get(self.upper_policy_ps.get_policy.remote())
        self.policies = ray.get(self.policy_ps.get_networks.remote())

        reward = ray.get(self.reset_ps.calculate_expected_rewards.remote())

        # set log
        seconds = time.time() - self.time_start
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        second = int(seconds % 60)
        hour = str(hours) + "h" + str(minutes) + "min" + str(second) + 's'
        training_steps = ray.get(self.policy_ps.get_update_times.remote())
        data = [
            [training_steps * 300, hour, reward, 0]
        ]
        with open(self.log_path, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(data)
        print('time:', hours, "h", minutes, "min", second, 's')
        print('eval_reward:', reward, 'eval_length:', 0)
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
