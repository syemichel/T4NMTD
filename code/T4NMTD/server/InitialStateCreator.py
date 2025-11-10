import copy
import sys
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.base_class import BaseAlgorithm, maybe_make_env
from stable_baselines3.common.utils import obs_as_tensor

sys.setrecursionlimit(1000000)
import csv
import torch as th
import time
import ray
from util.DFA import *
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

class InitialStateCreator:
    def __init__(self, option_num, initial_state, time_interval, store_interval, upper_policy_ps, dfa_text, reset_ps, env):
        self.option_num = option_num
        self.initial_state = initial_state.copy()
        self.time_interval = time_interval
        self.store_interval = store_interval
        self.dfa = get_dfa(dfa_text)
        self.reset_ps = reset_ps
        self.now_time = time.time()
        self.initial_infos = [[] for _ in range(self.option_num)]
        self.upper_policy_ps = upper_policy_ps
        self.upper_policy = None
        self.dfa_state = '@q' + str(self.initial_state['ds'].item() + 1)
        self.env = env


    def get_lower_model_index(self, obs, deterministic=False):
        dfa_state = '@q' + str(obs['ds'].item() + 1)
        next_edge, best_option = self.upper_policy.predict(dfa_state, deterministic=False)
        return best_option, [next_edge[1]]

    def run(self):
        self.upper_policy = ray.get(self.upper_policy_ps.get_policy.remote())
        while True:
            if time.time() - self.now_time > self.time_interval:
                time.sleep(0.1)
                index, end_states = self.get_lower_model_index(self.initial_state)
                self.initial_infos[index].append((self.dfa_state, [self.initial_state, end_states]))

            if time.time() - self.now_time > self.store_interval:
                self.reset_ps.set_states.remote(self.initial_infos)
                self.upper_policy = ray.get(self.upper_policy_ps.get_policy.remote())
                self.now_time = time.time()

