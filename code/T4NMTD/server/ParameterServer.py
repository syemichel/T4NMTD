import collections
import copy
import random
import sys
import time
from collections import Counter

class NetworkServer:
    def __init__(self, net, option_num):
        self.nets = [copy.deepcopy(net) for _ in range(option_num)]
        self.update_times = 0

    def get_network(self, option_index):
        return self.nets[option_index]

    def set_network(self, option_index, net):
        self.nets[option_index] = net
        self.update_times += 1

    def set_networks(self, nets, map=None):
        if map is None:
            self.nets = nets
        else:
            for i, net in enumerate(self.nets):
                mapped_index = map.get(str(i))
                if mapped_index is not None:
                    self.nets[i] = nets[mapped_index]

    def get_networks(self):
        return self.nets

    def get_update_times(self):
        return self.update_times

class ResetStatesServer:
    def __init__(self, state_index_list):
        self.option_to_dfa = state_index_list
        self.reset_states = [{} for _ in range(len(state_index_list))]
        self.dfa_trace = None
        self.upper_loggers = [None for _ in range(len(state_index_list))]
        self.reset_states1 = [collections.deque(maxlen=100) for _ in range(len(state_index_list))]

    def set_dfa_trace(self, dfa_trace):
        self.dfa_trace = dfa_trace

    def set_upper_logger(self, upper_logger, option_num):
        self.upper_loggers[option_num] = upper_logger
        '''print('hello', upper_logger)
        self.a = False'''

    # 根据概率选取键
    def select_key(self, probabilities):
        # 计算累积概率
        cumulative_prob = 0.0
        for key, prob in probabilities.items():
            cumulative_prob += prob
            if random.random() < cumulative_prob:
                return key

    def get_state(self, option_index):
        if len(self.reset_states1[option_index]) != 0:
            # epsilon greedy
            if random.random() < 0.8 and self.dfa_trace is not None and self.upper_loggers[option_index] is not None:
                dic = self.upper_loggers[option_index]
                sum_prob = 0
                for key, value in dic.items():
                    dic[key] = (1 - value + 0.001) * (int(key in self.dfa_trace[option_index]) + 0.001)
                    sum_prob += (1 - value + 0.001) * (int(key in self.dfa_trace[option_index]) + 0.001)
                for key, value in dic.items():
                    dic[key] = value / sum_prob
                dfa_state = self.select_key(dic)
                '''if not self.a:
                    print(dic)
                    self.a = True'''
                try:
                    return random.choice(self.reset_states[option_index][dfa_state])
                except Exception as e:
                    print(e)
                    print(option_index)
                    print(len(self.reset_states[option_index][dfa_state]))
                    time.sleep(1000)
            else:
                return random.choice(self.reset_states1[option_index])
        else:
            return None

    def set_states(self, initial_infos):
        for i, infos in enumerate(initial_infos):
            if not infos:
                continue
            for info in infos:
                dfa_state, info_ = info
                try:
                    self.reset_states[i].setdefault(dfa_state, collections.deque(maxlen=100)).append(info_)
                except Exception as e:
                    print(e)
                    print(i, dfa_state)
                    print('error in ResetStatesServer!!!')
                    time.sleep(1000)
                self.reset_states1[i].append(info_)


class ExperienceServer:
    def __init__(self):
        self.exp_buffer = []

    def get_states(self):
        exp_buffer = self.exp_buffer
        self.exp_buffer = []
        return exp_buffer

    def set_states(self, exps):
        self.exp_buffer.extend(exps)