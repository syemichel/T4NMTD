import argparse
import time
from collections import defaultdict
import re
import sympy as sp
import networkx as nx
import copy
from collections import OrderedDict
import gymnasium as gym
import gymnasium.spaces as spaces
from gymnasium.spaces import *
from abc import ABCMeta, abstractmethod
import random
import numpy as np
from collections import deque

class BaseAgent(metaclass=ABCMeta):

    @abstractmethod
    def take_action(self, state):
        pass

class UpperFlattenAction(gym.ActionWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.action_space = spaces.flatten_space(env.action_space)

    def action(self, act):
        return act
class Agent(BaseAgent):
    def __init__(self, action_space, num_actions=1):
        self.action_space = action_space
        self.num_actions = num_actions

    def take_action(self, s, state=None):
        action = {}
        selected_actions = random.sample(list(s), self.num_actions)  # problem!!!!!!!
        for sample in selected_actions:
            if isinstance(self.action_space[sample], gym.spaces.Box):
                action[sample] = s[sample][0].item()
            elif isinstance(self.action_space[sample], gym.spaces.Discrete):
                action[sample] = s[sample]
        return action

class UpperDiscreteEnvOverOne(gym.ActionWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.action_space = Box(-0.5, 0.4999, (1, ), dtype=np.float32)

        print(self.action_space)

    def action(self, act):
        return {'select': act}

class FlattenAction(gym.ActionWrapper):

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.action_space = spaces.flatten_space(env.action_space)

        print(self.action_space)

    def action(self, act):
        act = spaces.unflatten(self.env.action_space, act)
        agent = Agent(action_space=self.env.action_space, num_actions=self.env.numConcurrentActions)
        return agent.take_action(act)

class UpperDiscreteEnv(gym.ActionWrapper):
    def __init__(self, env: gym.Env, option_num):
        super().__init__(env)
        self.action_space = Box(0, option_num-1-0.001, (1, ), dtype=np.float32)
        print(self.action_space)

    def action(self, act):
        return {'select': act}

class UpperRMEnv(gym.ActionWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.action_space = Box(0, 0.999, (1, ), dtype=np.float32)
        print(self.action_space)

    def action(self, act):
        return {'select': act}


class BoxContinuousAction(gym.ActionWrapper):

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.l = []
        for k, v in self.action_space.items():
            self.l.append(k)
        self.new_action_space= copy.deepcopy(self.env.action_space)

        for i in range(self.env.numConcurrentActions):
            self.new_action_space[str(i)] = Box(0, len(self.env.action_space) - 0.00001, shape=(1, 1))
        self.action_space = spaces.flatten_space(self.new_action_space)

        print(self.action_space)

    def action(self, act):

        act = spaces.unflatten(self.new_action_space, act)
        act_dic = {}
        for i in range(self.env.numConcurrentActions):
            value = int(act[str(i)])
            action = self.l[value]
            action_value = float(act[action])
            act_dic[action] = action_value
        agent = Agent(action_space=self.env.action_space, num_actions=self.env.numConcurrentActions)
        return act_dic

class CartpoleActionWapper(gym.ActionWrapper):

    def __init__(self, env: gym.Env):
        super().__init__(env)

        self.action_space = Discrete(2)
        print(self.action_space)

    def action(self, act):
        dic = {}
        dic['force-side'] = act
        # print(dic)
        return dic
class CartpoleActionWapper2(gym.ActionWrapper):

    def __init__(self, env: gym.Env):
        super().__init__(env)

        self.action_space = Discrete(21)
        print(self.action_space)

    def action(self, act):
        dic = {}
        dic['force'] = act - 10
        # print(dic)
        return dic

class WaterworldActionWapper(gym.ActionWrapper):

    def __init__(self, env: gym.Env):
        super().__init__(env)

        self.action_space = Box(low=-35, high=35, shape=(2,), dtype=int)
        print(self.action_space)

    def action(self, act):
        dic = {}
        dic['ag-move___x'] = act[0] / 10
        dic['ag-move___y'] = act[1] / 10
        # print(dic)
        return dic

class WaterworldActionWapper2(gym.ActionWrapper):

    def __init__(self, env: gym.Env):
        super().__init__(env)

        self.action_space = Discrete(49)
        print(self.action_space)

    def action(self, act):
        actions = np.array([[i, j] for i in range(-3, 4) for j in range(-3, 4)])
        dic = {}
        dic['ag-move___x'] = actions[act][0] / 10
        dic['ag-move___y'] = actions[act][1] / 10
        # print(dic)
        return dic


class BoxDiscreteAction(gym.ActionWrapper):

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.l = []
        for k, v in self.action_space.items():
            self.l.append(k)
        self.action_space = Box(low=0, high=len(self.env.action_space) -1, shape=(self.env.numConcurrentActions,), dtype=np.int)
        print(self.action_space)

    def action(self, act):
        dic = {}
        for i in range(self.env.numConcurrentActions):
            dic[self.l[act[i]]] = 1
        #print(dic)
        return dic

class MultiDiscreteAction(gym.ActionWrapper):
    # support concurrent actions
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.l = []
        for k, v in self.action_space.items():
            self.l.append(k)

        matrix = np.full(self.env.numConcurrentActions, len(self.env.action_space))

        self.action_space = MultiDiscrete(matrix)
        print(self.action_space)

    def action(self, act):
        dic = {}
        for i in range(self.env.numConcurrentActions):
                dic[self.l[act[i]]] = 1
        # print(dic)
        return dic

class FlattenObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = flatten_space(env.observation_space)

    def observation(self, obs):
        return flatten(self.env.observation_space, obs)


def dfs(graph, node, visited, path, all_paths, end_node):
    visited[node] = True
    path.append(node)

    if node == end_node:
        all_paths.append(path.copy())
    elif node in graph:
        for child in graph[node]:
            if not visited[child]:
                dfs(graph, child, visited, path, all_paths, end_node)

    path.pop()
    visited[node] = False

def find_all_paths(graph, start_node, end_node):
    visited = defaultdict(bool)
    path = []
    all_paths = []

    dfs(graph, start_node, visited, path, all_paths, end_node)

    return all_paths

def calculate_avg_distance(graph, end_node):
    avg_distances = {}

    for start_node in graph:
        all_paths = find_all_paths(graph, start_node, end_node)
        total_distance = sum([len(path) - 1 for path in all_paths])
        avg_distance = round(total_distance / len(all_paths), 2) if all_paths else float('inf')
        avg_distances[start_node] = avg_distance

    return avg_distances

def insert_value_to_dict_list(d, key, value):
    # 使用 setdefault 方法，如果键不存在则初始化为空列表
    d.setdefault(key, []).append(value)

def get_graph(dfa):
    graph = {}
    for u, v in dfa.edges:
        insert_value_to_dict_list(graph, u, v)
    return graph

def set_states_potential(text, accepted_state='17', error_state='2'):
    graph = get_dfa(text)
    end_node = accepted_state
    avg_distances = calculate_avg_distance(graph, end_node)
    sorted_avg_distances = sorted(avg_distances.items(), key=lambda x: x[1])

    L_min = 999999
    L_max = 0
    for node, distance in sorted_avg_distances:
        if distance < L_min and distance > 0:
            L_min = distance
        if distance > L_max and distance > 0 and distance < 999999:
            L_max = distance
    potential = {}
    for node, distance in sorted_avg_distances:
        potential[node] = round((L_max - distance) / L_max * 10, 2)
        if potential[node] < 0:
            potential[node] = -10
    # print(potential)
    return potential

class OrderedSet:
    def __init__(self):
        self._data = OrderedDict()

    def add(self, value):
        self._data[value] = None

    def __iter__(self):
        return iter(self._data.keys())

    def __contains__(self, value):
        return value in self._data

    def __len__(self):
        return len(self._data)

def parse_logical_expression(expression):
    # 定义逻辑变量
    p1, p2, p3, p4, p5, p6 = sp.symbols('p1 p2 p3 p4 p5 p6')

    # 将字符串表达式转换为sympy的表达式
    parsed_expression = sp.sympify(expression)

    return parsed_expression

def extract_predicates(text):
    # 匹配谓词的正则表达式，寻找 ^ 和 ) 之间的内容
    pattern = re.compile(r'if\s*\(ds\s*==\s*(@q\d+)\s*\^\s*\((.*?)\)\s*\)\s*then\s*(@q\d+)')
    matches = pattern.findall(text)
    predicates = [(match[1], match[0], match[2]) for match in matches]
    return predicates

# if conflict return False
def identify_conflicts(pre_expression, expression):
    parsed_expression = parse_logical_expression(expression)
    p = pre_expression & parsed_expression
    return p.simplify()

def classify_predicates(text, error_states=['@q2']):
    # 提取谓词
    predicates = extract_predicates(text)

    # 使用集合来存储唯一的谓词
    unique_predicates = OrderedSet()
    # 输出结果，排除自循环边的谓词
    for predicate, start_state, end_state in predicates:
        if start_state != end_state and predicate not in unique_predicates and end_state not in error_states:
            unique_predicates.add(predicate)
    class_edge = [[]]
    pre_props = [True]


    for i in unique_predicates:
        if_append = False

        # 在遍历之前先打乱class_edge的顺序
        # random.shuffle(class_edge)

        combined = list(zip(class_edge, pre_props))
        combined.sort(key=lambda x: len(x[0]))
        class_edge, pre_props = zip(*combined)
        class_edge = list(class_edge)
        pre_props = list(pre_props)

        for j in range(len(class_edge)):
            p = identify_conflicts(pre_props[j], i)
            if p != False:
                class_edge[j].append(i)
                pre_props[j] = p
                if_append = True
                break
        if not if_append:
            class_edge.append([i])
            pre_props.append(parse_logical_expression(i))
    return class_edge

def find_element_index(nested_list, target):
    for index, sublist in enumerate(nested_list):
        if target in sublist:
            return index
    return -1  # 如果没有找到，返回 -1

def get_dfa(text):
    G = nx.DiGraph()
    # 提取谓词
    predicates = extract_predicates(text)

    # 使用集合来存储唯一的谓词
    unique_predicates = OrderedSet()
    error_states = ['@q2']
    # 输出结果，排除自循环边的谓词
    for predicate, start_state, end_state in predicates:
        G.add_node(start_state)
        G.add_node(end_state)
        G.add_edge(start_state, end_state, formula=predicate)
    return G


def extract_true_predicates(proposition):
    # 匹配未被否定的原子命题，使用负向回顾确保前面没有~符号
    positive = re.findall(r'(?<!~)p\d+', proposition)
    return positive

class TrainingLogger:
    def __init__(self, log_interval=100):
        self.episode_rewards = deque(maxlen=log_interval)
        self.episode_lengths = deque(maxlen=log_interval)
        self.episode_rewards.append(0)
        self.episode_lengths.append(0)

    def record(self, reward, done):
        if reward >= 0:
            self.episode_rewards[-1] += reward.item()
        self.episode_lengths[-1] += 1
        if done:
            self.episode_rewards.append(0)
            self.episode_lengths.append(0)

    def get_info(self, num_timesteps):
        print('ep_rew_mean:', sum(self.episode_rewards) / len(self.episode_rewards), 'ep_len_mean:', sum(self.episode_lengths) / len(self.episode_lengths),
              'num_timesteps:', num_timesteps)

class DFATransformer:
    def __init__(self, dfa_text):
        self.dfa = get_dfa(dfa_text)
        self.dfa_state = '@q1'
        self.accepting_state = "@q" + str(self.dfa.number_of_nodes())
        self.error_state = "@q2"

    def reset(self):
        self.dfa_state = '@q1'

    def evaluate_logic_formula(self, props:Dict, formula):
        # 替换公式中的命题名称为对应的布尔值

        for var, value in props.items():
            formula = re.sub(r'\b' + re.escape(var) + r'\b', str(value), formula)
        formula = formula.replace('~', 'not ').replace('&', 'and ').replace('|', 'or ')
        # 评估公式
        try:
            result = eval(formula)
            return result
        except Exception as e:
            print(f"评估公式时出错: {e}")
            return None

    # return terminate, if_success and if_failure
    def step(self, props):
        out_edges = self.dfa.out_edges(str(self.dfa_state), data=True)
        for edge in out_edges:
            if self.evaluate_logic_formula(props, formula=edge[2]['formula']):
                self.dfa_state = edge[1]
                break
        if self.dfa_state == self.error_state:
            return True, False, True
        if self.dfa_state == self.accepting_state:
            return True, True, False
        return False, False, False

def dfa_step(dfa, dfa_state, props):
    out_edges = dfa.out_edges(str(dfa_state), data=True)
    for edge in out_edges:
        if evaluate_logic_formula(props, formula=edge[2]['formula']):
            dfa_state = edge[1]
            break
    return dfa_state

def evaluate_logic_formula(props:Dict, formula):
    # 替换公式中的命题名称为对应的布尔值
    formula1 = formula
    for var, value in props.items():
        formula = re.sub(r'\b' + re.escape(var) + r'\b', str(value), formula)
    formula = formula.replace('~', ' not ').replace('&', ' and ').replace('|', ' or ').replace('^', ' and ')
    # 评估公式
    try:
        result = eval(formula)
        return result
    except Exception as e:
        print(f"评估公式时出错: {e}")
        return None

'''if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-log', type=str, default='task1.csv', help='log path')
    parser.add_argument('-i', type=str, default='inst31', help='inst name')
    parser.add_argument('-r', type=str, default='waterworld3', help='inst name')
    parser.add_argument('-o', type=int, default=8, help='process num')
    parser.add_argument('-t', type=int, default=5000, help='training time')
    args = parser.parse_args()

    training_time = args.t
    option_num = args.o
    upper_domain = 'high_level_benchmarks/waterworld/' + args.r + '.rddl'
    lower_domain = 'low_level_benchmarks/waterworld/' + args.r + '.rddl'
    instance = 'low_level_benchmarks/waterworld/' + args.i + '.rddl'
    text_path = 'dfa_text/waterworld/' + args.r + '.txt'
    with open(text_path, 'r', encoding='utf-8') as file:
        dfa_text = file.read()
    dfa = DFATransformer(dfa_text)
    print(dfa.error_state, dfa.accepting_state, dfa.dfa_state)
    dic = {'p1': True, 'p2': False, 'p3': False, 'p4': False, 'p5': False, 'p6': False, 'p7': False, 'p8': False}
    dfa.step(dic)
    print(dfa.dfa_state)'''