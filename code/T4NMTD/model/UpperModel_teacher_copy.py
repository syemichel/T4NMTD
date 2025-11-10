import random
import time

from util.DFA import *
import networkx as nx
import random
from typing import Dict, List

dfa_text = '''
if(ds==@q1 ^ (~p1 & ~p2 & ~p3)) then @q1
else if(ds==@q1 ^ (p3 & ~p1 & ~p2)) then @q3
else if(ds==@q1 ^ (p2 & ~p1 & ~p3)) then @q4
else if(ds==@q1 ^ ((p1 & p2) | (p1 & p3) | (p2 & p3))) then @q2
else if(ds==@q1 ^ (p1 & ~p2 & ~p3)) then @q5
else if(ds==@q3 ^ (~p1 & ~p2 & ~p3)) then @q1
else if(ds==@q3 ^ (p3 & ~p1 & ~p2 & ~p4)) then @q3
else if(ds==@q3 ^ (p2 & ~p1 & ~p3)) then @q4
else if(ds==@q3 ^ ((p1 & p2) | (p1 & p3) | (p2 & p3))) then @q2
else if(ds==@q3 ^ (p4 & ~p1 & ~p2)) then @q6
else if(ds==@q3 ^ (p1 & ~p2 & ~p3)) then @q5
else if(ds==@q4 ^ (~p1 & ~p2 & ~p3)) then @q1
else if(ds==@q4 ^ (p3 & ~p1 & ~p2)) then @q3
else if(ds==@q4 ^ (p2 & ~p1 & ~p3 & ~p4)) then @q4
else if(ds==@q4 ^ ((p1 & p2) | (p1 & p3) | (p2 & p3))) then @q2
else if(ds==@q4 ^ (p4 & ~p1 & ~p3)) then @q6
else if(ds==@q4 ^ (p1 & ~p2 & ~p3)) then @q5
else if(ds==@q2 ^ (true)) then @q2
else if(ds==@q5 ^ (~p1 & ~p2 & ~p3)) then @q1
else if(ds==@q5 ^ (p3 & ~p1 & ~p2)) then @q3
else if(ds==@q5 ^ (p2 & ~p1 & ~p3)) then @q4
else if(ds==@q5 ^ ((p1 & p2) | (p1 & p3) | (p2 & p3))) then @q2
else if(ds==@q5 ^ (p1 & ~p2 & ~p3 & ~p4)) then @q5
else if(ds==@q5 ^ (p4 & ~p2 & ~p3)) then @q6
else if(ds==@q6 ^ ((~p1 & ~p2) | (~p1 & ~p3) | (~p2 & ~p3))) then @q6
else if(ds==@q6 ^ ((p1 & p2) | (p1 & p3) | (p2 & p3))) then @q2
'''

class DFAPathFinder:
    def __init__(self, dfa: nx.DiGraph, epsilon: float = 0.1, option_num=4):
        self.dfa = dfa
        self.accepting_state = '@q' + str(dfa.number_of_nodes())
        self.epsilon = epsilon
        # 设置默认边奖励为字典,每个option都是-9999
        for edge in self.dfa.edges():
            if 'rewards' not in self.dfa.edges[edge]:
                self.dfa.edges[edge]['rewards'] = {i: -9999 for i in range(option_num)}  #
                self.dfa.edges[edge]['best_option'] = 0  # 记录最优的option

        # 检查是否存在正向环路
        if self._has_positive_cycle():
            raise ValueError("DFA中存在正向环路，可能导致无限累积奖励！")

        # 预计算所有状态到接受状态的最优路径
        self.optimal_paths = self._compute_all_optimal_paths()

    def set_edges_reward(self, rewards_dict: Dict[tuple, float], option: int):
        """
        通过字典一次性设置多个边的奖励值
        rewards_dict: 形如 {('@q1', '@q3'): 10, ('@q1', '@q4'): 5} 的字典
        或者接收元组列表 [('@q1', '@q3', 10), ('@q1', '@q4', 5)]
        option: 0-3之间的整数，表示哪个agent的奖励信息
        """

        # 如果输入是元组列表，转换为字典
        if isinstance(rewards_dict, list):
            rewards_dict = {(edge[0], edge[1]): edge[2] for edge in rewards_dict}

        # 保存旧的奖励值，以便在出现错误时恢复
        old_rewards = {}
        for (from_state, to_state), reward in rewards_dict.items():
            if self.dfa.has_edge(from_state, to_state):
                old_rewards[(from_state, to_state)] = dict(self.dfa.edges[from_state, to_state]['rewards'])
                # 更新该option的奖励值
                self.dfa.edges[from_state, to_state]['rewards'][option] = reward
                # 更新最优option
                best_reward = max(self.dfa.edges[from_state, to_state]['rewards'].values())
                best_options = [opt for opt, rew in self.dfa.edges[from_state, to_state]['rewards'].items()
                              if rew == best_reward]
                self.dfa.edges[from_state, to_state]['best_option'] = best_options[0]

        # 检查是否会导致正向环路
        try:
            if self._has_positive_cycle():
                # 如果检测到正向环路，恢复所有的奖励值
                for (from_state, to_state), old_reward in old_rewards.items():
                    self.dfa.edges[from_state, to_state]['rewards'] = old_reward
                raise ValueError("设置的奖励值会导致正向环路！")
            # 重新计算最优路径
            self.optimal_paths = self._compute_all_optimal_paths()
        except Exception as e:
            # 如果出现任何错误，恢复所有的奖励值
            for (from_state, to_state), old_reward in old_rewards.items():
                self.dfa.edges[from_state, to_state]['rewards'] = old_reward
            raise e

    def _get_edge_reward(self, from_state: str, to_state: str) -> tuple[float, int]:
        """获取边的最大奖励值和对应的option"""
        rewards = self.dfa.edges[from_state, to_state]['rewards']
        best_reward = max(rewards.values())
        best_option = self.dfa.edges[from_state, to_state]['best_option']
        return best_reward, best_option

    def _has_positive_cycle(self) -> bool:
        """
        使用Bellman-Ford算法检测是否存在正向环路
        """
        # 创建一个负权重图（将原始奖励取反）
        negative_graph = self.dfa.copy()
        for u, v in self.dfa.edges():
            max_reward, _ = self._get_edge_reward(u, v)
            negative_graph.edges[u, v]['weight'] = -max_reward

        # 选择任意节点作为源节点
        source = list(self.dfa.nodes())[0]

        # 初始化距离
        distance = {node: float('inf') for node in negative_graph.nodes()}
        distance[source] = 0

        # Bellman-Ford算法
        for _ in range(len(negative_graph.nodes()) - 1):
            for u, v in negative_graph.edges():
                weight = negative_graph.edges[u, v]['weight']
                if distance[u] != float('inf') and distance[u] + weight < distance[v]:
                    distance[v] = distance[u] + weight

        # 检查是否有负环（原图中的正环）
        for u, v in negative_graph.edges():
            weight = negative_graph.edges[u, v]['weight']
            if distance[u] != float('inf') and distance[u] + weight < distance[v]:
                return True

        return False

    def _compute_optimal_path(self, start_state: str) -> tuple[float, List[List[str]]]:
        """
        计算从起始状态到接受状态的所有最优路径
        返回: (总奖励, 所有最优路径的列表)
        """
        rewards = {state: float('-inf') for state in self.dfa.nodes()}
        rewards[start_state] = 0
        # 记录所有可能的前驱节点
        predecessors = {state: [] for state in self.dfa.nodes()}
        visited = set()
        visit_count = {state: 0 for state in self.dfa.nodes()}

        def get_max_reward_state():
            max_reward = float('-inf')
            max_states = []
            for state in self.dfa.nodes():
                if state not in visited:
                    if rewards[state] > max_reward:
                        max_reward = rewards[state]
                        max_states = [state]
                    elif rewards[state] == max_reward:
                        max_states.append(state)
            return max_states[0] if max_states else None  # 这里不需要随机选择，因为我们会探索所有路径

        while True:
            current = get_max_reward_state()
            if current is None:
                break

            visited.add(current)
            visit_count[current] += 1

            if visit_count[current] > len(self.dfa.nodes()):
                raise ValueError(f"检测到可能的环路，状态 {current} 被多次访问")

            for _, next_state in self.dfa.out_edges(current):
                if next_state == current or next_state == '@q2':  # 跳过自循环和通往@q2的边
                    continue
                edge_reward, _ = self._get_edge_reward(current, next_state)
                new_reward = rewards[current] + edge_reward

                if new_reward > rewards[next_state]:
                    rewards[next_state] = new_reward
                    predecessors[next_state] = [(current, edge_reward)]
                elif new_reward == rewards[next_state]:
                    predecessors[next_state].append((current, edge_reward))

        if rewards[self.accepting_state] == float('-inf'):
            return float('-inf'), []

        # 使用DFS收集所有最优路径
        def collect_all_paths(current, visited=None):
            if visited is None:
                visited = set()

            if current in visited:
                return []

            if current == start_state:
                return [[start_state]]

            visited.add(current)
            all_paths = []

            for prev, _ in predecessors[current]:
                for path in collect_all_paths(prev, visited.copy()):
                    all_paths.append(path + [current])

            return all_paths

        # 收集所有到达接受状态的最优路径
        all_optimal_paths = collect_all_paths(self.accepting_state)

        return rewards[self.accepting_state], all_optimal_paths

    def _compute_all_optimal_paths(self) -> Dict[str, tuple[float, List[List[str]]]]:
        """
        计算从每个状态到接受状态的所有最优路径
        """
        optimal_paths = {}
        for state in self.dfa.nodes():
            optimal_paths[state] = self._compute_optimal_path(state)
        return optimal_paths

    def get_optimal_next_state(self, current_state: str):
        """
        根据最优路径获取下一个状态
        """
        total_reward, path = self.optimal_paths[current_state]

        if not path or len(path) < 2:
            return None, 0

        if current_state in path:
            current_index = path.index(current_state)
            if current_index < len(path) - 1:
                next_state = path[current_index + 1]
                edge_reward, best_option = self._get_edge_reward(current_state, next_state)
                return next_state, edge_reward

        return None, 0

    def predict(self, current_state: str, deterministic=False):
        """
        epsilon-贪婪算法：有概率随机选择一条边，否则选择奖励最大的边
        返回: (selected_edge, best_option)
        """
        if not deterministic:
            epsilon = self.epsilon
        else:
            epsilon = 0

        def get_valid_options(formula):
            if formula == 'True' or formula == 'true':
                return list(range(4))
            positive = extract_true_predicates(formula)
            if not positive:
                return []  # 如果没有正命题，返回空列表
            return [int(p[1]) - 1 for p in positive]

        # 首先过滤出有效的边（排除@q2和自循环，以及没有正命题的边）
        valid_edges = []
        for edge in self.dfa.out_edges(current_state, data=True):
            if edge[1] != '@q2' and edge[0] != edge[1]:
                valid_options = get_valid_options(edge[2]['formula'])
                if valid_options:  # 只有存在有效options的边才会被考虑
                    valid_edges.append(edge)

        if not valid_edges:
            return None, None

        total_reward, optimal_paths = self.optimal_paths[current_state]

        if random.random() < epsilon:
            selected_edge = random.choice(valid_edges)
            formula = selected_edge[2]['formula']
            valid_options = get_valid_options(formula)

            valid_rewards = {opt: selected_edge[2]['rewards'][opt]
                             for opt in valid_options}
            max_reward = max(valid_rewards.values())
            best_options = [opt for opt, rew in valid_rewards.items()
                            if rew == max_reward]
            best_option = random.choice(best_options)
            return selected_edge, best_option
        else:
            # 从所有最优路径中随机选择一条
            if optimal_paths:
                selected_path = random.choice(optimal_paths)
                if len(selected_path) > 1:
                    current_index = selected_path.index(current_state)
                    if current_index < len(selected_path) - 1:
                        next_state = selected_path[current_index + 1]
                        # 找到对应的边
                        for edge in valid_edges:  # 使用valid_edges而不是out_edges
                            if edge[1] == next_state:
                                formula = edge[2]['formula']
                                valid_options = get_valid_options(formula)
                                valid_rewards = {opt: edge[2]['rewards'][opt]
                                                 for opt in valid_options}
                                max_reward = max(valid_rewards.values())
                                best_options = [opt for opt, rew in valid_rewards.items()
                                                if rew == max_reward]
                                return edge, random.choice(best_options)

            # 如果没有找到有效的最优路径，选择奖励最大的边
            best_reward = float('-inf')
            best_edges_and_options = []

            for edge in valid_edges:  # 使用valid_edges而不是out_edges
                formula = edge[2]['formula']
                valid_options = get_valid_options(formula)
                valid_rewards = {opt: edge[2]['rewards'][opt]
                                 for opt in valid_options}

                edge_max_reward = max(valid_rewards.values())
                if edge_max_reward > best_reward:
                    best_reward = edge_max_reward
                    best_edges_and_options = [(edge, opt)
                                              for opt, rew in valid_rewards.items()
                                              if rew == edge_max_reward]
                elif edge_max_reward == best_reward:
                    best_edges_and_options.extend([
                        (edge, opt) for opt, rew in valid_rewards.items()
                        if rew == edge_max_reward
                    ])

            if not best_edges_and_options:
                return None, None

            best_edge, best_option = random.choice(best_edges_and_options)
            return best_edge, best_option

    def print_all_edges_rewards(self):
        """打印所有边的奖励信息"""
        print("\n所有边的奖励信息:")
        print("=" * 60)
        for (u, v, data) in self.dfa.edges(data=True):
            print(f"边 {u} -> {v}:")
            for option, reward in data['rewards'].items():
                print(f"  Option {option}: {reward}")
            print(f"  最优Option: {data['best_option']}")
            print("-" * 30)

    def get_policy(self):
        return self

if __name__ == '__main__':
    dfa = get_dfa(dfa_text)
    path_finder = DFAPathFinder(dfa, epsilon=1)

    '''rewards = {
        ('@q1', '@q3'): -10,
        ('@q1', '@q4'): -5,
        ('@q1', '@q5'): -6,
        ('@q4', '@q6'): -6
    }
    path_finder.set_edges_reward(rewards, option=1)

    rewards = {
        ('@q1', '@q3'): -11,
        ('@q1', '@q4'): -7,
        ('@q1', '@q5'): -8,
        ('@q4', '@q6'): -9
    }
    path_finder.set_edges_reward(rewards, option=0)'''

    # 执行epsilon-greedy步
    while True:
        next_edge, best_option = path_finder.predict('@q5')
        print(next_edge, best_option)