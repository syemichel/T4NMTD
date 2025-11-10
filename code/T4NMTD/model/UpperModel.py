import random
import time

from util.DFA import *
import networkx as nx
import random
from typing import Dict, List

dfa_text = '''
if(ds==@q1 ^ (~p2 & ~p4)) then @q1
else if(ds==@q1 ^ (p4)) then @q3
else if(ds==@q1 ^ (p2 & ~p4)) then @q2
else if(ds==@q3 ^ (~p3 & ~p5)) then @q3
else if(ds==@q3 ^ (p5 & ~p3)) then @q2
else if(ds==@q3 ^ (p3)) then @q4
else if(ds==@q2 ^ (True)) then @q2
else if(ds==@q4 ^ (~p2 & ~p4)) then @q4
else if(ds==@q4 ^ (p4 & ~p2)) then @q2
else if(ds==@q4 ^ (p2)) then @q5
else if(ds==@q5 ^ (~p1 & ~p3)) then @q5
else if(ds==@q5 ^ (p1 & ~p3)) then @q2
else if(ds==@q5 ^ (p3)) then @q6
else if(ds==@q6 ^ (True)) then @q6
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
        优化的epsilon-贪婪算法，保证总是返回一条边
        返回: (selected_edge, best_option)
        """
        epsilon = 0 if deterministic else self.epsilon

        def get_self_loop_formula(state):
            """获取状态的自循环边的formula"""
            for _, next_state, data in self.dfa.out_edges(state, data=True):
                if next_state == state:
                    return data['formula']
            return ""

        def get_valid_options(formula, self_loop_formula=""):
            """获取有效的options"""
            if formula == 'True' or formula == 'true':
                return list(range(4))
            positive = extract_true_predicates(formula, self_loop_formula)
            if not positive:
                return []
            return [int(p[1]) - 1 for p in positive]

        def get_edge_score(edge, valid_options):
            """计算边的得分，用于排序"""
            rewards = edge[2]['rewards']
            valid_rewards = [rewards[opt] for opt in valid_options]
            max_reward = max(valid_rewards)

            # 优先级：非-9999奖励 > -9999奖励，然后按奖励值排序
            if max_reward != -9999:
                return (1, max_reward)  # 高优先级
            else:
                return (0, max_reward)  # 低优先级

        # 获取自循环formula
        self_loop_formula = get_self_loop_formula(current_state)

        # 获取所有出边并按优先级排序
        all_edges = list(self.dfa.out_edges(current_state, data=True))

        # 为每条边计算有效选项和得分
        edge_info = []
        for edge in all_edges:
            valid_options = get_valid_options(edge[2]['formula'], self_loop_formula)
            if valid_options:  # 只考虑有有效选项的边
                score = get_edge_score(edge, valid_options)
                edge_info.append((edge, valid_options, score))

        # 如果没有任何有效边，强制选择第一条边
        if not edge_info:
            if all_edges:
                edge = all_edges[0]
                # 强制使用所有选项
                return edge, 0
            else:
                raise ValueError(f"状态 {current_state} 没有任何出边！")

        # 按得分排序（优先级高的在前，同优先级按奖励值降序）
        edge_info.sort(key=lambda x: x[2], reverse=True)

        # 分离高优先级边（有非-9999奖励）和低优先级边
        high_priority_edges = [(edge, opts) for edge, opts, (priority, _) in edge_info if priority == 1]
        low_priority_edges = [(edge, opts) for edge, opts, (priority, _) in edge_info if priority == 0]

        # 优先从高优先级边中选择
        candidate_edges = high_priority_edges if high_priority_edges else low_priority_edges

        # epsilon-greedy决策
        if random.random() < epsilon:
            # 随机选择
            selected_edge, valid_options = random.choice(candidate_edges)
        else:
            # 贪婪选择：优先选择最优路径上的边，否则选择得分最高的边
            total_reward, optimal_paths = self.optimal_paths[current_state]

            selected_edge, valid_options = None, None

            # 尝试从最优路径中选择
            if total_reward != float('-inf') and optimal_paths:
                selected_path = random.choice(optimal_paths)
                if len(selected_path) > 1:
                    current_index = selected_path.index(current_state)
                    if current_index < len(selected_path) - 1:
                        next_state = selected_path[current_index + 1]
                        # 在候选边中找到通往next_state的边
                        for edge, opts in candidate_edges:
                            if edge[1] == next_state:
                                selected_edge, valid_options = edge, opts
                                break

            # 如果最优路径不可行，选择得分最高的边
            if selected_edge is None:
                selected_edge, valid_options = candidate_edges[0]  # 已经按得分排序

        # 选择最佳option
        rewards = selected_edge[2]['rewards']
        valid_rewards = {opt: rewards[opt] for opt in valid_options}
        max_reward = max(valid_rewards.values())
        best_options = [opt for opt, rew in valid_rewards.items() if rew == max_reward]
        best_option = random.choice(best_options)

        return selected_edge, best_option

    def _select_best_available_edge(self, out_edges):
        """当没有可通过的边时，选择奖励最大的边"""
        best_reward = float('-inf')
        best_choices = []

        for edge in out_edges:
            formula = edge[2]['formula']
            valid_options = self.get_valid_options(formula)
            if not valid_options:  # 跳过纯否定公式
                continue

            for opt in valid_options:
                reward = edge[2]['rewards'][opt]
                if reward > best_reward:
                    best_reward = reward
                    best_choices = [(edge, opt)]
                elif reward == best_reward:
                    best_choices.append((edge, opt))

        if best_choices:
            return random.choice(best_choices)

        # 如果连有效选项都没有，返回None
        return None, None

    def get_valid_options(self, formula):
        """获取formula对应的有效options，纯否定公式返回空列表"""
        if formula == 'True' or formula == 'true':
            return list(range(4))
        positive = extract_true_predicates(formula)
        if not positive:  # 如果没有正命题，说明是纯否定公式
            return []
        return [int(p[1]) - 1 for p in positive]

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

    def compute_average_distances_to_accepting(self) -> Dict[str, float]:
        """
        计算DFA中所有节点到可接受节点的平均距离
        平均距离是指所有从节点出发到可接受节点的路径的平均距离
        路径中不包括环，每条边的长度为1
        """

        def find_all_paths_to_accepting(start_state, visited=None):
            """找到从start_state到accepting_state的所有无环路径"""
            if visited is None:
                visited = set()

            if start_state in visited:
                return []

            if start_state == self.accepting_state:
                return [[start_state]]

            visited.add(start_state)
            all_paths = []

            for _, next_state in self.dfa.out_edges(start_state):
                # 跳过自循环
                if next_state == start_state:
                    continue

                for path in find_all_paths_to_accepting(next_state, visited.copy()):
                    all_paths.append([start_state] + path)

            return all_paths

        average_distances = {}

        for state in self.dfa.nodes():
            paths = find_all_paths_to_accepting(state)

            if not paths:
                # 如果没有路径到达接受状态
                average_distances[state] = float('inf')
            else:
                # 计算所有路径长度的平均值
                path_lengths = [len(path) - 1 for path in paths]  # 减1因为路径长度是边的数量
                average_distances[state] = sum(path_lengths) / len(path_lengths)

        return average_distances

    def print_average_distances(self):
        """打印所有节点到接受状态的平均距离"""
        distances = self.compute_average_distances_to_accepting()
        print(f"\n所有节点到接受状态 {self.accepting_state} 的平均距离:")
        print("=" * 50)
        for state, distance in distances.items():
            if distance == float('inf'):
                print(f"{state}: 无可达路径")
            else:
                print(f"{state}: {distance:.2f}")

    def get_overall_average_distance(self, accepting_state: str) -> float:
        """
        计算所有节点到可接受节点的总体平均距离
        (排除无法到达的节点和目标节点本身)
        """
        distances = self.compute_average_distances_to_accepting()

        # 过滤掉无法到达的节点和目标节点本身
        valid_distances = [dist for state, dist in distances.items()
                           if dist != float('inf') and state != accepting_state]

        if not valid_distances:
            return 0.0

        return sum(valid_distances) / len(valid_distances)

if __name__ == '__main__':
    dfa = get_dfa(dfa_text)
    path_finder = DFAPathFinder(dfa, epsilon=1)

    '''# 计算平均距离
    distances = path_finder.compute_average_distances_to_accepting()

    # 打印详细信息
    path_finder.print_average_distances()

    # 获取总体平均距离
    overall_avg = path_finder.get_overall_average_distance('@q11')
    print(f"\n总体平均距离: {overall_avg:.2f}")

    # 查看具体某个状态的平均距离
    print(f"状态@q1到@q6的平均距离: {distances['@q1']:.2f}")
'''
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
        next_edge, best_option = path_finder.predict('@q4')
        print(next_edge, best_option)