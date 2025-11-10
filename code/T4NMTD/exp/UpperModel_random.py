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
else if(ds==@q3 ^ (p3 & ~p1 & ~p2 & ~p5)) then @q3
else if(ds==@q3 ^ (p3 & p5 & ~p1 & ~p2)) then @q6
else if(ds==@q3 ^ (p2 & ~p1 & ~p3)) then @q4
else if(ds==@q3 ^ ((p1 & p2) | (p1 & p3) | (p2 & p3))) then @q2
else if(ds==@q3 ^ (p1 & ~p2 & ~p3)) then @q5
else if(ds==@q4 ^ (~p1 & ~p2 & ~p3)) then @q1
else if(ds==@q4 ^ (p3 & ~p1 & ~p2)) then @q3
else if(ds==@q4 ^ (p2 & ~p1 & ~p3 & ~p4)) then @q4
else if(ds==@q4 ^ ((p1 & p2) | (p1 & p3) | (p2 & p3))) then @q2
else if(ds==@q4 ^ (p2 & p4 & ~p1 & ~p3)) then @q7
else if(ds==@q4 ^ (p1 & ~p2 & ~p3)) then @q5
else if(ds==@q2 ^ (true)) then @q2
else if(ds==@q5 ^ (~p1 & ~p2 & ~p3)) then @q1
else if(ds==@q5 ^ (p3 & ~p1 & ~p2)) then @q3
else if(ds==@q5 ^ (p2 & ~p1 & ~p3)) then @q4
else if(ds==@q5 ^ ((p1 & p2) | (p1 & p3) | (p2 & p3))) then @q2
else if(ds==@q5 ^ (p1 & ~p2 & ~p3 & ~p4)) then @q5
else if(ds==@q5 ^ (p1 & p4 & ~p2 & ~p3)) then @q7
else if(ds==@q6 ^ (~p1 & ~p2)) then @q6
else if(ds==@q6 ^ (p2 & ~p1 & ~p3)) then @q8
else if(ds==@q6 ^ ((p1 & p2) | (p1 & p3) | (p2 & p3))) then @q2
else if(ds==@q6 ^ (p1 & ~p2 & ~p3)) then @q9
else if(ds==@q7 ^ (~p3 & (~p1 | ~p2))) then @q7
else if(ds==@q7 ^ (p3 & ~p1 & ~p2)) then @q10
else if(ds==@q7 ^ ((p1 & p2) | (p1 & p3) | (p2 & p3))) then @q2
else if(ds==@q8 ^ (~p1 & ~p2)) then @q6
else if(ds==@q8 ^ (p2 & ~p1 & ~p3 & ~p4)) then @q8
else if(ds==@q8 ^ ((p1 & p2) | (p1 & p3) | (p2 & p3))) then @q2
else if(ds==@q8 ^ (p2 & p4 & ~p1 & ~p3)) then @q11
else if(ds==@q8 ^ (p1 & ~p2 & ~p3)) then @q9
else if(ds==@q9 ^ (~p1 & ~p2)) then @q6
else if(ds==@q9 ^ (p2 & ~p1 & ~p3)) then @q8
else if(ds==@q9 ^ ((p1 & p2) | (p1 & p3) | (p2 & p3))) then @q2
else if(ds==@q9 ^ (p1 & ~p2 & ~p3 & ~p4)) then @q9
else if(ds==@q9 ^ (p1 & p4 & ~p2 & ~p3)) then @q11
else if(ds==@q10 ^ (~p3 & (~p1 | ~p2))) then @q7
else if(ds==@q10 ^ (p3 & ~p1 & ~p2 & ~p5)) then @q10
else if(ds==@q10 ^ (p3 & p5 & ~p1 & ~p2)) then @q11
else if(ds==@q10 ^ ((p1 & p2) | (p1 & p3) | (p2 & p3))) then @q2
else if(ds==@q11 ^ ((~p1 & ~p2) | (~p1 & ~p3) | (~p2 & ~p3))) then @q11
else if(ds==@q11 ^ ((p1 & p2) | (p1 & p3) | (p2 & p3))) then @q2'''


class DFAPathFinder:
    def __init__(self, dfa: nx.DiGraph, epsilon: float = 0.1, option_num=5):
        self.dfa = dfa
        self.accepting_state = '@q' + str(dfa.number_of_nodes())
        self.epsilon = epsilon

        # 为每个边分配一个option，确保同一个节点的有效出边有不同的option
        self.edge_option_map = self._assign_edge_options()

        # 设置默认边奖励为字典,每个option都是-9999
        for edge in self.dfa.edges():
            if 'rewards' not in self.dfa.edges[edge]:
                self.dfa.edges[edge]['rewards'] = {i: -99999999 for i in range(option_num)}
                self.dfa.edges[edge]['best_option'] = 0

        # 检查是否存在正向环路
        if self._has_positive_cycle():
            raise ValueError("DFA中存在正向环路，可能导致无限累积奖励！")

        # 预计算所有状态到接受状态的最优路径
        self.optimal_paths = self._compute_all_optimal_paths()

    def _is_excluded_edge(self, from_state: str, to_state: str) -> bool:
        """
        判断边是否应该被排除（自循环边或通往@q2的边）
        """
        return to_state == from_state or to_state == '@q2'

    def _assign_edge_options(self):
        """
        为每个边分配option，确保同一个节点的有效出边（排除自循环和通往@q2的边）有不同的option
        """
        edge_option_map = {}
        predicate_options = [0, 1, 2, 3, 4]  # 对应 p1, p2, p3, p4, p5

        # 按节点分组处理出边
        for node in self.dfa.nodes():
            all_out_edges = list(self.dfa.out_edges(node))

            # 分离有效边（需要确保不重复option的边）和排除边
            valid_edges = []
            excluded_edges = []

            for edge in all_out_edges:
                from_state, to_state = edge
                if self._is_excluded_edge(from_state, to_state):
                    excluded_edges.append(edge)
                else:
                    valid_edges.append(edge)

            # 检查有效边数量是否超过可用option数量
            if len(valid_edges) > len(predicate_options):
                raise ValueError(
                    f"节点 {node} 的有效出边数量 ({len(valid_edges)}) 超过了可用的option数量 ({len(predicate_options)})")

            # 为有效边随机分配不重复的option
            available_options = predicate_options.copy()
            random.shuffle(available_options)

            for i, edge in enumerate(valid_edges):
                assigned_option = available_options[i]
                edge_option_map[edge] = assigned_option

            # 为排除的边随机分配option（可以重复，因为它们不参与option唯一性检查）
            for edge in excluded_edges:
                assigned_option = random.choice(predicate_options)
                edge_option_map[edge] = assigned_option

        return edge_option_map

    def get_edge_option(self, from_state: str, to_state: str) -> int:
        """
        获取边对应的option
        """
        edge = (from_state, to_state)
        return self.edge_option_map.get(edge, 0)  # 默认返回option 0

    def extract_true_predicates(self, formula, from_state="", to_state=""):
        """
        根据边的分配option返回对应的predicate
        """
        if formula.lower() in ['true']:
            # 对于True formula，返回所有predicates
            return ['p1', 'p2', 'p3', 'p4', 'p5']
        else:
            # 根据边的option返回对应的predicate
            if from_state and to_state:
                edge_option = self.get_edge_option(from_state, to_state)
                return [f'p{edge_option + 1}']  # option 0对应p1，option 1对应p2，等等
            else:
                # 如果没有提供边信息，返回随机predicate
                return [f'p{random.randint(1, 5)}']

    def set_edges_reward(self, rewards_dict: Dict[tuple, float], option: int):
        """
        通过字典一次性设置多个边的奖励值
        rewards_dict: 形如 {('@q1', '@q3'): 10, ('@q1', '@q4'): 5} 的字典
        或者接收元组列表 [('@q1', '@q3', 10), ('@q1', '@q4', 5)]
        option: 0-4之间的整数，表示哪个agent的奖励信息
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
            return max_states[0] if max_states else None

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

        def get_valid_options(edge_tuple):
            """获取边对应的有效option"""
            from_state, to_state, data = edge_tuple
            formula = data['formula']

            if formula == 'True' or formula == 'true':
                return list(range(5))  # 返回所有可用的option
            else:
                # 返回该边分配的option
                edge_option = self.get_edge_option(from_state, to_state)
                return [edge_option]

        def get_edge_score(edge, valid_options):
            """计算边的得分，用于排序"""
            rewards = edge[2]['rewards']
            valid_rewards = [rewards[opt] for opt in valid_options if opt in rewards]
            if not valid_rewards:
                return (0, -float('inf'))

            max_reward = max(valid_rewards)

            # 优先级：非-9999999999奖励 > -9999999999奖励，然后按奖励值排序
            if max_reward != -9999999999:
                return (1, max_reward)  # 高优先级
            else:
                return (0, max_reward)  # 低优先级

        # 获取自循环formula
        self_loop_formula = get_self_loop_formula(current_state)

        # 获取所有出边并过滤掉自循环和通往@q2的边
        all_edges = []
        for edge in self.dfa.out_edges(current_state, data=True):
            from_state, to_state, data = edge
            # 排除自循环和通往@q2的边
            if to_state == current_state or to_state == '@q2':
                continue
            all_edges.append(edge)

        # 为每条边计算有效选项和得分
        edge_info = []
        for edge in all_edges:
            valid_options = get_valid_options(edge)
            if valid_options:  # 只考虑有有效选项的边
                score = get_edge_score(edge, valid_options)
                edge_info.append((edge, valid_options, score))

        # 如果没有任何有效边，检查是否有自循环边可用
        if not edge_info:
            # 查找自循环边
            for edge in self.dfa.out_edges(current_state, data=True):
                from_state, to_state, data = edge
                if to_state == current_state:  # 自循环边
                    valid_options = get_valid_options(edge)
                    if valid_options:
                        return edge, random.choice(valid_options)

            # 如果连自循环边都没有，抛出异常
            raise ValueError(f"状态 {current_state} 没有任何可用的边！")

        # 按得分排序（优先级高的在前，同优先级按奖励值降序）
        edge_info.sort(key=lambda x: x[2], reverse=True)

        # 分离高优先级边（有非-9999999999奖励）和低优先级边
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
        valid_rewards = {opt: rewards[opt] for opt in valid_options if opt in rewards}
        if not valid_rewards:
            # 如果没有有效奖励，随机选择一个有效option
            best_option = random.choice(valid_options)
        else:
            max_reward = max(valid_rewards.values())
            best_options = [opt for opt, rew in valid_rewards.items() if rew == max_reward]
            best_option = random.choice(best_options)

        return selected_edge, best_option

    def get_valid_options(self, formula, from_state="", to_state=""):
        """获取formula对应的有效options"""
        if formula == 'True' or formula == 'true':
            return list(range(5))

        if from_state and to_state:
            edge_option = self.get_edge_option(from_state, to_state)
            return [edge_option]
        else:
            return []

    def print_all_edges_rewards(self):
        """打印所有边的奖励信息"""
        print("\n所有边的奖励信息:")
        print("=" * 60)
        for (u, v, data) in self.dfa.edges(data=True):
            edge_option = self.get_edge_option(u, v)
            edge_type = ""
            if self._is_excluded_edge(u, v):
                if v == u:
                    edge_type = " (自循环边)"
                elif v == '@q2':
                    edge_type = " (通往@q2)"
            print(f"边 {u} -> {v} (分配option: {edge_option}){edge_type}:")
            for option, reward in data['rewards'].items():
                print(f"  Option {option}: {reward}")
            print(f"  最优Option: {data['best_option']}")
            print("-" * 30)

    def print_edge_option_assignments(self):
        """打印所有边的option分配情况"""
        print("\n边的option分配情况:")
        print("=" * 50)

        # 按节点分组显示
        for node in sorted(self.dfa.nodes()):
            all_out_edges = list(self.dfa.out_edges(node))
            if all_out_edges:
                print(f"\n节点 {node} 的出边:")

                # 分离有效边和排除边
                valid_edges = []
                excluded_edges = []

                for from_state, to_state in all_out_edges:
                    edge_option = self.get_edge_option(from_state, to_state)
                    edge_info = f"  {from_state} -> {to_state}: Option {edge_option} (p{edge_option + 1})"

                    if self._is_excluded_edge(from_state, to_state):
                        if to_state == from_state:
                            edge_info += " [自循环边]"
                        elif to_state == '@q2':
                            edge_info += " [通往@q2]"
                        excluded_edges.append((edge_option, edge_info))
                    else:
                        valid_edges.append((edge_option, edge_info))

                # 显示有效边
                if valid_edges:
                    print("  有效边（需要唯一option）:")
                    used_options = []
                    for edge_option, edge_info in valid_edges:
                        print(edge_info)
                        used_options.append(edge_option)

                    # 检查有效边是否有重复的option
                    if len(used_options) != len(set(used_options)):
                        print(f"    ⚠️  警告: 有效边存在重复的option!")

                # 显示排除边
                if excluded_edges:
                    print("  排除边（允许重复option）:")
                    for edge_option, edge_info in excluded_edges:
                        print(edge_info)

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

    # 执行epsilon-greedy步（不打印信息）
    for i in range(5):
        next_edge, best_option = path_finder.predict('@q4')
        print(next_edge, best_option)