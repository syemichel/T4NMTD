# latest version
import argparse
import copy
import csv
import heapq
import math
import os
import random
import sys
import time
from typing import Optional

from prompt_toolkit.key_binding.bindings.named_commands import self_insert
from sympy.physics.units import moles

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
        self.accepting_state = '@q' + str(self.dfa.number_of_nodes())
        self.upper_buffer_action = None
        self.learned_tasks = set()

        lower_env = FlattenAction(LowerEnv(domain=lower_domain, instance=instance))
        eval_env = maybe_make_env(copy.deepcopy(lower_env), 1)
        self.eval_env = BaseAlgorithm._wrap_env(eval_env, 1, True)
        self.lower_model = LowerSAC("MultiInputPolicy", lower_env, verbose=1, learning_starts=500,
                  learning_rate=3e-4, batch_size=256, device='cpu', dfa_text=text, train_freq=5)
        self.max_currentH = 200
        for u, v in self.dfa.edges():
            self.dfa[u][v]['cost'] = 99999999
        self.U_p = set()
        self.upper_logger = {}
        self.test_upper_logger = {}
        self.last_add_u = []
        self.selected_nodes_history = set()

    def find_min_cost_path(self):
        """
        使用Dijkstra算法找到从@q1到其他节点的最小cost路径
        排除自循环边和通往@q2的边
        U_p中的节点不参与最小cost计算，但可以作为中转节点
        如果有多个最小cost的节点，优先选择之前没有选过的，如果都选过了，再随机选择
        返回: (目标节点, 最小cost, 路径)
        """
        # 初始化类成员变量（如果不存在的话）
        if not hasattr(self, 'selected_nodes_history'):
            self.selected_nodes_history = set()

        start_node = '@q1'

        # 所有节点都参与路径计算，但排除@q2
        valid_nodes = {node for node in self.dfa.nodes() if node != '@q2'}

        # 初始化距离字典和前驱节点字典
        distances = {node: float('inf') for node in valid_nodes}
        predecessors = {node: None for node in valid_nodes}

        if start_node not in valid_nodes:
            return None, float('inf'), []

        distances[start_node] = 0

        # 优先队列：(距离, 节点)
        pq = [(0, start_node)]
        visited = set()

        while pq:
            current_dist, current_node = heapq.heappop(pq)

            if current_node in visited:
                continue

            visited.add(current_node)

            # 检查所有邻居节点
            for neighbor in self.dfa.neighbors(current_node):
                # 跳过条件：自循环、通往@q2
                if (neighbor == current_node or  # 自循环
                        neighbor == '@q2'):  # 通往@q2
                    continue

                if neighbor not in valid_nodes:
                    continue

                edge_cost = self.dfa[current_node][neighbor]['cost']
                new_dist = current_dist + edge_cost

                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    predecessors[neighbor] = current_node
                    heapq.heappush(pq, (new_dist, neighbor))

        # 找到所有最小cost的节点（排除U_p中的节点，且距离不为inf）
        min_cost = float('inf')
        min_cost_nodes = []

        for node, dist in distances.items():
            # 排除U_p中的节点、不可达节点
            if (node not in self.U_p and
                    dist != float('inf') and
                    dist < min_cost):
                min_cost = dist
                min_cost_nodes = [node]
            elif (node not in self.U_p and
                  dist != float('inf') and
                  dist == min_cost):
                min_cost_nodes.append(node)

        if not min_cost_nodes:
            return None, float('inf'), []

        # 优先选择之前没有选过的节点
        unselected_nodes = [node for node in min_cost_nodes if node not in self.selected_nodes_history]

        if unselected_nodes:
            # 从没有选过的节点中随机选择
            target_node = random.choice(unselected_nodes)
        else:
            # 如果都选过了，从所有最小cost节点中随机选择
            target_node = random.choice(min_cost_nodes)

        # 记录选择的节点
        self.selected_nodes_history.add(target_node)

        # 重构路径
        path = []
        current = target_node
        while predecessors[current] is not None:
            prev = predecessors[current]
            path.append((prev, current))
            current = prev

        path.reverse()  # 反转路径使其从起点到终点

        return target_node, min_cost, path

    def find_edge_path(self, goal_dfa_state):
        """
        使用Dijkstra算法找到从@q1到指定目标节点的最优路径
        当目标不可达时，每次选边都贪心的选cost最小的边
        如果没有成功率>0的边，则选择通往目标的最短路径
        cost = -log(P)，P是边的通过成功率
        返回: 路径的边列表 [(u1,v1), (u2,v2), ...] 或 [] 如果无任何路径
        """
        start_node = '@q1'
        IMPOSSIBLE_COST_THRESHOLD = 10  # 成功率为0的阈值

        # 检查目标节点是否有效
        if (goal_dfa_state == '@q2' or
                goal_dfa_state in self.U_p or
                goal_dfa_state not in self.dfa.nodes()):
            return []

        # 有效节点（排除@q2和U_p中的节点，但始终包含起始节点@q1）
        valid_nodes = {node for node in self.dfa.nodes()
                       if node != '@q2'}

        # 确保起始节点总是有效的
        if start_node not in self.dfa.nodes():
            return []

        # 先尝试找到目标节点的路径（只用可能的边）
        path = self._dijkstra_search(start_node, goal_dfa_state, valid_nodes, IMPOSSIBLE_COST_THRESHOLD, True)
        if path:
            return path

        # 如果目标不可达，使用贪心策略构建路径
        path = self._greedy_path_search(start_node, goal_dfa_state, valid_nodes, IMPOSSIBLE_COST_THRESHOLD)
        if path:
            return path

        # 如果仍然为空，返回随机一个通往goal_dfa_state的路径（忽略成功率）
        return self._find_random_path_to_goal(start_node, goal_dfa_state, valid_nodes)

    def _greedy_path_search(self, start_node, goal_node, valid_nodes, cost_threshold):
        """
        贪心搜索：每次选择当前节点出发cost最小的边（成功率>0）
        如果没有成功率>0的边，则找到通往目标的最短路径
        """
        visited = set()
        path = []
        current_node = start_node

        while current_node != goal_node:
            visited.add(current_node)

            # 找到当前节点的所有成功率>0的有效出边
            best_neighbor = None
            best_cost = float('inf')

            for neighbor in self.dfa.neighbors(current_node):
                if (neighbor == current_node or  # 自循环
                        neighbor == '@q2' or  # 通往@q2
                        neighbor not in valid_nodes or  # 不在有效节点中
                        neighbor in visited):  # 已访问过（避免环）
                    continue

                edge_cost = self.dfa[current_node][neighbor]['cost']

                # 只考虑成功率>0的边
                if edge_cost < cost_threshold and edge_cost < best_cost:
                    best_cost = edge_cost
                    best_neighbor = neighbor

            # 如果找到了成功率>0的边，使用它
            if best_neighbor is not None:
                path.append((current_node, best_neighbor))
                current_node = best_neighbor
            else:
                # 没有成功率>0的边了，找到从当前位置到目标的最短路径（忽略成功率）
                remaining_path = self._find_shortest_path_to_goal(current_node, goal_node, valid_nodes, visited)
                if remaining_path:
                    path.extend(remaining_path)
                    break
                else:
                    # 无法到达目标
                    break

        return path

    def _find_shortest_path_to_goal(self, start_node, goal_node, valid_nodes, already_visited):
        """
        使用BFS找到从start_node到goal_node的最短路径（忽略边的成功率）
        避免访问already_visited中的节点
        """
        if goal_node not in valid_nodes:
            return []

        queue = [(start_node, [])]
        visited = set(already_visited)  # 复制已访问节点集合
        visited.add(start_node)

        while queue:
            current_node, current_path = queue.pop(0)

            if current_node == goal_node:
                return current_path

            for neighbor in self.dfa.neighbors(current_node):
                if (neighbor == current_node or  # 自循环
                        neighbor == '@q2' or  # 通往@q2
                        neighbor not in valid_nodes or  # 不在有效节点中
                        neighbor in visited):  # 已访问过
                    continue

                new_path = current_path + [(current_node, neighbor)]

                if neighbor == goal_node:
                    return new_path

                visited.add(neighbor)
                queue.append((neighbor, new_path))

        return []  # 无法到达目标

    def _dijkstra_search(self, start_node, target_node, valid_nodes, cost_threshold, find_target):
        """
        统一的Dijkstra搜索
        find_target=True: 寻找target_node
        find_target=False: 寻找最远可达节点
        """
        distances = {node: float('inf') for node in valid_nodes}
        predecessors = {node: None for node in valid_nodes}
        distances[start_node] = 0

        pq = [(0, start_node)]
        visited = set()

        while pq:
            current_dist, current_node = heapq.heappop(pq)

            if current_node in visited:
                continue

            visited.add(current_node)

            # 如果找到目标节点
            if find_target and current_node == target_node:
                break

            # 检查邻居节点
            for neighbor in self.dfa.neighbors(current_node):
                if (neighbor == current_node or  # 自循环
                        neighbor == '@q2' or  # 通往@q2
                        neighbor not in valid_nodes):  # 不在有效节点中
                    continue

                edge_cost = self.dfa[current_node][neighbor]['cost']

                # 只使用成功率>0的边
                if edge_cost >= cost_threshold:
                    continue

                new_dist = current_dist + edge_cost

                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    predecessors[neighbor] = current_node
                    heapq.heappush(pq, (new_dist, neighbor))

        # 检查是否可达
        if not find_target or distances[target_node] == float('inf'):
            return []

        # 重构路径
        path = []
        current = target_node
        while predecessors[current] is not None:
            prev = predecessors[current]
            path.append((prev, current))
            current = prev

        path.reverse()
        return path

    def _find_random_path_to_goal(self, start_node, goal_node, valid_nodes):
        """
        找到一个随机的从start_node到goal_node的路径（忽略边的成功率）
        使用DFS随机搜索
        """
        import random

        if goal_node not in valid_nodes:
            return []

        # 使用DFS随机搜索路径
        visited = set()
        path = []

        def dfs_random(current_node):
            if current_node == goal_node:
                return True

            if current_node in visited:
                return False

            visited.add(current_node)

            # 获取所有有效的邻居节点并随机打乱
            neighbors = list(self.dfa.neighbors(current_node))
            valid_neighbors = [n for n in neighbors
                               if (n != current_node and  # 非自循环
                                   n != '@q2' and  # 不通往@q2
                                   n in valid_nodes)]  # 在有效节点中

            random.shuffle(valid_neighbors)  # 随机打乱顺序

            # 尝试每个邻居
            for neighbor in valid_neighbors:
                path.append((current_node, neighbor))
                if dfs_random(neighbor):
                    return True
                path.pop()  # 回溯

            visited.remove(current_node)
            return False

        # 开始搜索
        if dfs_random(start_node):
            return path
        else:
            return []

    '''def example_edge_path_usage(self):
        goal_state = '@q4'
        cost, path = self.find_edge_path(goal_state)

        if cost != float('inf'):
            print(f"到达 {goal_state} 的最小cost: {cost}")
            print(f"路径: {path}")
        else:
            print(f"无法到达目标节点 {goal_state}")'''

    '''# 使用示例
    def example_usage(self):
        self.U_p.add('@q3')
        self.U_p.add('@q4')
        target, cost, path = self.find_min_cost_path()

        if target:
            print(f"最小cost目标节点: {target}")
            print(f"最小cost: {cost}")
            print(f"路径: {path}")
        else:
            print("没有找到可达的目标节点")'''

    def set_upper_logger(self, logger, dfa_state, v, if_success):
        success_time = logger.get((dfa_state, v), [[], 0, 0])
        success_time[0].append(int(if_success))
        if len(success_time[0]) > 500:
            success_time[0].pop(0)
        success_time[1] = sum(success_time[0]) / len(success_time[0])
        success_time[2] += 1
        logger[(dfa_state, v)] = success_time

    def policy_advance(self, goal_dfa_state, print_path=False):
        iteration = 0
        self.lower_model._last_obs = self.lower_model.env.reset()
        dfa_state = '@q' + str(self.lower_model._last_obs['ds'].item() + 1)
        while dfa_state != goal_dfa_state:
            iteration += 1
            self.lower_model._last_obs = self.lower_model.env.reset()
            edge_path = self.find_edge_path(goal_dfa_state)
            if print_path:
                print('policy_advance_path', edge_path)
            dfa_state = '@q' + str(self.lower_model._last_obs['ds'].item() + 1)
            length = 0
            for edge in edge_path:
                done = False
                u, v = edge
                assert dfa_state == u
                while dfa_state != v or length == self.max_currentH:
                    action, _ = self.lower_model.predict(self.lower_model._last_obs, u, v, deterministic=True)
                    new_obs, reward_, done, info = self.lower_model.env.step(action)
                    length += 1
                    if done:
                        break
                    self.lower_model._last_obs = new_obs
                    dfa_state = '@q' + str(self.lower_model._last_obs['ds'].item() + 1)
                if done:
                    break
            if iteration > 100:
                print('policy advance error.')
                print('error path is ', edge_path)
                break

        assert dfa_state == goal_dfa_state
        return dfa_state == goal_dfa_state

    def eval_env_advance(self, goal_dfa_state, print_path=False):
        """
        将eval_env推进到指定的DFA状态

        参数:
            goal_dfa_state: 目标DFA状态

        返回:
            是否成功到达目标状态
        """
        iteration = 0
        max_iterations = 100

        while iteration < max_iterations:
            iteration += 1
            obs = self.eval_env.reset()
            dfa_state = '@q' + str(obs['ds'].item() + 1)

            # 如果已经在目标状态，直接返回
            if dfa_state == goal_dfa_state:
                return True, obs

            # 找到从当前状态到目标状态的路径
            edge_path = self.find_edge_path(goal_dfa_state)
            if print_path:
                print(edge_path)
            if not edge_path:
                continue

            # 沿着路径前进
            length = 0
            success = True

            for edge in edge_path:
                done = False
                u, v = edge

                # 确保当前状态匹配边的起点
                if dfa_state != u:
                    success = False
                    break

                # 尝试通过这条边
                edge_length = 0
                max_edge_steps = self.max_currentH

                while dfa_state != v and edge_length < max_edge_steps and not done:
                    # 使用predict_eval预测动作
                    action, _ = self.lower_model.predict_eval(obs, u, v, deterministic=True)
                    new_obs, reward_, done, info = self.eval_env.step(action)
                    edge_length += 1
                    length += 1

                    if done:
                        success = False
                        break

                    obs = new_obs
                    dfa_state = '@q' + str(obs['ds'].item() + 1)

                # 检查是否成功通过边
                if done or dfa_state != v or edge_length >= max_edge_steps:
                    success = False
                    break

            # 检查是否到达目标状态
            if success and dfa_state == goal_dfa_state:
                return True, obs

        print(f'eval_env_advance failed to reach {goal_dfa_state} after {iteration} iterations.')
        assert False
        return False, obs

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
            success_advance = self.policy_advance(u)
            if not success_advance:
                return False

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
            next_dfa_state = '@q' + str(new_obs['ds'].item() + 1) if not dones else '@q' + str(
                infos[0]['terminal_observation']['ds'] + 1)
            next_values = None

            # set dones
            if (next_dfa_state != dfa_state) or self.lower_model.currentH == self.max_currentH:
                dones = np.array([True])
                infos[0]['terminal_observation'] = new_obs
                # set logger
                self.set_upper_logger(self.upper_logger, dfa_state, v, next_dfa_state == v)

            # reward shaping
            if dones:
                if next_dfa_state == v:
                    rewards += 10
                else:
                    rewards = np.array([-10])
            else:
                rewards = np.array([0])

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

    def evaluate(self, seconds, eval_time, goal_dfa_state):
        total_reward = 0
        total_length = 0

        # 初始化边的统计字典
        edge_attempts = {}  # 记录每条边的尝试次数
        edge_successes = {}  # 记录每条边的成功次数

        for episode in range(eval_time):
            episode_reward = 0
            episode_length = 0
            done = False
            obs = self.eval_env.reset()

            while not done:
                path = self.find_edge_path(goal_dfa_state)
                print('path:', path)
                if not path:
                    break

                for u, v in path:
                    if done:
                        break

                    # 初始化边的统计（如果还没有）
                    edge_key = (u, v)
                    if edge_key not in edge_attempts:
                        edge_attempts[edge_key] = 0
                        edge_successes[edge_key] = 0

                    # 记录尝试次数
                    edge_attempts[edge_key] += 1
                    edge_traversed_successfully = False

                    while not done:
                        action, _ = self.lower_model.predict_eval(obs, u, v, deterministic=True)
                        new_obs, reward_, done, info = self.eval_env.step(actions=action)
                        episode_length += 1

                        prev_dfa_state = '@q' + str(obs['ds'].item() + 1)
                        next_dfa_state = '@q' + str(new_obs['ds'].item() + 1)
                        obs = new_obs

                        if done:
                            if reward_ >= 1:
                                episode_reward = 100
                                # 如果在边上结束且获得奖励，可能也算成功（根据您的定义）
                                if next_dfa_state == v:
                                    edge_traversed_successfully = True
                            break
                        elif next_dfa_state != prev_dfa_state:
                            episode_reward += 10
                            # 检查是否成功转移到目标状态
                            if next_dfa_state == v:
                                edge_traversed_successfully = True
                            break

                    # 更新边的成功次数
                    if edge_traversed_successfully:
                        edge_successes[edge_key] += 1

                    if done:
                        break

            total_reward += episode_reward
            total_length += episode_length

        avg_reward = total_reward / eval_time
        avg_length = total_length / eval_time

        # 计算每条边的成功概率
        edge_success_prob = {}
        for edge, attempts in edge_attempts.items():
            if attempts > 0:
                success_rate = edge_successes[edge] / attempts
                edge_success_prob[edge] = {
                    'attempts': attempts,
                    'successes': edge_successes[edge],
                    'success_rate': success_rate
                }

        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        second = int(seconds % 60)
        time_str = f"{hours}h{minutes}min{second}s"


        data = [[time_str, avg_reward, avg_length]]

        with open(self.log_path, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(data)

        print(f'Time: {hours}h {minutes}min {second}s')
        print(f'Eval reward: {avg_reward:.2f}, length: {avg_length:.2f}')


        return avg_reward, avg_length, edge_success_prob

    def test_edge_success_rate(self, edge, test_times=100):
        """
        测试特定边的成功率

        参数:
            edge: 元组 (u, v) 表示要测试的边
            test_times: 测试次数，默认100次

        返回:
            成功率
        """
        u, v = edge

        # 检查边的有效性
        if (u == '@q2' or v == '@q2' or
                u in self.U_p or v in self.U_p or
                u not in self.dfa.nodes() or v not in self.dfa.nodes()):
            print(f"Invalid edge: {edge}")
            return 0.0

        # 初始化统计
        attempts = 0
        successes = 0

        for i in range(test_times):
            # 将eval_env推进到边的起始节点u
            b, obs = self.eval_env_advance(u)
            if not b:
                print(f"Failed to reach start node {u} in attempt {i + 1}")
                continue

            # 或者您可能需要在eval_env_advance中返回最后的obs
            current_dfa_state = '@q' + str(obs['ds'].item() + 1)
            if current_dfa_state != u:
                print(f"Warning: Expected to be at {u}, but at {current_dfa_state}")
                continue

            # 尝试通过边
            attempts += 1
            done = False
            length = 0
            max_steps = self.max_currentH  # 最大步数限制

            while current_dfa_state != v and length < max_steps and not done:
                # 使用predict_eval预测动作
                action, _ = self.lower_model.predict_eval(obs, u, v, deterministic=True)
                new_obs, reward_, done, info = self.eval_env.step(action)
                length += 1

                obs = new_obs
                current_dfa_state = '@q' + str(new_obs['ds'].item() + 1)

            # 判断是否成功
            if current_dfa_state == v:
                successes += 1

        # 计算成功率
        success_rate = successes / attempts if attempts > 0 else 0

        # 更新DFA中的cost
        if success_rate > 0:
            new_cost = -np.log(success_rate)
            self.dfa[u][v]['cost'] = new_cost
            print(f"  Updated cost: {new_cost:.3f}")

        if success_rate == 1:
            for _ in range(50):
                b, obs = self.eval_env_advance(edge[1], print_path=True)
                attempts += 1
                if b:
                    successes += 1
                b = self.policy_advance(edge[1], print_path=True)
                attempts += 1
                if b:
                    successes += 1
        success_rate = successes / attempts if attempts > 0 else 0



        # 更新DFA中的cost
        if success_rate > 0:
            new_cost = -np.log(success_rate)
            self.dfa[u][v]['cost'] = new_cost
            print(f"  Updated cost: {new_cost:.3f}")


        print(edge, success_rate)
        return success_rate

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
            target, cost, path = self.find_min_cost_path()
            if target == self.accepting_state:
                break
            for neighbor in self.dfa.successors(target):
                # 跳过自循环和到@q2的边
                if neighbor == target or neighbor == '@q2':
                    continue
                u = target
                v = neighbor
                print("learn task (" + u + ', ' + v + ')')

                now_time1 = time.time()

                while time.time() - now_time < 300:
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


                    # print(time.time() - now_time1)
                    if time.time() - now_time1 > 30:
                        avg_reward, avg_length, edge_success_prob = self.evaluate(time.time() - self.time_start, eval_time=1, goal_dfa_state=self.accepting_state)

                        success_rate = self.test_edge_success_rate((u, v), test_times=5)
                        now_time1 = time.time()
                        log = {}
                        for key, value in self.upper_logger.items():
                            log[key] = value[1]
                        print(log)

                        print("self.lower_model.num_timesteps:", self.lower_model.num_timesteps)

                        '''if (u, v) in edge_success_prob.keys() and edge_success_prob[(u, v)]['success_rate'] != 0:
                            self.dfa[u][v]['cost'] = -math.log(edge_success_prob[(u, v)]['success_rate'])
                        elif self.upper_logger[(u, v)][1] != 0:
                            self.dfa[u][v]['cost'] = -math.log(self.upper_logger[(u, v)][1])'''

                        if success_rate == 1.0 or (time.time() - now_time > 300 and self.upper_logger[(u, v)][1] == 0):
                            now_time = time.time()
                            break

                now_time = time.time()

            for neighbor in self.dfa.successors(target):
                if self.dfa[target][neighbor]['cost'] == 0:
                    self.U_p.add(target)
                    print('add dfa state ' + target)
                    break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-log', type=str, default='task2.csv', help='log path')
    parser.add_argument('-i', type=str, default='inst1', help='inst name')
    parser.add_argument('-r', type=str, default='racecar1', help='inst name')
    parser.add_argument('-c', type=int, default=6, help='process num')
    parser.add_argument('-t', type=int, default=6000, help='training time')
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
    model.learn()

