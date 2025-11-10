import csv
import os
import random
import sys
import time
from typing import Optional
sys.setrecursionlimit(1000000)
import numpy as np
from stable_baselines3.common.base_class import maybe_make_env, BaseAlgorithm
from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer, RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit
from utils import *
from LowerModel import LowerSAC
from utils import TrainingLogger
from frozen_lake2 import FrozenLakeEnv2
import copy
from stable_baselines3.common.utils import should_collect_more_steps, polyak_update, get_parameters_by_name, \
    obs_as_tensor
import time
import warnings
import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule, TensorDict
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    VecNormalize,
    VecTransposeImage,
    is_vecenv_wrapped,
    unwrap_vec_normalize,
)
from typing import Any, ClassVar, Dict, Iterable, List, Optional, Tuple, Type, TypeVar, Union
SelfBaseAlgorithm = TypeVar("SelfBaseAlgorithm", bound="BaseAlgorithm")
from LowerModel import *

class PredMod:
    def __init__(self, upper_domain, lower_domain, instance, text, training_time, log_path):
        # classify predicates
        self.log_path = log_path
        self.time_start = time.time()
        self.training_time = training_time
        self.dfa = get_dfa(text)
        self.dfa_trans = DFATransformer(text)
        self.upper_buffer_action = None
        self.active_tasks = set()
        self.learned_tasks = []
        self.discard_tasks = set()

        lower_env = FrozenLakeEnv2(desc=None, map_name="8x8", is_slippery=False)
        eval_env = maybe_make_env(copy.deepcopy(lower_env), 1)
        self.eval_env = BaseAlgorithm._wrap_env(eval_env, 1, True)
        self.lower_model = LowerPPO("MultiInputPolicy", lower_env, verbose=1,
                  learning_rate=1e-5, device='cpu', dfa_text=text)
        self.logger = TrainingLogger(log_interval=100)
        self.max_currentH = 500
        self.sub_task_learn_time = 60
        self.set_active_tasks("@q1")
        self.sample_active_task()
        self.upper_logger = {}

    def update_q_value(self, u, v, average_rewards):
        alpha = 0.2
        self.lower_model.dfa[u][v]['Q_value'] = alpha * average_rewards + (1 - alpha) * self.lower_model.dfa[u][v]['Q_value']

    def set_active_tasks(self, dfa_state):
        out_edges = self.dfa.out_edges(dfa_state, data=False)
        out_edges = [(u, v) for u, v in out_edges if u != v and v != '@q2']
        stop_training = True
        for edge in out_edges:
            if edge not in self.learned_tasks or edge not in self.discard_tasks:
                self.active_tasks.add(edge)
                stop_training = False
        return stop_training

    def sample_active_task(self):
        Q_value = {}
        for task in self.active_tasks:
            u, v = task
            Q_value[task] = self.lower_model.dfa[u][v]['Q_value']
        if random.random() < 0.7:
            active_task = random.choice(list(self.active_tasks))
        else:
            active_task = max(Q_value.keys(), key=lambda k: Q_value[k])
        return active_task

    def set_discarded_tasks(self, goal_dfa_state):
        """直接通过主路径边集合计算需要删除的边
        Args:
            G: nx.DiGraph
            main_edges: 主路径边集合，如 [(v0, v1), (v1, v2), ..., (vk, p)]

        Returns:
            list: 需要删除的边列表 [(src, dst), ...]
        """
        # find edge path
        main_edges = self.find_edge_path(self.learned_tasks, goal_dfa_state)

        # 提取主路径覆盖的关键节点（所有被指向的节点）
        critical_nodes = {v for _, v in main_edges}

        # 生成需要删除的边列表（指向关键节点但不在主路径中的边）
        discard_tasks = [
            (u, v) for u, v in self.dfa.edges()
            if v in critical_nodes and (u, v) not in main_edges
        ]
        self.discard_tasks.update(discard_tasks)

    def find_edge_path(self, edges, target):
        # 构建父节点映射 {子节点: 父节点}
        parent_map = {}
        for u, v in edges:
            parent_map[v] = u

        # 回溯节点路径
        path = []
        current = target
        while current in parent_map:
            path.append(current)
            current = parent_map[current]
        path.append(current)  # 添加起点
        path.reverse()

        # 将节点路径转为边路径
        edge_path = []
        for i in range(len(path) - 1):
            edge = (path[i], path[i + 1])
            edge_path.append(edge)

        return edge_path
    def policy_advance(self, goal_dfa_state):
        iteration = 0
        self.lower_model._last_obs = self.lower_model.env.reset()
        dfa_state = '@q' + str(self.lower_model._last_obs['ds'].item() + 1)
        while dfa_state != goal_dfa_state:
            iteration += 1
            self.lower_model._last_obs = self.lower_model.env.reset()
            edge_path = self.find_edge_path(self.learned_tasks, goal_dfa_state)
            dfa_state = '@q' + str(self.lower_model._last_obs['ds'].item() + 1)
            length = 0
            for edge in edge_path:
                done = False
                u, v = edge
                assert dfa_state == u
                while dfa_state != v or length == 1500:
                    action, _ = self.lower_model.dfa[u][v]['policy'].predict(self.lower_model._last_obs, u, v, deterministic=False)
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
                break

        assert dfa_state == goal_dfa_state
        print('success')
        return True

    def find_max_key(self, input_node):
        """
        传入一个节点，找到以该节点为第一个元素的键值对中值最大的键
        如果有相同的最大值，随机返回一个
        """
        # 筛选出以input_node为第一个元素的键值对
        filtered_items = {k: v for k, v in self.upper_logger.items() if k[0] == input_node}

        if not filtered_items:
            return None  # 如果没有找到对应的键

        # 找到最大值
        max_value = max(filtered_items.values())

        # 找到所有具有最大值的键
        max_keys = [k for k, v in filtered_items.items() if v == max_value]

        # 如果有多个最大值，随机选择一个
        return random.choice(max_keys)

    def select_convergence(self, last_average_rewards, average_rewards):
        success_rate = sum(self.logger.if_successes) / len(self.logger.if_successes)
        print(success_rate, average_rewards, round(abs(average_rewards - last_average_rewards), 2), len(self.logger.if_successes))
        return success_rate > 0.9 # and abs(average_rewards - last_average_rewards) < 1

    def set_upper_logger(self, logger, dfa_state, v, if_success):
        success_time = logger.get((dfa_state, v), [[], 0, 0])
        success_time[0].append(int(if_success))
        if len(success_time[0]) > 500:
            success_time[0].pop(0)
        success_time[1] = sum(success_time[0]) / len(success_time[0])
        success_time[2] += 1
        logger[(dfa_state, v)] = success_time

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
        u: int,
        v: int,
    ) -> bool:
        policy = self.lower_model.dfa[u][v]['policy']
        rollout_buffer = self.lower_model.dfa[u][v]['rollout_buffer']

        # set last obs
        dfa_state = '@q' + str(self.lower_model._last_obs['ds'].item() + 1)
        if dfa_state != u:
            success_advance = False
            while not success_advance:
                try:
                    success_advance = self.policy_advance(u)
                except:
                    pass
        assert self.lower_model._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.lower_model.use_sde:
            policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.lower_model.use_sde and self.lower_model.sde_sample_freq > 0 and n_steps % self.lower_model.sde_sample_freq == 0:
                # Sample a new noise matrix
                policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self.lower_model._last_obs, self.lower_model.device)
                actions, values, log_probs = policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.lower_model.action_space, spaces.Box):
                if policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(actions, self.lower_model.action_space.low, self.lower_model.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)


            original_dones = dones.copy()
            dfa_state = '@q' + str(self.lower_model._last_obs['ds'].item() + 1)
            next_dfa_state = '@q' + str(new_obs['ds'].item() + 1)
            next_values = None

            # set upper log
            if dones or next_dfa_state != dfa_state:
                '''if next_dfa_state != dfa_state:
                    print(next_dfa_state)'''
                if dfa_state != '@q1':
                    print('aaaa')
                self.set_upper_logger(self.upper_logger, dfa_state, v, next_dfa_state == v)

            # set dones
            if ((next_dfa_state != dfa_state) or self.lower_model.currentH == self.max_currentH) and not original_dones:
                dones = np.array([True])
                infos[0]['terminal_observation'] = new_obs

            # reward shaping
            if next_dfa_state == v:
                rewards = np.array([10])
            else:
                rewards = np.array([0])

            if self.logger:
                self.logger.record(reward=rewards, done=dones)

            self.lower_model.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self.lower_model._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.lower_model.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    for key, value in infos[idx]["terminal_observation"].items():
                        infos[idx]["terminal_observation"][key] = np.array([infos[idx]["terminal_observation"][key]])
                    terminal_obs = policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.lower_model.gamma * terminal_value

            rollout_buffer.add(
                self.lower_model._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self.lower_model._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
            )
            self.lower_model._last_obs = new_obs  # type: ignore[assignment]
            self.lower_model._last_episode_starts = dones

            if dones:
                self.lower_model._last_obs = self.lower_model.env.reset()

            if dones:
                # set last obs
                dfa_state = '@q' + str(self.lower_model._last_obs['ds'].item() + 1)
                if dfa_state != u:
                    success_advance = False
                    while not success_advance:
                        try:
                            success_advance = self.policy_advance(u)
                        except:
                            pass

        with th.no_grad():
            # Compute value for the last timestep
            values = policy.predict_values(obs_as_tensor(new_obs, self.lower_model.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True

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

        for u, v in self.lower_model.dfa.edges():
            self.lower_model.dfa[u][v]['policy'] = copy.deepcopy(self.lower_model.policy)

        now_time = time.time()
        while time.time() - self.time_start <= self.training_time:
            u, v = self.sample_active_task()
            print("sample task is:", u, v)
            while time.time() - now_time < 30:
                continue_training = self.collect_rollouts(self.lower_model.env, callback, None,
                                                          n_rollout_steps=self.lower_model.n_steps, u=u, v=v)

                if not continue_training:
                    break

                self.lower_model.train(u=u, v=v)

            average_rewards = self.logger.get_average_rewards()
            self.update_q_value(u, v, average_rewards)
            if self.select_convergence(self.lower_model.dfa[u][v]['average_rewards'], average_rewards):
                try:
                    self.learned_tasks.append((u, v))
                    for i in range(20):
                        self.policy_advance(v)
                    self.active_tasks.remove((u, v))
                    self.set_discarded_tasks(v)
                    stop = self.set_active_tasks(v)
                    if stop:
                        break
                except:
                    self.learned_tasks.remove((u, v))
                    print("policy_advance error.")
            self.lower_model.dfa[u][v]['average_rewards'] = average_rewards
            self.logger.reset()
            self.evaluate(time.time() - self.time_start)

            log = {}
            for key, value in self.upper_logger.items():
                log[key] = value[1]
            print(log)

            now_time = time.time()

    def sample_random_task(self, dfa_state):
        out_edges = self.dfa.out_edges(dfa_state, data=False)
        out_edges = [(u, v) for u, v in out_edges if u != v and v != '@q2']
        Q_value = {}
        for task in out_edges :
            u, v = task
            Q_value[task] = self.lower_model.dfa[u][v]['Q_value']
        if random.random() < 0.1:
            active_task = random.choice(list(self.active_tasks))
        else:
            active_task = max(Q_value.keys(), key=lambda k: Q_value[k])
        return active_task


    def get_learned_task(self, dfa_state):
        for task in self.learned_tasks:
            if dfa_state == task[0]:
                return task
        return self.sample_random_task(dfa_state)

    def evaluate(self, seconds):
        reward = np.array([0])
        done = False
        obs = self.eval_env.reset()
        length = 0
        eval_time = 5
        reward1 = -9999
        for _ in range(eval_time):
            reward = 0
            while not done:
                dfa_state = '@q' + str(obs['ds'].item() + 1)
                task = self.get_learned_task(dfa_state)

                u, v = task
                while not done:
                    action, _ = self.lower_model.dfa[u][v]['policy'].predict(obs, u, v, deterministic=False)
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
            reward1 = max(reward1, reward)
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        second = int(seconds % 60)
        hour = str(hours) + "h" + str(minutes) + "min" + str(second) + 's'
        data = [
            [hour, reward1, length / eval_time]
        ]
        with open(self.log_path, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(data)
        print('time:', hours, "h", minutes, "min", second, 's')
        print('eval_reward:', reward1, 'eval_length:', length / eval_time)

        if reward > 0:
            return True
        else:
            return False



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-log', type=str, default='task2.csv', help='log path')
    parser.add_argument('-i', type=str, default='inst21', help='inst name')
    parser.add_argument('-r', type=str, default='waterworld2', help='inst name')
    parser.add_argument('-c', type=int, default=6, help='process num')
    parser.add_argument('-t', type=int, default=6000, help='training time')
    args = parser.parse_args()
    upper_domain = 'high_level_benchmarks/waterworld/' + args.r + '.rddl'
    lower_domain = 'low_level_benchmarks/waterworld/' + args.r + '.rddl'
    instance = 'low_level_benchmarks/waterworld/' + args.i + '.rddl'
    text_path = 'dfa_text/frozenlake/frozenlake2.txt'
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
