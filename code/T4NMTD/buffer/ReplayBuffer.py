import random
import time
import ray
from stable_baselines3.common.buffers import *
import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer
from stable_baselines3.common.type_aliases import DictReplayBufferSamples
from collections import deque
from util.DFA import *
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

class Single_Replay_Buffer(DictReplayBuffer):
    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            device: Union[th.device, str] = "auto",
            n_envs: int = 1,
            optimize_memory_usage: bool = False,
            handle_timeout_termination: bool = True,
            option_num: int = 3,
            learn_start:int = 500,
            option_index: int = 1,
            dfa = None,
            upper_policy = None,
    ):
        buffer_size = 100000
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            n_envs,
            optimize_memory_usage,
            handle_timeout_termination)
        self.policy_update_time = 30
        self.upper_policy = upper_policy
        self.dfa = dfa
        self.dfa_index = option_index
        self.deque = [] # store transiformed exps
        self.learning_start = learn_start
        self.sample_start = False
        self.option_num = option_num

    def add(
            self,
            obs: Dict[str, np.ndarray],
            next_obs: Dict[str, np.ndarray],
            action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            infos: List[Dict[str, Any]],
            next_option_index=None
    ) -> None:
        # Copy to avoid modification by reference
        for key in self.observations.keys():
            # Reshape needed when using multiple envs with discrete observations
            # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                obs[key] = obs[key].reshape((self.n_envs,) + self.obs_shape[key])
            self.observations[key][self.pos] = np.array(obs[key])

        for key in self.next_observations.keys():
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                next_obs[key] = next_obs[key].reshape((self.n_envs,) + self.obs_shape[key])
            self.next_observations[key][self.pos] = np.array(next_obs[key])

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.dones[self.pos] = np.array(done)

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        if self.pos in self.deque:
            self.deque.remove(self.pos)

        if next_option_index is not None:
            self.deque.append(self.pos)

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def get_lower_model_index(self, obs, deterministic=False):
        dfa_state = '@q' + str(obs['ds'].item() + 1)
        next_edge, best_option = self.upper_policy.predict(dfa_state, deterministic=False)
        try:
            assert 'p' + str(best_option + 1) in extract_true_predicates(self.dfa[dfa_state][next_edge[1]]['formula'],
                                                                         self.dfa[dfa_state][next_edge[0]]['formula'])
        except Exception as e:
            print(e)
            print(next_edge, best_option)
        return best_option, [next_edge[1]]

    def sample(
            self,
            batch_size: int,
            dfa_state: str,
            env: Optional[VecNormalize] = None,
    ):
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        batch_inds_ = []
        new_batch_inds = [[] for _ in range(self.option_num)]
        nums = [0] * self.option_num
        for i, pos in enumerate(batch_inds):
            if pos not in self.deque:
                new_batch_inds[self.dfa_index].append(pos)
                nums[self.dfa_index] += 1
            else:
                batch_inds_.append(pos)

        # calculate nums
        if len(batch_inds_) != 0:
            next_observations = {}
            for key, value in self.next_observations.items():
                next_observations[key] = np.array([value[i] for i in batch_inds_])
                next_observations[key] = next_observations[key].squeeze(1)

            for i, dfa_state in enumerate(next_observations['ds']):
                dfa = '@q' + str(dfa_state.item() + 1)
                next_edge, best_option = self.upper_policy.predict(dfa, deterministic=False)
                new_batch_inds[best_option].append(batch_inds_[i])
                nums[best_option] += 1
        new_batch_inds = np.concatenate(new_batch_inds).astype(np.int32)
        return self._get_samples(new_batch_inds, env=env), nums


from stable_baselines3.common.buffers import *
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer
from stable_baselines3.common.type_aliases import DictReplayBufferSamples


class RBC_Replay_Buffer(DictReplayBuffer):
    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            device: Union[th.device, str] = "auto",
            n_envs: int = 1,
            optimize_memory_usage: bool = False,
            handle_timeout_termination: bool = True,
            automata_states: int = 3,
    ):
        n_envs = 1
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            n_envs,
            optimize_memory_usage,
            handle_timeout_termination)
        self.automata_states = automata_states
        self.state_full = np.array([False for _ in range(self.automata_states)])
        self.state_pos = np.array([i * (buffer_size // self.automata_states) for i in range(self.automata_states)])
        self.lower_bound = np.array([self.buffer_size // self.automata_states * add_state for add_state in range(self.automata_states)])

    def add(
            self,
            obs: Dict[str, np.ndarray],
            next_obs: Dict[str, np.ndarray],
            action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            infos: List[Dict[str, Any]],
            state_exp_num: np.ndarray = np.array([]),
            add_state: int = 0
    ) -> None:
        # calculate pos
        pos = self.state_pos[add_state]
        # Copy to avoid modification by reference
        for key in self.observations.keys():
            # Reshape needed when using multiple envs with discrete observations
            # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                obs[key] = obs[key].reshape((self.n_envs,) + self.obs_shape[key])
            self.observations[key][pos] = np.array(obs[key])

        for key in self.next_observations.keys():
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                next_obs[key] = next_obs[key].reshape((self.n_envs,) + self.obs_shape[key])
            self.next_observations[key][pos] = np.array(next_obs[key]).copy()

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.actions[pos] = np.array(action).copy()
        self.rewards[pos] = np.array(reward).copy()
        self.dones[pos] = np.array(done).copy()

        if self.handle_timeout_termination:
            self.timeouts[pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.state_pos[add_state] += 1
        if self.state_pos[add_state] == self.buffer_size // self.automata_states * (add_state + 1):
            self.state_full[add_state] = True
            self.state_pos[add_state] = self.buffer_size // self.automata_states * add_state

    def sample(
            self,
            batch_size: int,
            env: Optional[VecNormalize] = None,
    ) -> tuple[DictReplayBufferSamples, Any]:
        upper_bound = np.array([(self.buffer_size // self.automata_states * (state + 1) if self.state_full[
            state] else self.state_pos[state]) for state in range(self.automata_states)])
        dfa_state_num = upper_bound - self.lower_bound
        prob = dfa_state_num / np.sum(dfa_state_num)
        chosen_state_num = np.zeros(self.automata_states, dtype=int)
        for _ in range(batch_size):
            chosen_state_num[np.random.choice(a=range(self.automata_states), p=prob)] += 1
        batch_inds = np.array([], dtype=int)
        i = 0
        for num in chosen_state_num:
            a = np.random.randint(self.lower_bound[i],
                                  upper_bound[i], size=num)
            batch_inds = np.concatenate((batch_inds, a), axis=0)
            i += 1
        return self._get_samples(batch_inds, env=env), chosen_state_num

class Clem_Replay_Buffer(DictReplayBuffer):
    def add(
        self,
        obs: Dict[str, np.ndarray],
        next_obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
        parallel_num: int = 1,
    ) -> None:  # pytype: disable=signature-mismatch
        # Copy to avoid modification by reference
        for i in range(parallel_num):
            for key in self.observations.keys():
                # Reshape needed when using multiple envs with discrete observations
                # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
                if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                    obs[key] = obs[key].reshape((self.n_envs,) + self.obs_shape[key])
                self.observations[key][i] = np.array(obs[key][i])

            for key in self.next_observations.keys():
                if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                    next_obs[key] = next_obs[key].reshape((self.n_envs,) + self.obs_shape[key])
                self.next_observations[key][i] = np.array(next_obs[key][i]).copy()

            # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
            action = action.reshape((self.n_envs, self.action_dim))

            self.actions[i] = np.array(action[i]).copy()
            self.rewards[i] = np.array(reward[i]).copy()
            self.dones[i] = np.array(done[i]).copy()


    def sample(
        self,
        batch_size: int,
        env: Optional[VecNormalize] = None,
    ) -> DictReplayBufferSamples:  # type: ignore[signature-mismatch] #FIXME:
        batch_inds = np.array(range(batch_size))
        buffer = self._get_samples(batch_inds, env=env)
        return buffer

