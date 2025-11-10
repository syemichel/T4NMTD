from stable_baselines3.common.buffers import DictRolloutBuffer
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Union

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.type_aliases import (
    DictReplayBufferSamples,
    DictRolloutBufferSamples,
    ReplayBufferSamples,
    RolloutBufferSamples,
)
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecNormalize

class RBC_Rollout_Buffer:
    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            device: Union[th.device, str] = "auto",
            gae_lambda: float = 1,
            gamma: float = 0.99,
            n_envs: int = 1,
            automata_states: int = 3,
            num_per_class: int = 4,
    ):
        self.automata_states = automata_states
        self.buffers = [SingleDictRolloutBuffer_OX(buffer_size, observation_space, action_space, device, gae_lambda, gamma,
                                                n_envs) for _ in range(self.automata_states)]
        self.num_per_class = num_per_class
    def reset(self) -> None:
        for buffer in self.buffers:
            buffer.reset()

    def add(
            self,
            obs: Dict[str, np.ndarray],
            action: np.ndarray,
            reward: np.ndarray,
            value: th.Tensor,
            log_prob: th.Tensor,
            dones: np.ndarray,
            next_value: th.Tensor = None,
    ) -> None:
        add_state = int(obs['as'].item() / self.num_per_class)
        self.buffers[add_state].add(obs, action, reward, value, log_prob, dones, next_value)

    def get(
            self,
            batch_size: Optional[int] = None,
    ) -> Generator[DictRolloutBufferSamples, None, None]:
        index = 0
        for buffer in self.buffers:
            for batch in buffer.get(batch_size):
                yield batch, index
            index += 1

    def compute_returns_and_advantage(self, last_values: th.Tensor, special_add, dfa_state) -> None:
        for i in range(self.automata_states):
            if i == dfa_state:
                self.buffers[dfa_state].compute_returns_and_advantage(last_values, special_add)
            else:
                self.buffers[dfa_state].compute_returns_and_advantage(last_values, True)

class SingleDictRolloutBuffer_OX(DictRolloutBuffer):
    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Dict,
            action_space: spaces.Space,
            device: Union[th.device, str] = "auto",
            gae_lambda: float = 1,
            gamma: float = 0.99,
            n_envs: int = 1,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        self.next_values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.special_add = True

    def reset(self):
        self.next_values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.special_dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        super().reset()

    def add(
        self,
        obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
        dones: np.ndarray,
        next_value: th.Tensor = None,
    ) -> None:
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        for key in self.observations.keys():
            obs_ = np.array(obs[key])
            # Reshape needed when using multiple envs with discrete observations
            # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                obs_ = obs_.reshape((self.n_envs,) + self.obs_shape[key])
            self.observations[key][self.pos] = obs_

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.dones[self.pos] = np.array(dones)
        self.values[self.pos] = value.clone().cpu().numpy().flatten()

        if not self.special_add:
            self.next_values[self.pos-1] = value.clone().cpu().numpy().flatten()
        else:
            self.special_add = False
        if next_value is not None:
            self.next_values[self.pos] = next_value.clone().cpu().numpy().flatten()
            self.special_add = True
            self.special_dones[self.pos] = np.array(True)
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(  # type: ignore[override]
        self,
        batch_size: Optional[int] = None,
    ) -> Generator[DictRolloutBufferSamples, None, None]:
        indices = np.random.permutation(self.pos * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            for key, obs in self.observations.items():
                self.observations[key] = self.swap_and_flatten(obs)

            _tensor_names = ["actions", "values", "log_probs", "advantages", "returns"]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        start_idx = 0
        while start_idx < self.pos * self.n_envs:
            if start_idx + batch_size < self.pos:
                yield self._get_samples(indices[start_idx : start_idx + batch_size])
            else:
                yield self._get_samples(indices[start_idx: self.pos-1])
            start_idx += batch_size

    def compute_returns_and_advantage(self, last_values: th.Tensor, special_add) -> None:
        if not special_add:
            self.next_values[self.buffer_size - 1] = last_values.clone().cpu().numpy().flatten()
        delta = self.rewards + self.gamma * self.next_values * (1 - self.dones) - self.values
        last_gae_lam = 0
        dones = self.dones + self.special_dones
        for step in reversed(range(self.pos)):
            last_gae_lam = delta[step] + self.gamma * self.gae_lambda * (1 - dones[step]) * last_gae_lam
            self.advantages[step] = last_gae_lam
        self.returns = self.advantages + self.values
'''class SingleDictRolloutBuffer0(DictRolloutBuffer):
    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Dict,
            action_space: spaces.Space,
            device: Union[th.device, str] = "auto",
            gae_lambda: float = 1,
            gamma: float = 0.99,
            n_envs: int = 1,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        self.next_observations = {}
        self.next_values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

    def reset(self):
        self.next_observations = {}
        for key, obs_input_shape in self.obs_shape.items():
            self.next_observations[key] = np.zeros((self.buffer_size, self.n_envs, *obs_input_shape), dtype=np.float32)
        self.next_values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        super().reset()

    def add(
        self,
        obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
        next_obs: Dict[str, np.ndarray],
        dones: np.ndarray,
    ) -> None:
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        for key in self.observations.keys():
            obs_ = np.array(obs[key])
            # Reshape needed when using multiple envs with discrete observations
            # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                obs_ = obs_.reshape((self.n_envs,) + self.obs_shape[key])
            self.observations[key][self.pos] = obs_

        for key in self.next_observations.keys():
            next_obs_ = np.array(next_obs[key])
            # Reshape needed when using multiple envs with discrete observations
            # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                next_obs_ = next_obs_.reshape((self.n_envs,) + self.obs_shape[key])
            self.next_observations[key][self.pos] = next_obs_

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.dones[self.pos] = np.array(dones)
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        if self.pos != 0:
            self.next_values[self.pos-1] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True


    def compute_returns_and_advantage(self, last_values: th.Tensor) -> None:
        self.next_values[self.buffer_size - 1] = last_values.clone().cpu().numpy().flatten()
        delta = self.rewards + self.gamma * self.next_values * (1 - self.dones) - self.values
        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            last_gae_lam = delta[step] + self.gamma * self.gae_lambda * (1 - self.dones[step]) * last_gae_lam
            self.advantages[step] = last_gae_lam
        self.returns = self.advantages + self.values'''
class SingleDictRolloutBuffer(DictRolloutBuffer):
    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Dict,
            action_space: spaces.Space,
            device: Union[th.device, str] = "auto",
            gae_lambda: float = 1,
            gamma: float = 0.99,
            n_envs: int = 1,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        self.next_values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.special_add = True

    def reset(self):
        '''self.next_observations = {}
        for key, obs_input_shape in self.obs_shape.items():
            self.next_observations[key] = np.zeros((self.buffer_size, self.n_envs, *obs_input_shape), dtype=np.float32)'''
        self.next_values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.special_dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        super().reset()

    def add(
        self,
        obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
        dones: np.ndarray,
        next_value: th.Tensor = None,
    ) -> None:
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        for key in self.observations.keys():
            obs_ = np.array(obs[key])
            # Reshape needed when using multiple envs with discrete observations
            # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                obs_ = obs_.reshape((self.n_envs,) + self.obs_shape[key])
            self.observations[key][self.pos] = obs_

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.dones[self.pos] = np.array(dones)
        self.values[self.pos] = value.clone().cpu().numpy().flatten()

        if not self.special_add:
            self.next_values[self.pos-1] = value.clone().cpu().numpy().flatten()
        else:
            self.special_add = False
        if next_value is not None:
            self.next_values[self.pos] = next_value.clone().cpu().numpy().flatten()
            self.special_add = True
            self.special_dones[self.pos] = np.array(True)
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True


    def compute_returns_and_advantage(self, last_values: th.Tensor, special_add) -> None:
        if not special_add:
            self.next_values[self.buffer_size - 1] = last_values.clone().cpu().numpy().flatten()
        delta = self.rewards + self.gamma * self.next_values * (1 - self.dones) - self.values
        last_gae_lam = 0
        dones = self.dones + self.special_dones
        for step in reversed(range(self.buffer_size)):
            last_gae_lam = delta[step] + self.gamma * self.gae_lambda * (1 - dones[step]) * last_gae_lam
            self.advantages[step] = last_gae_lam
        self.returns = self.advantages + self.values

