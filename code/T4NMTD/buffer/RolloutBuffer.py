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

class MyDictRolloutBuffer(DictRolloutBuffer):
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

    def reset(self):
        '''self.next_observations = {}
        for key, obs_input_shape in self.obs_shape.items():
            self.next_observations[key] = np.zeros((self.buffer_size, self.n_envs, *obs_input_shape), dtype=np.float32)'''
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

            if obs_.shape == (7, 7, 3):
                a = np.transpose(obs_, (2, 0, 1))
                obs_ = a.reshape(1, 3, 7, 7)
            self.observations[key][self.pos] = obs_

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.dones[self.pos] = np.array(dones)
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.next_values[self.pos] = next_value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True


    def compute_returns_and_advantage(self):
        delta = self.rewards + self.gamma * self.next_values * (1 - self.dones) - self.values
        self.advantages = delta
        self.returns = self.advantages + self.values

