import random
import time

from stable_baselines3.common.buffers import *
import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer
from stable_baselines3.common.type_aliases import DictReplayBufferSamples
from collections import deque
from utils import *
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union


class Combined_Replay_Buffer:
    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            device: Union[th.device, str] = "auto",
            n_envs: int = 1,
            optimize_memory_usage: bool = False,
            handle_timeout_termination: bool = True,
            dfa=None,
    ):
        buffer_size = 200000
        self.dfa = dfa
        for u, v in self.dfa.edges():
            self.dfa[u][v]['buffer'] = DictReplayBuffer(
                buffer_size,
                observation_space,
                action_space,
                device,
                n_envs,
                optimize_memory_usage,
                handle_timeout_termination)
            self.dfa[u][v]['sample_start'] = False

    def add(
            self,
            obs: Dict[str, np.ndarray],
            next_obs: Dict[str, np.ndarray],
            action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            infos: List[Dict[str, Any]],
            u: str,
            v: str,
    ) -> None:
        self.dfa[u][v]['buffer'].add(
            obs,
            next_obs,
            action,
            reward,
            done,
            infos,
        )
        if self.dfa[u][v]['buffer'].pos > 300:
            self.dfa[u][v]['sample_start'] = True

    def sample(
            self,
            batch_size: int,
            env: Optional[VecNormalize],
            u: str,
            v: str,
    ) -> tuple[DictReplayBufferSamples, Any]:
        return self.dfa[u][v]['buffer'].sample(batch_size, env)
