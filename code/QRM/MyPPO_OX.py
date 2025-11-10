import copy
import sys
import time
from copy import deepcopy
from numpy import ndarray
from stable_baselines3 import PPO
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union
from stable_baselines3.common.buffers import *
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.td3.policies import TD3Policy
from torch.nn import functional as F
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
import numpy as np
import torch as th
from collections import deque
from gymnasium import spaces
from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit

from RolloutBuffer import RBC_Rollout_Buffer


class MyPPO(PPO):

    def __init__(
            self,
            policy: Union[str, Type[ActorCriticPolicy]],
            env: Union[GymEnv, str],
            learning_rate: Union[float, Schedule] = 3e-4,
            n_steps: int = 2048,
            batch_size: int = 64,
            n_epochs: int = 10,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            clip_range: Union[float, Schedule] = 0.2,
            clip_range_vf: Union[None, float, Schedule] = None,
            normalize_advantage: bool = True,
            ent_coef: float = 0.0,
            vf_coef: float = 0.5,
            max_grad_norm: float = 0.5,
            use_sde: bool = False,
            sde_sample_freq: int = -1,
            target_kl: Optional[float] = None,
            stats_window_size: int = 100,
            tensorboard_log: Optional[str] = None,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            _init_setup_model: bool = True,
            classify_num: int = 0,
            term_num: int = 2,
    ):

        self.automata_states = env.env.observation_space['as'].n - 1
        self.horizon = env.env.horizon
        self.episode_reward = 0
        self.episode_length = 0
        if classify_num == 0:
            self.classify_num = self.automata_states - term_num
        else:
            self.classify_num = classify_num
        self.num_per_class = (self.automata_states - term_num) / self.classify_num

        # 调用父类的__init__方法，从而继承其属性和方法
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            target_kl=target_kl,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
            rollout_buffer_class=RBC_Rollout_Buffer,
            rollout_buffer_kwargs={'automata_states': self.classify_num, 'num_per_class':self.num_per_class}
        )

