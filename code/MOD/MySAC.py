import copy
import sys
import time
from copy import deepcopy
from stable_baselines3 import *
from stable_baselines3.sac.policies import CnnPolicy, MlpPolicy, MultiInputPolicy, SACPolicy
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union
from stable_baselines3.common.buffers import *
from stable_baselines3.td3.policies import TD3Policy
from torch.nn import functional as F
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
from ReplayBuffer import RBC_Replay_Buffer
import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.utils import safe_mean, should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv

class MySAC(SAC):

    def __init__(
            self,
            # 从SAC类中继承所有参数
            policy: Union[str, Type[SACPolicy]],
            env: Union[GymEnv, str],
            learning_rate: Union[float, Schedule] = 3e-4,
            buffer_size: int = 2_000_000,
            learning_starts: int = 100,
            batch_size: int = 0,
            tau: float = 0.005,
            gamma: float = 0.99,
            train_freq: Union[int, Tuple[int, str]] = 1,
            gradient_steps: int = 1,
            action_noise: Optional[ActionNoise] = None,
            replay_buffer_class: Optional[Type[ReplayBuffer]] = RBC_Replay_Buffer,
            replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
            optimize_memory_usage: bool = False,
            ent_coef: Union[str, float] = "auto",
            target_update_interval: int = 1,
            target_entropy: Union[str, float] = "auto",
            use_sde: bool = False,
            sde_sample_freq: int = -1,
            use_sde_at_warmup: bool = False,
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
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs={"automata_states": self.classify_num},
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            use_sde=use_sde,
        )

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.actor_targets = [copy.deepcopy(self.actor) for _ in range(self.classify_num)]
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target
        self.critic_targets = [copy.deepcopy(self.policy.critic_target) for _ in range(self.classify_num)]

    def _store_transition(
            self,
            replay_buffer: RBC_Replay_Buffer,
            buffer_action: np.ndarray,
            new_obs: Union[np.ndarray, Dict[str, np.ndarray]],
            reward: np.ndarray,
            dones: np.ndarray,
            infos: List[Dict[str, Any]],
    ) -> None:

        # Store only the unnormalized version
        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            reward_ = self._vec_normalize_env.get_original_reward()
        else:
            # Avoid changing the original ones
            self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

        # Avoid modification by reference
        next_obs = deepcopy(new_obs_)
        # As the VecEnv resets automatically, new_obs is already the
        # first observation of the next episode
        for i, done in enumerate(dones):
            if done and infos[i].get("terminal_observation") is not None:
                if isinstance(next_obs, dict):
                    next_obs_ = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs_ = self._vec_normalize_env.unnormalize_obs(next_obs_)
                    # Replace next obs for the correct envs
                    for key in next_obs.keys():
                        next_obs[key][i] = next_obs_[key]
                else:
                    next_obs[i] = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs[i] = self._vec_normalize_env.unnormalize_obs(next_obs[i, :])

        add_state = int(next_obs['as'].item() / self.num_per_class)
        if add_state >= self.classify_num:
            add_state = self.classify_num - 1
        replay_buffer.add(
            self._last_original_obs,
            next_obs,
            buffer_action,
            reward_,
            dones,
            infos,
            add_state=add_state,
        )

        # Save the unnormalized observation
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_

    def collect_rollouts(
            self,
            env: VecEnv,
            callback: BaseCallback,
            train_freq: TrainFreq,
            replay_buffer: ReplayBuffer,
            action_noise: Optional[ActionNoise] = None,
            learning_starts: int = 0,
            log_interval: Optional[int] = None,
            reset_states: List = None,
            events: List = None,
            dfa_state: int = -1
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ``ReplayBuffer``.

        :param reset_states:
        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param train_freq: How much experience to collect
            by doing rollouts of current policy.
            Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
            or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
            with ``<n>`` being an integer greater than 0.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """

        '''if dfa_state == 1:
            print('a')'''
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        # Vectorize action noise if needed
        if action_noise is not None and env.num_envs > 1 and not isinstance(action_noise, VectorizedActionNoise):
            action_noise = VectorizedActionNoise(action_noise, env.num_envs)

        if self.use_sde:
            self.actor.reset_noise(env.num_envs)

        callback.on_rollout_start()
        continue_training = True
        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.actor.reset_noise(env.num_envs)

            # Select action randomly or according to policy
            actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)

            dfa_state = '@q' + str(self._last_obs['as'].item() + 1)

            # Rescale and perform action
            new_obs, rewards, dones, infos = env.step(actions)
            next_dfa_state = '@q' + str(new_obs['as'].item() + 1)

            # reward shaping
            if dones:
                if rewards > 0:
                    rewards = np.array([100])
                else:
                    rewards = np.array([-10])
            else:
                if next_dfa_state != dfa_state:
                    rewards = np.array([10])
                else:
                    rewards = np.array([0])

            self.episode_reward += rewards.item()
            self.episode_length += 1
            original_dones = dones
            next_dfa_state = int(new_obs['as'].item() / self.num_per_class)
            if next_dfa_state != dfa_state and not original_dones:
                deque = reset_states[next_dfa_state]
                deque.append(new_obs)
                reset_states[next_dfa_state] = deque
                if not events[next_dfa_state].is_set():
                    events[next_dfa_state].set()
                    print(next_dfa_state, 'event set')
                dones = np.array([True])
                infos = [{}]
                infos[0]['episode'] = {'l': self.episode_length, 'r': self.episode_reward, 't': 3.0}
            if dones:
                self.episode_reward = 0
                self.episode_length = 0
            self.num_timesteps += 1
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if callback.on_step() is False:
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes,
                                     continue_training=False)

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            # Store data in replay buffer (normalized action and unnormalized observation)
            self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, original_dones, infos)

            if next_dfa_state != dfa_state and not original_dones:
                new_obs = env.reset()
            self._last_obs = new_obs
            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        pass
                        # print("dfa state is", dfa_state)
                        # self._dump_logs()

        callback.on_rollout_end()

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)

class MySAC_DummyVec(SAC):

    def __init__(
            self,
            # 从SAC类中继承所有参数
            policy: Union[str, Type[SACPolicy]],
            env: Union[GymEnv, str],
            learning_rate: Union[float, Schedule] = 3e-4,
            buffer_size: int = 2_000_000,
            learning_starts: int = 100,
            batch_size: int = 0,
            tau: float = 0.005,
            gamma: float = 0.99,
            train_freq: Union[int, Tuple[int, str]] = 1,
            gradient_steps: int = 1,
            action_noise: Optional[ActionNoise] = None,
            replay_buffer_class: Optional[Type[ReplayBuffer]] = RBC_Replay_Buffer,
            replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
            optimize_memory_usage: bool = False,
            ent_coef: Union[str, float] = "auto",
            target_update_interval: int = 1,
            target_entropy: Union[str, float] = "auto",
            use_sde: bool = False,
            sde_sample_freq: int = -1,
            use_sde_at_warmup: bool = False,
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

        self.automata_states = env.envs[0].env.observation_space['as'].n - 1
        self.horizon = env.envs[0].env.horizon
        self.episode_reward = np.zeros(env.num_envs)
        self.episode_length = np.zeros(env.num_envs)
        if classify_num == 0:
            self.classify_num = self.automata_states - term_num
        else:
            self.classify_num = classify_num
        self.num_per_class = (self.automata_states - term_num) / self.classify_num

        # 调用父类的__init__方法，从而继承其属性和方法
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs={"automata_states": self.classify_num},
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            use_sde=use_sde,
        )

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.actor_targets = [copy.deepcopy(self.actor) for _ in range(self.classify_num)]
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target
        self.critic_targets = [copy.deepcopy(self.policy.critic_target) for _ in range(self.classify_num)]

    def _store_transition(
            self,
            replay_buffer: RBC_Replay_Buffer,
            buffer_action: np.ndarray,
            new_obs: Union[np.ndarray, Dict[str, np.ndarray]],
            reward: np.ndarray,
            dones: np.ndarray,
            infos: List[Dict[str, Any]],
    ) -> None:

        # Store only the unnormalized version
        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            reward_ = self._vec_normalize_env.get_original_reward()
        else:
            # Avoid changing the original ones
            self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

        # Avoid modification by reference
        next_obs = deepcopy(new_obs_)
        # As the VecEnv resets automatically, new_obs is already the
        # first observation of the next episode
        for i, done in enumerate(dones):
            if done and infos[i].get("terminal_observation") is not None:
                if isinstance(next_obs, dict):
                    next_obs_ = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs_ = self._vec_normalize_env.unnormalize_obs(next_obs_)
                    # Replace next obs for the correct envs
                    for key in next_obs.keys():
                        next_obs[key][i] = next_obs_[key]
                else:
                    next_obs[i] = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs[i] = self._vec_normalize_env.unnormalize_obs(next_obs[i, :])

        for i in range(self.env.num_envs):
            add_state = int(new_obs['as'][i].item() / self.num_per_class)
            if add_state >= self.classify_num:
                add_state = self.classify_num - 1
            obs = {}
            next_new_obs = {}
            for key, value in self._last_original_obs.items():
                obs[key] = value[i]
            for key, value in next_obs.items():
                next_new_obs[key] = value[i]
            replay_buffer.add(
                obs,
                next_new_obs,
                buffer_action[i],
                reward_[i],
                dones[i],
                [infos[i]],
                add_state=add_state,
            )
        self._last_obs = new_obs
        # Save the unnormalized observation
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
        reset_states: List = None,
        events: List = None,
        dfa_state: int = -1
    ) -> RolloutReturn:

        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        # Vectorize action noise if needed
        if action_noise is not None and env.num_envs > 1 and not isinstance(action_noise, VectorizedActionNoise):
            action_noise = VectorizedActionNoise(action_noise, env.num_envs)

        if self.use_sde:
            self.actor.reset_noise(env.num_envs)

        callback.on_rollout_start()
        continue_training = True
        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.actor.reset_noise(env.num_envs)

            # Select action randomly or according to policy
            actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)

            # Rescale and perform action
            new_obs, rewards, dones, infos = env.step(actions)
            self.episode_reward += rewards
            self.episode_length += 1
            original_dones = dones
            next_dfa_state = new_obs['as'] // self.num_per_class

            for i in range(self.env.num_envs):
                if next_dfa_state[i] != dfa_state and not original_dones[i]:
                    idx = int(next_dfa_state[i])
                    deque = reset_states[idx]
                    obs = {}
                    for key, value in new_obs.items():
                        obs[key] = value[i]
                    deque.append(obs)
                    reset_states[idx] = deque
                    if not events[idx].is_set():
                        events[idx].set()

                    dones[i] = True
                    infos[i] = {}
                    infos[i]['episode'] = {'l':self.episode_length[i], 'r':self.episode_reward[i], 't':3.0}
                    new_obs = env.reset(idx=i)
                if dones[i]:
                    self.episode_reward[i] = 0
                    self.episode_length[i] = 0

            self.num_timesteps += 1
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if callback.on_step() is False:
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            # Store data in replay buffer (normalized action and unnormalized observation)
            self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, original_dones, infos)

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        pass
                        #print("dfa state is", dfa_state)
                        #self._dump_logs()

        callback.on_rollout_end()

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)