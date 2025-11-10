import copy
from copy import deepcopy

import numpy as np
from stable_baselines3 import SAC, DQN
from stable_baselines3.common.off_policy_algorithm import SelfOffPolicyAlgorithm
from stable_baselines3.common.utils import should_collect_more_steps, polyak_update, get_parameters_by_name
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.sac.policies import CnnPolicy, MlpPolicy, MultiInputPolicy, SACPolicy
import torch as th
from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit
from utils import *
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
from torch.nn import functional as F
from New_ReplayBuffer import Combined_Replay_Buffer

import warnings
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_linear_fn, get_parameters_by_name, polyak_update
from stable_baselines3.dqn.policies import CnnPolicy, DQNPolicy, MlpPolicy, MultiInputPolicy, QNetwork

SelfDQN = TypeVar("SelfDQN", bound="DQN")

class LowerSAC(SAC):
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
            replay_buffer_class: Optional[Type[ReplayBuffer]] = Combined_Replay_Buffer,
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
            dfa_text=None,
            upper_model=None,
    ):
        self.currentH = 0
        self.dfa = get_dfa(dfa_text)
        self_loops = list(nx.selfloop_edges(self.dfa))  # 获取所有自循环边
        self.dfa.remove_edges_from(self_loops)  # 删除自循环边
        # 删除所有指向 "@q2" 的边
        edges_to_q2 = list(self.dfa.in_edges("@q2"))  # 获取所有指向 "@q2" 的边
        self.dfa.remove_edges_from(edges_to_q2)  # 删除这些边
        self.upper_model = upper_model
        self.current_outedge = None
        self.sample_start = False
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
            replay_buffer_kwargs={"dfa": self.dfa},
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
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target
        for u, v in self.dfa.edges():
            self.dfa[u][v]['actor'] = copy.deepcopy(self.actor)
            self.dfa[u][v]['critic'] = copy.deepcopy(self.critic)
            self.dfa[u][v]['critic_target'] = copy.deepcopy(self.critic_target)

    def _setup_model(self) -> None:
        super(SAC, self)._setup_model()
        self._create_aliases()
        # Running mean and running var
        self.batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])
        # Target entropy is used when learning the entropy coefficient
        if self.target_entropy == "auto":
            # automatically set target entropy if needed
            self.target_entropy = float(-np.prod(self.env.action_space.shape).astype(np.float32))  # type: ignore
        else:
            # Force conversion
            # this will also throw an error for unexpected string
            self.target_entropy = float(self.target_entropy)

        # The entropy coefficient or entropy can be learned automatically
        # see Automating Entropy Adjustment for Maximum Entropy RL section
        # of https://arxiv.org/abs/1812.05905
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            # Default initial value of ent_coef when learned
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"

            for u, v in self.dfa.edges():
                self.dfa[u][v]['log_ent_coef'] = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)
                self.dfa[u][v]['ent_coef_optimizer'] = th.optim.Adam([self.dfa[u][v]['log_ent_coef']], lr=self.lr_schedule(1))

        else:
            # Force conversion to float
            # this will throw an error if a malformed string (different from 'auto')
            # is passed
            self.ent_coef_tensor = th.tensor(float(self.ent_coef), device=self.device)

    def predict(
            self,
            observation: Union[np.ndarray, Dict[str, np.ndarray]],
            u: str,
            v: str,
            state: Optional[Tuple[np.ndarray, ...]] = None,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:

        observation, vectorized_env = self.policy.obs_to_tensor(observation)

        with th.no_grad():
            actions = self.dfa[u][v]['actor'](observation, deterministic)
        # Convert to numpy, and reshape to the original action shape
        actions = actions.cpu().numpy().reshape((-1, *self.policy.action_space.shape))

        if isinstance(self.policy.action_space, spaces.Box):
            if self.policy.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.policy.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.policy.action_space.low, self.policy.action_space.high)

        # Remove batch dimension if needed
        if not vectorized_env:
            actions = actions.squeeze(axis=0)

        return actions, state

    def predict_eval(
            self,
            observation: Union[np.ndarray, Dict[str, np.ndarray]],
            u: str,
            v: str,
            state: Optional[Tuple[np.ndarray, ...]] = None,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:

        observation, vectorized_env = self.policy.obs_to_tensor(observation)

        with th.no_grad():
            actions = self.dfa[u][v]['actor'](observation, deterministic)
        # Convert to numpy, and reshape to the original action shape
        actions = actions.cpu().numpy().reshape((-1, *self.policy.action_space.shape))

        if isinstance(self.policy.action_space, spaces.Box):
            if self.policy.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.policy.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.policy.action_space.low, self.policy.action_space.high)

        # Remove batch dimension if needed
        if not vectorized_env:
            actions = actions.squeeze(axis=0)

        return actions, state


    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
        n_envs: int = 1,
        u: str = "",
        v: str = ""
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Select action randomly or according to policy
        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            # Warmup phase
            unscaled_action = np.array([self.action_space.sample() for _ in range(n_envs)])
        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
            assert self._last_obs is not None, "self._last_obs was not set"
            unscaled_action, _ = self.predict(self._last_obs, deterministic=False, u=u, v=v)
        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)

            # Add noise to the action (improve exploration)
            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action
        return action, buffer_action


    def _store_transition(
        self,
        replay_buffer: ReplayBuffer,
        buffer_action: np.ndarray,
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
        new_obs: Union[np.ndarray, Dict[str, np.ndarray]],
        reward: np.ndarray,
        dones: np.ndarray,
        infos: List[Dict[str, Any]],
        u: str,
        v: str,
    ) -> None:
        # Store only the unnormalized version
        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            reward_ = self._vec_normalize_env.get_original_reward()
        else:
            # Avoid changing the original ones
            new_obs_, reward_ = new_obs, reward

        # As the VecEnv resets automatically, new_obs is already the
        # first observation of the next episode
        replay_buffer.add(
            obs,  # type: ignore[arg-type]
            new_obs_,  # type: ignore[arg-type]
            buffer_action,
            reward_,
            dones,
            infos,
            u,
            v,
        )


    def train(self, gradient_steps: int, batch_size: int, u: str, v: str) -> None:
        actor = self.dfa[u][v]['actor']
        critic = self.dfa[u][v]['critic']
        critic_target = self.dfa[u][v]['critic_target']
        log_ent_coef = self.dfa[u][v]['log_ent_coef']
        ent_coef_optimizer = self.dfa[u][v]['ent_coef_optimizer']
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [actor.optimizer, critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, self._vec_normalize_env, u, v)  # type: ignore[union-attr]

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if ent_coef_optimizer is not None and log_ent_coef is not None:
                ent_coef = th.exp(log_ent_coef.detach())
                ent_coef_loss = -(log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None and ent_coef_optimizer is not None:
                ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                ent_coef_optimizer.step()

            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = actor.action_log_prob(replay_data.next_observations)
                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)  # for type checker
            critic_losses.append(critic_loss.item())  # type: ignore[union-attr]

            # Optimize the critic
            critic.optimizer.zero_grad()
            critic_loss.backward()
            critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Min over all critic networks
            q_values_pi = th.cat(critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            actor.optimizer.zero_grad()
            actor_loss.backward()
            actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(critic.parameters(), critic_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

class LowerDQN(DQN):
    def __init__(
            self,
            policy: Union[str, Type[DQNPolicy]],
            env: Union[GymEnv, str],
            learning_rate: Union[float, Schedule] = 1e-4,
            buffer_size: int = 1_000_000,  # 1e6
            learning_starts: int = 100,
            batch_size: int = 32,
            tau: float = 1.0,
            gamma: float = 0.99,
            train_freq: Union[int, Tuple[int, str]] = 4,
            gradient_steps: int = 1,
            replay_buffer_class: Optional[Type[ReplayBuffer]] = Combined_Replay_Buffer,
            replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
            optimize_memory_usage: bool = False,
            target_update_interval: int = 10000,
            exploration_fraction: float = 0.1,
            exploration_initial_eps: float = 1.0,
            exploration_final_eps: float = 0.05,
            max_grad_norm: float = 10,
            stats_window_size: int = 100,
            tensorboard_log: Optional[str] = None,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            _init_setup_model: bool = True,
            dfa_text=None,
            upper_model=None,
    ):
        self.currentH = 0
        self.dfa = get_dfa(dfa_text)
        self_loops = list(nx.selfloop_edges(self.dfa))  # 获取所有自循环边
        self.dfa.remove_edges_from(self_loops)  # 删除自循环边
        # 删除所有指向 "@q2" 的边
        edges_to_q2 = list(self.dfa.in_edges("@q2"))  # 获取所有指向 "@q2" 的边
        self.dfa.remove_edges_from(edges_to_q2)  # 删除这些边
        self.upper_model = upper_model
        self.current_outedge = None
        self.sample_start = False
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,  # 1e6
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            replay_buffer_class,
            {"dfa": self.dfa},
            optimize_memory_usage,
            target_update_interval,
            exploration_fraction,
            exploration_initial_eps,
            exploration_final_eps,
            max_grad_norm,
            stats_window_size,
            tensorboard_log,
            policy_kwargs,
            verbose,
            seed,
            device,
            _init_setup_model,
        )

    def _create_aliases(self) -> None:
        self.q_net = self.policy.q_net
        self.q_net_target = self.policy.q_net_target

        for u, v in self.dfa.edges():
            self.dfa[u][v]['q_net'] = copy.deepcopy(self.q_net)
            self.dfa[u][v]['q_net_target'] = copy.deepcopy(self.q_net_target)
            self.dfa[u][v]['policy'] = copy.deepcopy(self.policy)

    def train(self, gradient_steps: int, batch_size: int, u: str, v: str) -> None:
        policy = self.dfa[u][v]['policy']
        q_net = self.dfa[u][v]['q_net']
        q_net_target = self.dfa[u][v]['q_net_target']
        # Switch to train mode (this affects batch norm / dropout)
        policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env, u=u, v=v)  # type: ignore[union-attr]

            with th.no_grad():
                # Compute the next Q-values using the target network
                next_q_values = q_net_target(replay_data.next_observations)
                # Follow greedy policy: use the one with the highest value
                next_q_values, _ = next_q_values.max(dim=1)
                # Avoid potential broadcast issue
                next_q_values = next_q_values.reshape(-1, 1)
                # 1-step TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates
            current_q_values = q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            # Optimize the policy
            policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(policy.parameters(), self.max_grad_norm)
            policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

    def _store_transition(
        self,
        replay_buffer: ReplayBuffer,
        buffer_action: np.ndarray,
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
        new_obs: Union[np.ndarray, Dict[str, np.ndarray]],
        reward: np.ndarray,
        dones: np.ndarray,
        infos: List[Dict[str, Any]],
        u: str,
        v: str,
    ) -> None:
        # Store only the unnormalized version
        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            reward_ = self._vec_normalize_env.get_original_reward()
        else:
            # Avoid changing the original ones
            new_obs_, reward_ = new_obs, reward

        # As the VecEnv resets automatically, new_obs is already the
        # first observation of the next episode
        replay_buffer.add(
            obs,  # type: ignore[arg-type]
            new_obs_,  # type: ignore[arg-type]
            buffer_action,
            reward_,
            dones,
            infos,
            u,
            v,
        )

    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
        n_envs: int = 1,
        u: str = "",
        v: str = ""
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Select action randomly or according to policy
        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            # Warmup phase
            unscaled_action = np.array([self.action_space.sample() for _ in range(n_envs)])
        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
            assert self._last_obs is not None, "self._last_obs was not set"
            unscaled_action, _ = self.predict(observation=self._last_obs, deterministic=False, u=u, v=v)
        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)

            # Add noise to the action (improve exploration)
            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action
        return action, buffer_action

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        u: str,
        v: str,
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        if not deterministic and np.random.rand() < self.exploration_rate:
            if self.policy.is_vectorized_observation(observation):
                if isinstance(observation, dict):
                    n_batch = observation[next(iter(observation.keys()))].shape[0]
                else:
                    n_batch = observation.shape[0]
                action = np.array([self.action_space.sample() for _ in range(n_batch)])
            else:
                action = np.array(self.action_space.sample())
        else:
            policy = self.dfa[u][v]['policy']
            action, state = policy.predict(observation, state, episode_start, deterministic)
        return action, state