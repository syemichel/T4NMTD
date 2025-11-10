import copy
from copy import deepcopy
from stable_baselines3 import SAC
from stable_baselines3.common.off_policy_algorithm import SelfOffPolicyAlgorithm
from stable_baselines3.common.utils import should_collect_more_steps, polyak_update, get_parameters_by_name, \
    obs_as_tensor
from stable_baselines3.sac.policies import CnnPolicy, MlpPolicy, MultiInputPolicy, SACPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit
from torch.nn import functional as F
from buffer.ReplayBuffer import *
import ray
import io
import pathlib
import time
import warnings
import numpy as np
import torch as th
from util.DFA import *
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.save_util import load_from_zip_file, recursive_getattr, recursive_setattr, save_to_zip_file
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule, TensorDict
from stable_baselines3.common.utils import (
    check_for_correct_spaces,
    get_device,
    get_schedule_fn,
    get_system_info,
    set_random_seed,
    update_learning_rate,
)
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    VecNormalize,
    VecTransposeImage,
    is_vecenv_wrapped,
    unwrap_vec_normalize,
)
from stable_baselines3.common.vec_env.patch_gym import _convert_space, _patch_env
from typing import Any, ClassVar, Dict, Iterable, List, Optional, Tuple, Type, TypeVar, Union
SelfBaseAlgorithm = TypeVar("SelfBaseAlgorithm", bound="BaseAlgorithm")

class LoadLowerSAC:
    def __init__(
            self,
            # 从SAC类中继承所有参数
            policy: Union[str, Type[SACPolicy]],
            path: str,
            env: Union[GymEnv, str],
            actor_ps,
            critic_ps,
            reset_ps,
            upper_model_ps,
            option_index,
            evaluator,
            dfa_text,
            learning_rate: Union[float, Schedule] = 3e-4,
            buffer_size: int = 2_000_000,
            learning_starts: int = 100,
            batch_size: int = 0,
            tau: float = 0.005,
            gamma: float = 0.99,
            train_freq: Union[int, Tuple[int, str]] = 1,
            gradient_steps: int = 1,
            action_noise: Optional[ActionNoise] = None,
            replay_buffer_class: Optional[Type[ReplayBuffer]] = Single_Replay_Buffer,
            use_sde: bool = False,
            stats_window_size: int = 100,
            tensorboard_log: Optional[str] = None,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            _init_setup_model: bool = True,
            option_num: int = 3,
            training_time=100,
    ):
        self.model = LowerSAC.load(path=path,
                             env=env,
                             policy="MultiInputPolicy",
                             actor_ps=actor_ps,
                             critic_ps=critic_ps,
                             reset_ps=reset_ps,
                             option_index=option_index,
                             dfa_text=dfa_text,
                             upper_model_ps=upper_model_ps,
                             evaluator=evaluator,
                             verbose=1,
                             learning_starts=1000,
                             learning_rate=3e-4,
                             batch_size=256,
                             train_freq=1,
                             device='cpu',
                             option_num=option_num)
        self.model.time_start = time.time()

    def learn(self):
        self.model.learn()

    def upload_nets(self):
        self.model.upload_nets()

    def download_nets(self):
        self.model.download_nets()

class LowerSAC(SAC):

    def __init__(
            self,
            # 从SAC类中继承所有参数
            policy: Union[str, Type[SACPolicy]],
            env: Union[GymEnv, str],
            actor_ps,
            critic_ps,
            reset_ps,
            upper_model_ps,
            evaluator,
            option_index,
            dfa_text,
            learning_rate: Union[float, Schedule] = 3e-4,
            buffer_size: int = 2_000_000,
            learning_starts: int = 100,
            batch_size: int = 0,
            tau: float = 0.005,
            gamma: float = 0.99,
            train_freq: Union[int, Tuple[int, str]] = 1,
            gradient_steps: int = 1,
            action_noise: Optional[ActionNoise] = None,
            replay_buffer_class: Optional[Type[ReplayBuffer]] = Single_Replay_Buffer,
            use_sde: bool = False,
            stats_window_size: int = 100,
            tensorboard_log: Optional[str] = None,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            _init_setup_model: bool = True,
            option_num: int = 3,
            if_store_env = False,
    ):

        self.option_num = option_num
        self.actor_ps = actor_ps
        self.critic_ps = critic_ps
        self.reset_ps = reset_ps
        self.upper_model_ps = upper_model_ps
        self.evaluator = evaluator
        self.upper_end_states = None
        self.dfa = get_dfa(dfa_text)
        self.option_index = option_index
        self.currentH = 0
        self.max_subgoal_steps = 300
        self.initial_infos = [[] for _ in range(self.option_num)]
        self.upper_exp_buffer = []
        self.upper_logger = {}
        self.teacher_logger = {}
        self.upper_policy = ray.get(self.upper_model_ps.get_policy.remote())
        self.del_keys_for_save = ['actor_ps', 'critic_ps', 'reset_ps', 'upper_policy_ps', 'evaluator', 'upper_buffer_action', 'upper_end_states',
                                  'upper_current_obs', 'upper_value', 'upper_log_prob', 'dfa', 'option_index', 'option_num', 'currentH', 'max_subgoal_steps', 'initial_infos',
                                  'upper_exp_buffer', 'upper_exp_ps', 'upper_logger', 'upper_policy', 'actor_target', 'actor_targets', 'critic_targets', 'upper_model_ps',
                                    'teacher_logger']
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
            replay_buffer_kwargs={"option_num": option_num, "option_index": option_index, "learn_start": 500, "upper_policy": self.upper_policy, "dfa": self.dfa},
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            use_sde=use_sde,
        )

    def upload_nets(self):
        ray.wait([self.actor_ps.set_network.remote(self.option_index, self.actor_target)])
        ray.wait([self.critic_ps.set_network.remote(self.option_index, self.critic_target)])

    def download_nets(self):
        self.actor_targets = ray.get(self.actor_ps.get_networks.remote())
        self.critic_targets = ray.get(self.critic_ps.get_networks.remote())

    def learn(
        self,
        total_timesteps: int = 100000000000,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "run",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfOffPolicyAlgorithm:
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )
        stop = False
        while not stop:
            training_stop = self.collect_rollouts(
                self.env,
                train_freq=self.train_freq,
                action_noise=self.action_noise,
                callback=callback,
                learning_starts=self.learning_starts,
                replay_buffer=self.replay_buffer,
                log_interval=log_interval,
            )
            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts and not training_stop:
                self.train(batch_size=self.batch_size, gradient_steps=self.gradient_steps)

            if (self.num_timesteps+1) % 300 == 0:
                ray.wait([self.actor_ps.set_network.remote(self.option_index, self.actor_targets[self.option_index])])
                ray.wait([self.critic_ps.set_network.remote(self.option_index, self.critic_targets[self.option_index])])
                self.reset_ps.set_states.remote(self.initial_infos)
                self.actor_targets = ray.get(self.actor_ps.get_networks.remote())
                self.critic_targets = ray.get(self.critic_ps.get_networks.remote())
                self.upper_policy = ray.get(self.upper_model_ps.get_policy.remote())
                self.initial_states = [[] for _ in range(self.option_num)]
                self.replay_buffer.upper_policy = self.upper_policy
                stop = ray.get(self.evaluator.select_training_stop.remote())
                # print(self.option_index, self.num_timesteps+1, self.upper_logger)


            if (self.num_timesteps + 1) % 3000 == 0:
                log = {}
                for key, value in self.teacher_logger.items():
                    log[key] = (value[1], value[2])
                print(self.option_index, log)
                for key, value in self.teacher_logger.items():
                    log[key] = value[1]
                self.upper_model_ps.set_edges_reward.remote(log, self.option_index)

                log = {}
                for key, value in self.upper_logger.items():
                    log[key] = value[1]
                print(self.option_index, log)
                self.reset_ps.set_upper_logger.remote(log, self.option_index)

        print('Lower model' + str(self.option_index) + ' stop.')

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.actor_targets = ray.get(self.actor_ps.get_networks.remote())
        self.critic_targets = ray.get(self.critic_ps.get_networks.remote())
        self.upper_policy = ray.get(self.upper_model_ps.get_policy.remote())
        self.critic = self.policy.critic

        # copy actor_target and critic_target
        self.policy.actor_target = self.actor_targets[self.option_index]
        self.actor_target = self.policy.actor_target
        self.policy.critic_target = self.critic_targets[self.option_index]
        self.critic_target = self.policy.critic_target

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

            # Note: we optimize the log of the entropy coeff which is slightly different from the paper
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            self.log_ent_coefs = [None for _ in range(self.option_num)]
            self.ent_coef_optimizers = [None for _ in range(self.option_num)]
            for i in range(self.option_num):
                self.log_ent_coefs[i] = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)
                self.ent_coef_optimizers[i] = th.optim.Adam([self.log_ent_coefs[i]], lr=self.lr_schedule(1))

            self.log_ent_coef = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef], lr=self.lr_schedule(1))
        else:
            # Force conversion to float
            # this will throw an error if a malformed string (different from 'auto')
            # is passed
            self.ent_coef_tensor = th.tensor(float(self.ent_coef), device=self.device)

    def predict(
            self,
            observation: Union[np.ndarray, Dict[str, np.ndarray]],
            state: Optional[Tuple[np.ndarray, ...]] = None,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:

        observation, vectorized_env = self.policy.obs_to_tensor(observation)

        with th.no_grad():
            actions = self.actor(observation, deterministic)
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

    def set_upper_logger(self, logger, dfa_state, current_option_index, if_success):
        success_time = logger.get(dfa_state, [[], 0, 0])
        success_time[0].append(int(if_success))
        if len(success_time[0]) > 200:
            success_time[0].pop(0)
        success_time[1] = sum(success_time[0]) / len(success_time[0])
        success_time[2] += 1
        logger[dfa_state] = success_time

    def _store_transition(
        self,
        replay_buffer: ReplayBuffer,
        buffer_action: np.ndarray,
        new_obs: Union[np.ndarray, Dict[str, np.ndarray]],
        reward: np.ndarray,
        dones: np.ndarray,
        infos: List[Dict[str, Any]],
        next_option_index=None,
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

        replay_buffer.add(
            self._last_original_obs,  # type: ignore[arg-type]
            next_obs,  # type: ignore[arg-type]
            buffer_action,
            reward_,
            dones,
            infos,
            next_option_index,
        )

        self._last_obs = new_obs
        # Save the unnormalized observation
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_

    def get_buffer_action(self, action):
        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.upper_action_space, spaces.Box):
            scaled_action = self.upper_policy.scale_action(action)
            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = self.upper_policy.unscale_action(scaled_action)
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = action
            action = buffer_action

        return buffer_action

    def get_lower_model_index(self, obs, deterministic=False):
        dfa_state = '@q' + str(obs['ds'].item() + 1)
        next_edge, best_option = self.upper_policy.predict(dfa_state, deterministic=False)
        '''try:
            assert 'p' + str(best_option + 1) in extract_true_predicates(self.dfa[dfa_state][next_edge[1]]['formula'],
                                                                         self.dfa[dfa_state][next_edge[0]]['formula'])
        except Exception as e:
            print(e)
            print(next_edge, best_option)'''
        return best_option, [next_edge[1]]

    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
        n_envs: int = 1,
        determinstic: bool = False
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
            unscaled_action, _ = self.predict(self._last_obs, deterministic=determinstic)
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

    def set_teacher_logger(self, logger, dfa_state, next_dfa_state, reward, if_success):
        success_time = logger.get((dfa_state, next_dfa_state), [[], 0, 0])
        if if_success:
            success_time[0].append(reward)
        else:
            success_time[0].append(-9999)
        if len(success_time[0]) > 200:
            success_time[0].pop(0)
        success_time[1] = sum(success_time[0]) / len(success_time[0])
        success_time[2] += 1
        logger[(dfa_state, next_dfa_state)] = success_time


    def collect_rollouts(
            self,
            env: VecEnv,
            callback: BaseCallback,
            train_freq: TrainFreq,
            replay_buffer: ReplayBuffer,
            action_noise: Optional[ActionNoise] = None,
            learning_starts: int = 0,
            log_interval: Optional[int] = 100,
    ):
        if self.upper_end_states is None:
            self.upper_end_states = self.env.envs[0].env.get_wrapper_attr('end_states')
        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0
        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."
        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        if self.use_sde:
            self.actor.reset_noise(env.num_envs)

        callback.on_rollout_start()
        continue_training = True
        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                self.actor.reset_noise(env.num_envs)

            # select deterministic
            dfa_state = '@q' + str(self._last_obs['ds'].item() + 1)
            # Select action randomly or according to policy
            actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs,
                                                          determinstic=False)
            # Rescale and perform action
            new_obs, rewards, dones, infos = env.step(actions)
            original_dones = dones.copy()
            self.currentH += 1
            # reward shaping

            next_dfa_state = '@q' + str(new_obs['ds'].item() + 1) if not dones else '@q' + str(infos[0]['terminal_observation']['ds'] + 1)

            next_values = None
            # set dones
            if (next_dfa_state != dfa_state and not(next_dfa_state in self.upper_end_states)) or self.currentH == self.max_subgoal_steps:
                dones = np.array([True])
                infos[0]['terminal_observation'] = new_obs
            # set upper log
            if dones or next_dfa_state in self.upper_end_states:
                self.set_upper_logger(self.upper_logger, dfa_state, self.option_index,
                                      (next_dfa_state in self.upper_end_states) or (dones and rewards > 0))

            # reward shaping
            if next_dfa_state in self.upper_end_states or rewards > 0:
                rewards = np.array([10])
            elif dones and rewards < 1:
                rewards = np.array([0])
            else:
                rewards = np.array([0])
            next_index = None

            # set upper logger
            if dones or next_dfa_state in self.upper_end_states:
                if not (next_dfa_state != dfa_state and not (next_dfa_state in self.upper_end_states) and not original_dones):
                    self.set_teacher_logger(self.teacher_logger, dfa_state, self.upper_end_states[0], -self.currentH, next_dfa_state in self.upper_end_states)
                self.set_upper_logger(self.upper_logger, dfa_state, self.option_index,
                                          next_dfa_state in self.upper_end_states)

            # 若迁移成功则预测next option index and add infos in self.initial_infos
            if next_dfa_state in self.upper_end_states and not dones:
                next_index, upper_end_states = self.get_lower_model_index(new_obs)
                self.initial_infos[next_index].append((next_dfa_state, [new_obs, upper_end_states]))

            self.num_timesteps += env.num_envs
            num_collected_steps += 1
            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if not callback.on_step():
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes,
                                     continue_training=False)

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)


            # Store data in replay buffer (normalized action and unnormalized observation)
            self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos, None)

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            self._on_step()

            # reset lower env
            if dones or next_dfa_state in self.upper_end_states:
                self._last_obs = self.env.reset()
                self.currentH = 0
                self.upper_end_states = self.upper_buffer_action = None

            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

        callback.on_rollout_end()

        return False


    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        op = self.option_index
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizers[op]]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            dfa_state = '@q' + str(self._last_obs['ds'].item() + 1)
            replay_data, nums = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env, dfa_state=dfa_state)  # type: ignore[union-attr]
            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizers[op] is not None and self.log_ent_coefs[op] is not None:
                ent_coef = th.exp(self.log_ent_coefs[op].detach())
                ent_coef_loss = -(self.log_ent_coefs[op] * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None and self.ent_coef_optimizers[op] is not None:
                self.ent_coef_optimizers[op].zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizers[op].step()

            target_q_values = th.tensor([], device=self.device)
            index = 0
            for i in range(self.option_num):
                if nums[i] == 0:
                    continue
                next_observations = {}
                for key, value in replay_data.next_observations.items():
                    next_observations[key] = value[index:index + nums[i]]
                with th.no_grad():
                    # Select action according to policy
                    next_actions, next_log_prob = self.actor_targets[i].action_log_prob(next_observations)
                    # Compute the next Q values: min over all critics targets
                    next_q_values = th.cat(self.critic_targets[i](next_observations, next_actions), dim=1)
                    next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                    # add entropy term
                    next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                    # td error + entropy term
                    target_q_values1 = replay_data.rewards[index:index + nums[i]] + (
                            1 - replay_data.dones[index:index + nums[i]]) * self.gamma * next_q_values
                    target_q_values = th.cat((target_q_values, target_q_values1), dim=0)
                index += nums[i]

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)  # for type checker
            critic_losses.append(critic_loss.item())  # type: ignore[union-attr]

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Min over all critic networks
            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_targets[op].parameters(),
                              self.tau)
                polyak_update(self.actor.parameters(), self.actor_targets[op].parameters(),
                              self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

    def save(
        self,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        exclude: Optional[Iterable[str]] = None,
        include: Optional[Iterable[str]] = None,
    ) -> None:
        """
        Save all the attributes of the object and the model parameters in a zip-file.

        :param path: path to the file where the rl agent should be saved
        :param exclude: name of parameters that should be excluded in addition to the default ones
        :param include: name of parameters that might be excluded but should be included anyway
        """
        # copy actor_target and critic_target
        self.policy.actor_target = self.actor_targets[self.option_index]
        self.actor_target = self.policy.actor_target
        self.policy.critic_target = self.critic_targets[self.option_index]
        self.critic_target = self.policy.critic_target


        # Copy parameter list so we don't mutate the original dict
        data = self.__dict__.copy()

        # Exclude is union of specified parameters (if any) and standard exclusions
        if exclude is None:
            exclude = []
        exclude = set(exclude).union(self._excluded_save_params())

        # Do not exclude params if they are specifically included
        if include is not None:
            exclude = exclude.difference(include)
        state_dicts_names, torch_variable_names = self._get_torch_save_params()
        all_pytorch_variables = state_dicts_names + torch_variable_names
        for torch_var in all_pytorch_variables:
            # We need to get only the name of the top most module as we'll remove that
            var_name = torch_var.split(".")[0]
            # Any params that are in the save vars must not be saved by data
            exclude.add(var_name)


        # Remove parameter entries of parameters which are to be excluded
        for param_name in exclude:
            data.pop(param_name, None)

        # Build dict of torch variables
        pytorch_variables = None
        if torch_variable_names is not None:
            pytorch_variables = {}
            for name in torch_variable_names:
                attr = recursive_getattr(self, name)
                pytorch_variables[name] = attr

        # Build dict of state_dicts
        params_to_save = self.get_parameters()
        data = {k: v for k, v in data.items() if k not in self.del_keys_for_save}
        '''for key in data.keys():
            print(key)
        print(path, params_to_save['policy']['actor_target.mu.bias'])'''
        save_to_zip_file(path, data=data, params=params_to_save, pytorch_variables=pytorch_variables)
        return True


    @classmethod
    def load(  # noqa: C901
        cls: Type[SelfBaseAlgorithm],
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        env: Optional[GymEnv] = None,
        device: Union[th.device, str] = "auto",
        custom_objects: Optional[Dict[str, Any]] = None,
        print_system_info: bool = False,
        force_reset: bool = True,
        **kwargs,
    ) -> SelfBaseAlgorithm:
        if print_system_info:
            print("== CURRENT SYSTEM INFO ==")
            get_system_info()

        data, params, pytorch_variables = load_from_zip_file(
            path,
            device=device,
            custom_objects=custom_objects,
            print_system_info=print_system_info,
        )

        assert data is not None, "No data found in the saved file"
        assert params is not None, "No params found in the saved file"

        # Remove stored device information and replace with ours
        if "policy_kwargs" in data:
            if "device" in data["policy_kwargs"]:
                del data["policy_kwargs"]["device"]
            # backward compatibility, convert to new format
            if "net_arch" in data["policy_kwargs"] and len(data["policy_kwargs"]["net_arch"]) > 0:
                saved_net_arch = data["policy_kwargs"]["net_arch"]
                if isinstance(saved_net_arch, list) and isinstance(saved_net_arch[0], dict):
                    data["policy_kwargs"]["net_arch"] = saved_net_arch[0]

        if "policy_kwargs" in kwargs and kwargs["policy_kwargs"] != data["policy_kwargs"]:
            raise ValueError(
                f"The specified policy kwargs do not equal the stored policy kwargs."
                f"Stored kwargs: {data['policy_kwargs']}, specified kwargs: {kwargs['policy_kwargs']}"
            )

        if "observation_space" not in data or "action_space" not in data:
            raise KeyError("The observation_space and action_space were not given, can't verify new environments")

        # Gym -> Gymnasium space conversion
        for key in {"observation_space", "action_space"}:
            data[key] = _convert_space(data[key])

        if env is not None:
            # Wrap first if needed
            env = cls._wrap_env(env, data["verbose"])
            # Check if given env is valid
            check_for_correct_spaces(env, data["observation_space"], data["action_space"])
            # Discard `_last_obs`, this will force the env to reset before training
            # See issue https://github.com/DLR-RM/stable-baselines3/issues/597
            if force_reset and data is not None:
                data["_last_obs"] = None
            # `n_envs` must be updated. See issue https://github.com/DLR-RM/stable-baselines3/issues/1018
            if data is not None:
                data["n_envs"] = env.num_envs
        else:
            # Use stored env, if one exists. If not, continue as is (can be used for predict)
            if "env" in data:
                env = data["env"]

        model = cls(
            **dict(
                kwargs,
                policy=data["policy_class"],
                env=env,
                device=device,
                _init_setup_model=False
            )
        )

        # load parameters
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model._setup_model()

        try:
            # put state_dicts back in place
            model.set_parameters(params, exact_match=True, device=device)
        except RuntimeError as e:
            # Patch to load Policy saved using SB3 < 1.7.0
            # the error is probably due to old policy being loaded
            # See https://github.com/DLR-RM/stable-baselines3/issues/1233
            if "pi_features_extractor" in str(e) and "Missing key(s) in state_dict" in str(e):
                model.set_parameters(params, exact_match=False, device=device)
                warnings.warn(
                    "You are probably loading a model saved with SB3 < 1.7.0, "
                    "we deactivated exact_match so you can save the model "
                    "again to avoid issues in the future "
                    "(see https://github.com/DLR-RM/stable-baselines3/issues/1233 for more info). "
                    f"Original error: {e} \n"
                    "Note: the model should still work fine, this only a warning."
                )
            else:
                raise e
        # put other pytorch variables back in place
        if pytorch_variables is not None:
            for name in pytorch_variables:
                # Skip if PyTorch variable was not defined (to ensure backward compatibility).
                # This happens when using SAC/TQC.
                # SAC has an entropy coefficient which can be fixed or optimized.
                # If it is optimized, an additional PyTorch variable `log_ent_coef` is defined,
                # otherwise it is initialized to `None`.
                if pytorch_variables[name] is None:
                    continue
                # Set the data attribute directly to avoid issue when using optimizers
                # See https://github.com/DLR-RM/stable-baselines3/issues/391
                recursive_setattr(model, f"{name}.data", pytorch_variables[name].data)

        # Sample gSDE exploration matrix, so it uses the right device
        # see issue #44
        if model.use_sde:
            model.policy.reset_noise()  # type: ignore[operator]
        return model




