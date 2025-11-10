import copy
import sys
from copy import deepcopy
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.utils import should_collect_more_steps, polyak_update, get_parameters_by_name, \
    obs_as_tensor
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
from stable_baselines3.ppo.ppo import SelfPPO
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
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from typing import Any, ClassVar, Dict, Iterable, List, Optional, Tuple, Type, TypeVar, Union
from buffer.LowerRolloutBuffer import LowerRolloutBuffer

SelfBaseAlgorithm = TypeVar("SelfBaseAlgorithm", bound="BaseAlgorithm")


class LoadLowerPPO:
    def __init__(
            self,
            path: str,
            policy_ps,
            reset_ps,
            upper_policy_ps,
            evaluator,
            option_index,
            dfa_text,
            option_num,
            distances,
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
            policy_kwargs: Optional[Dict[str, Any]] = {},
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            _init_setup_model: bool = True,
    ):
        self.model = LowerPPO.load(path=path,
                                   policy_ps=policy_ps,
                                   reset_ps=reset_ps,
                                   upper_policy_ps=upper_policy_ps,
                                   evaluator=evaluator,
                                   option_index=option_index,
                                   dfa_text=dfa_text,
                                   option_num=option_num,
                                   distances=distances,
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
                                    _init_setup_model=_init_setup_model)
        self.model.time_start = time.time()

    def learn(self):
        self.model.learn()

    def upload_nets(self):
        self.model.upload_nets()

    def download_nets(self):
        self.model.download_nets()

class LowerPPO(PPO):

    def __init__(
            self,
            policy_ps,
            reset_ps,
            upper_policy_ps,
            evaluator,
            option_index,
            dfa_text,
            option_num,
            distances,
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
            policy_kwargs: Optional[Dict[str, Any]] = {},
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            _init_setup_model: bool = True,
    ):

        self.option_num = option_num
        self.policy_ps = policy_ps
        self.reset_ps = reset_ps
        self.upper_policy_ps = upper_policy_ps
        self.evaluator = evaluator
        self.upper_end_states = None
        self.dfa = get_dfa(dfa_text)
        self.option_index = option_index
        self.currentH = 0
        self.max_subgoal_steps = 300
        self.initial_infos = [[] for _ in range(self.option_num)]
        self.upper_logger = {}
        self.teacher_logger = {}
        self.upper_policy = ray.get(self.upper_policy_ps.get_policy.remote())
        self.distances = distances
        self.del_keys_for_save = ['policy_ps', 'reset_ps', 'upper_policy_ps', 'evaluator', 'upper_buffer_action', 'upper_end_states',
                                  'upper_current_obs', 'upper_value', 'upper_log_prob', 'dfa', 'option_index', 'option_num', 'currentH',
                                  'max_subgoal_steps', 'initial_infos', 'policy_kwargs',
                                 'upper_exp_buffer', 'upper_exp_ps', 'upper_logger', 'upper_policy', 'teacher_logger', 'distances']
        # self.if_store_env = if_store_env
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
            rollout_buffer_class=LowerRolloutBuffer,
        )
        if _init_setup_model:
            self.policies = [copy.deepcopy(self.policy) for _ in range(self.option_num)]

    def upload_nets(self):
        ray.wait([self.policy_ps.set_network.remote(self.option_index, self.policy)])

    def download_nets(self):
        self.policies = ray.get(self.policy_ps.get_networks.remote())

    def learn(
            self: SelfPPO,
            total_timesteps: int = 500000000,
            callback: MaybeCallback = None,
            log_interval: int = 1,
            tb_log_name: str = "PPO",
            reset_num_timesteps: bool = True,
            progress_bar: bool = False,
    )-> SelfPPO:
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )
        assert self.env is not None
        train_time = 0
        stop = False

        now_time = time.time()
        while not stop:
            self.collect_rollouts(self.env, self.rollout_buffer, self.n_steps)
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)
            self.train()
            train_time += 1
            ray.wait([self.policy_ps.set_network.remote(self.option_index, self.policy)])
            self.reset_ps.set_states.remote(self.initial_infos)
            self.policies = ray.get(self.policy_ps.get_networks.remote())
            self.upper_policy = ray.get(self.upper_policy_ps.get_policy.remote())
            self.initial_states = [[] for _ in range(self.option_num)]
            stop = ray.get(self.evaluator.select_training_stop.remote())

            if time.time() - now_time > 30:
                log = {}
                for key, value in self.teacher_logger.items():
                    log[key] = (value[1], value[2])
                print(self.option_index, log)
                for key, value in self.teacher_logger.items():
                    log[key] = value[1]
                self.upper_policy_ps.set_edges_reward.remote(log, self.option_index)

                log = {}
                for key, value in self.upper_logger.items():
                    log[key] = value[1]
                print(self.option_index, log)
                self.reset_ps.set_upper_logger.remote(log, self.option_index)
                now_time = time.time()

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

    def set_upper_logger(self, logger, dfa_state, next_dfa_state, if_success):
        success_time = logger.get((dfa_state, next_dfa_state), [[], 0, 0])
        success_time[0].append(int(if_success))
        if len(success_time[0]) > 200:
            success_time[0].pop(0)
        success_time[1] = sum(success_time[0]) / len(success_time[0])
        success_time[2] += 1
        logger[(dfa_state, next_dfa_state)] = success_time

    def get_lower_model_index(self, obs, deterministic=False):
        dfa_state = '@q' + str(obs['ds'].item() + 1)
        next_edge, best_option = self.upper_policy.predict(dfa_state, deterministic=False)
        '''try:
            assert 'p' + str(best_option + 1) in extract_true_predicates(self.dfa[dfa_state][next_edge[1]]['formula'], self.dfa[dfa_state][next_edge[0]]['formula'])
        except Exception as e:
            print(e)
            print(next_edge, best_option)'''
        return best_option, None, [next_edge[1]], None, None, None

    def match_observation_dims(self, terminal_obs, new_obs):
        matched_obs = {}
        for key in terminal_obs.keys():
            if key in new_obs:
                # 获取目标形状
                target_shape = new_obs[key].shape
                # 调整数组形状
                matched_obs[key] = np.reshape(terminal_obs[key], target_shape)
            else:
                matched_obs[key] = terminal_obs[key]
        return matched_obs

    def collect_rollouts(
        self,
        env: VecEnv,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:

        if self.upper_end_states is None:
            self.upper_end_states = self.env.envs[0].env.get_wrapper_attr('end_states')
            self.currentH = 0

        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            dfa_state = '@q' + str(self._last_obs['ds'].item() + 1)
            new_obs, rewards, dones, infos = env.step(clipped_actions) # step
            original_dones = dones.copy()
            self.currentH += 1
            next_dfa_state = '@q' + str(new_obs['ds'].item() + 1) if not dones else '@q' + str(infos[0]['terminal_observation']['ds'].item() + 1)

            next_values = None
            # reshape done
            if (next_dfa_state != dfa_state and not(next_dfa_state in self.upper_end_states)) or self.currentH == self.max_subgoal_steps:
                dones = np.array([True])

            # reshape reward
            if next_dfa_state in self.upper_end_states:
                # rewards += 100 / (self.distances[next_dfa_state] + 1) - 100 / (self.distances[dfa_state] + 1)
                rewards = (0 * int(next_dfa_state == '@q1') + 10 * int(
                    next_dfa_state == '@q3' or next_dfa_state == '@q4' or next_dfa_state == '@q5') + 100 * int(
                    next_dfa_state == '@q6') - 0 * int(dfa_state == '@q1') - 10 * int(
                    dfa_state == '@q3' or dfa_state == '@q4' or dfa_state == '@q5') - 100 * int(dfa_state == '@q6'))
                rewards = np.array([rewards], dtype=np.float64)
            else:
                rewards += np.array([0], dtype=np.float64)

            # set upper log
            if dones or next_dfa_state in self.upper_end_states:
                if not (next_dfa_state != dfa_state and not (
                        next_dfa_state in self.upper_end_states) and not original_dones):
                    self.set_teacher_logger(self.teacher_logger, dfa_state, self.upper_end_states[0],
                                            rewards.item(), next_dfa_state in self.upper_end_states)
                self.set_upper_logger(self.upper_logger, dfa_state, self.upper_end_states[0],
                                      next_dfa_state in self.upper_end_states)

            # save initial obs
            special_add = False
            if next_dfa_state in self.upper_end_states and not dones:
                next_index, _,  upper_end_states, upper_buffer_action, upper_value, upper_log_prob = self.get_lower_model_index(new_obs)
                self.initial_infos[next_index].append((next_dfa_state, [new_obs, upper_end_states]))
                # policy ganyu
                special_add = True
                with th.no_grad():
                    # Compute value for the last timestep
                    next_values = self.policies[next_index].predict_values(obs_as_tensor(new_obs, self.device))

            self.num_timesteps += env.num_envs
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
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
                    '''print(infos[idx]["terminal_observation"])
                    print(infos[idx])
                    print(self.policy.obs_to_tensor(infos[idx]["terminal_observation"]))'''
                    infos[idx]["terminal_observation"]['ds'] = np.array([infos[idx]["terminal_observation"]['ds']])
                    obs = self.match_observation_dims(infos[idx]["terminal_observation"], new_obs)
                    terminal_obs = self.policy.obs_to_tensor(obs)[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]

                    rewards[idx] += self.gamma * terminal_value.numpy()
                    '''try:
                        rewards[idx] += self.gamma * terminal_value
                    except:
                        print(rewards[idx], terminal_value, terminal_obs)
                        rewards[idx] += self.gamma * terminal_value.numpy()'''

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                values,
                log_probs,
                dones,
                next_values,
            )

            # reset lower env
            if dones or next_dfa_state in self.upper_end_states:
                self._last_obs = self.env.reset()
                self.currentH = 0
                self.upper_end_states = self.env.envs[0].env.get_wrapper_attr('end_states')
            else:
                self._last_obs = new_obs


        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))
        rollout_buffer.compute_returns_and_advantage(values, special_add)

        return True

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

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

        '''if "policy_kwargs" in kwargs and kwargs["policy_kwargs"] != data["policy_kwargs"]:
            raise ValueError(
                f"The specified policy kwargs do not equal the stored policy kwargs."
                f"Stored kwargs: {data['policy_kwargs']}, specified kwargs: {kwargs['policy_kwargs']}"
            )'''

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




