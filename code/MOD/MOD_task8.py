import argparse
import copy
import csv
import os
import re
import sys

from stable_baselines3 import *
from stable_baselines3.common.base_class import maybe_make_env, BaseAlgorithm
from stable_baselines3.common.off_policy_algorithm import SelfOffPolicyAlgorithm
from stable_baselines3.ppo.ppo import SelfPPO
from stable_baselines3.sac.policies import CnnPolicy, MlpPolicy, MultiInputPolicy, SACPolicy
from torch.nn import functional as F
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update, obs_as_tensor
from copy import deepcopy
from ReplayBuffer import RBC_Replay_Buffer
import numpy as np
import torch as th
from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer, RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.utils import safe_mean, should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv
from PRG_SB3 import *
from pyRDDLGym import RDDLEnv
import time
import threading
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
from MyPPO_OX import MyPPO
from frozen_lakeOX2 import FrozenLakeEnvOX2
from frozen_lake2 import FrozenLakeEnv2

time_start = time.time()

class PPOOX:
    def __init__(self, domain, eval_domain, instance, classify_num, term_num, log_path, model_class=MyPPO, action_noise=None, max_time=3000):
        env = FrozenLakeEnv1(desc=None, map_name="8x8", is_slippery=False)
        self.model = model_class("MultiInputPolicy", env, device='cpu', classify_num=classify_num, term_num=term_num)
        eval_env = FrozenLakeEnv1(desc=None, map_name="8x8", is_slippery=False)
        self.eval_env = BaseAlgorithm._wrap_env(maybe_make_env(eval_env, 1), 1, True)
        self.eval_interval = 4
        self.eval_time = 5
        self.classify_num = classify_num
        self.num_per_class = self.model.num_per_class
        self.log_path = log_path
        self.policies = [copy.deepcopy(self.model.policy) for _ in range(classify_num)]
        self.time_start = time.time()
        self.max_time = max_time
        self.current_time = time.time()

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        for i in range(self.classify_num):
            self.policies[i].set_training_mode(True)
            self.model._update_learning_rate(self.policies[i].optimizer)
        # Compute current clip range
        clip_range = self.model.clip_range(self.model._current_progress_remaining)
        # Optional: clip range for the value function
        if self.model.clip_range_vf is not None:
            clip_range_vf = self.model.clip_range_vf(self.model._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.model.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer

            '''rollout_datas = []
            dfas = []
            for rollout_data, dfa in self.model.rollout_buffer.get(self.model.batch_size):
                rollout_datas.append(rollout_data)
                dfas.append(dfa)
            print(dfas)'''
            for rollout_data, dfa in self.model.rollout_buffer.get(self.model.batch_size):
                if rollout_data.actions.numel() == 0:
                    break
                actions = rollout_data.actions
                if isinstance(self.model.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.model.use_sde:
                    self.model.policy.reset_noise(self.model.batch_size)

                values, log_prob, entropy = self.policies[dfa].evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.model.normalize_advantage and len(advantages) > 1:
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

                if self.model.clip_range_vf is None:
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

                loss = policy_loss + self.model.ent_coef * entropy_loss + self.model.vf_coef * value_loss

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.model.target_kl is not None and approx_kl_div > 1.5 * self.model.target_kl:
                    continue_training = False
                    if self.model.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policies[dfa].optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policies[dfa].parameters(), self.model.max_grad_norm)
                self.policies[dfa].optimizer.step()

            self.model._n_updates += 1
            if not continue_training:
                break

    def learn(
            self: SelfPPO,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 1,
            tb_log_name: str = "PPO",
            reset_num_timesteps: bool = True,
            progress_bar: bool = False,
    ) -> SelfPPO:
        total_timesteps, callback = self.model._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )
        assert self.model.env is not None
        train_time = 0
        while self.model.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(self.model.env, self.model.rollout_buffer,
                                                            self.model.n_steps)

            if continue_training is False:
                break
            self.model._update_current_progress_remaining(self.model.num_timesteps, total_timesteps)
            self.train()
            train_time += 1

            if time.time() - self.current_time > 30:
                self.evaluate()
                self.current_time = time.time()
        return self

    def evaluate(self):
        mean_reward, mean_length = 0.0, 0.0
        for times in range(self.eval_time):
            state = self.eval_env.reset()
            done = False
            while not done:
                dfa_state = int(state['ds'].item() / self.num_per_class)
                action, _ = self.policies[dfa_state].predict(state, deterministic=False)
                next_state, reward, done, info = self.eval_env.step(action)
                state = next_state
                mean_reward += reward
                mean_length += 1
        time_now = time.time()
        seconds = time_now - self.time_start
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        second = int(seconds % 60)
        hour = str(hours) + "h" + str(minutes) + "min" + str(second) + 's'
        data = [
            [hour, mean_reward / self.eval_time, mean_length / self.eval_time],
        ]
        with open(self.log_path, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(data)
        print('time:', hours, "h", minutes, "min", second, 's')
        print('mean_reward:', mean_reward / self.eval_time, 'mean_length:',
              mean_length / self.eval_time)

        if seconds >= self.max_time:
            sys.exit()

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
            n_rollout_steps: int
    ) -> bool:
        for i in range(self.classify_num):
            self.policies[i].set_training_mode(False)
        n_steps = 0
        rollout_buffer.reset()

        while n_steps < n_rollout_steps:
            dfa_state = int(self.model._last_obs['ds'].item())
            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self.model._last_obs, self.model.device)
                actions, values, log_probs = self.policies[dfa_state](obs_tensor)
            actions = actions.cpu().numpy()


            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.model.action_space, spaces.Box):
                if self.model.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.model.policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(actions, self.model.action_space.low, self.model.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)
            next_dfa_state = int(new_obs['ds'].item() / self.num_per_class)
            next_values = None
            special_add = (next_dfa_state != dfa_state and not dones)

            if special_add:
                with th.no_grad():
                    # Compute value for the last timestep
                    next_values = self.policies[next_dfa_state].predict_values(obs_as_tensor(new_obs, self.model.device))
            self.model.num_timesteps += env.num_envs
            n_steps += 1

            if isinstance(self.model.action_space, spaces.Discrete):
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
                    infos[idx]["terminal_observation"]['ds'] = np.array([infos[idx]["terminal_observation"]['ds']])
                    obs = self.match_observation_dims(infos[idx]["terminal_observation"], new_obs)
                    terminal_obs = self.model.policy.obs_to_tensor(obs)[0]
                    with th.no_grad():
                        terminal_value = self.model.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.model.gamma * terminal_value

            rollout_buffer.add(
                self.model._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                values,
                log_probs,
                dones,
                next_values,
            )
            self.model._last_obs = new_obs  # type: ignore[assignment]
            last_dfa_state = dfa_state

        dfa_state = int(self.model._last_obs['ds'].item())
        with th.no_grad():
            # Compute value for the last timestep
            values = self.policies[dfa_state].predict_values(obs_as_tensor(new_obs, self.model.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, special_add=special_add, dfa_state=last_dfa_state)

        return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-log', type=str, default='task1/task1_1.csv', help='log path')
    parser.add_argument('-i', type=str, default='inst31', help='inst name')
    parser.add_argument('-r', type=str, default='waterworld3', help='inst name')
    parser.add_argument('-t', type=int, default=3600, help='max_second')
    parser.add_argument('-c', type=int, default=4, help='classify_num')
    args = parser.parse_args()
    subpath = re.search(r'^[^\d]+', args.r)
    domain = 'benchmarks/' + subpath[0] + '/' + args.r + '.rddl'
    eval_domain = domain[:-5] + '_eval.rddl'
    instance = 'benchmarks/' + subpath[0] + '/' + args.i + '.rddl'
    log_path = 'log2/' + args.log
    data = [
        ['time', 'mean_reward', 'mean_length'],
    ]
    directory = os.path.dirname(log_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    with open(log_path, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(data)
    env = FlattenAction(RDDLEnv.RDDLEnv(domain=domain,
                                        instance=instance))
    eval_env = FlattenAction(RDDLEnv.RDDLEnv(domain=eval_domain,
                                             instance=instance))


    agent = PPOOX(domain, eval_domain, instance, classify_num=args.c, term_num=2, log_path=log_path, max_time=args.t)

    agent.learn(total_timesteps=30000000)