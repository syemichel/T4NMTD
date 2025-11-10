import argparse
import collections
import copy
import os
import re
import sys

from stable_baselines3.ppo.ppo import SelfPPO

sys.setrecursionlimit(1000000)
import csv
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLInvalidNumberOfArgumentsError
from stable_baselines3.common.base_class import maybe_make_env, BaseAlgorithm
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.type_aliases import MaybeCallback
import torch as th
from torch.nn import functional as F
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
from PRG_SB3 import *
from pyRDDLGym import RDDLEnv
from MyPPO import MyPPO
import time
from multiprocess import Process, Manager, Event, Queue, Pipe
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
from MyEnv import MyEnv

class MyProcess(Process): #继承Process类
    def __init__(self, domain, eval_domain, instance, dfa_state, policy_queue, policy_result_queues,
                 reset_states, events, total_timesteps, classify_num, term_num, log_path, model_class=MyPPO, action_noise=None):
        super(MyProcess,self).__init__()
        if dfa_state == 0:
            self.env = FlattenAction(RDDLEnv.RDDLEnv(domain=domain, instance=instance))
            eval_env = FlattenAction(RDDLEnv.RDDLEnv(domain=eval_domain, instance=instance))
            self.eval_env = BaseAlgorithm._wrap_env(maybe_make_env(eval_env, 1), 1, True)
            self.eval_interval = 3
            self.eval_time = 1
        else:
            #self.env = FlattenAction(RDDLEnv.RDDLEnv(domain=domain, instance=instance))
            self.env = FlattenAction(MyEnv(domain, instance, dfa_state, reset_states, events[dfa_state]))
        # policy_kwargs = dict(net_arch=[256, 256])
        self.model = model_class("MultiInputPolicy", self.env, device='cpu', classify_num=classify_num, term_num=term_num)
        self.thread_terminal = False
        self.policy_queue = policy_queue
        self.policy_result_queues = policy_result_queues
        self.total_timesteps = total_timesteps
        self.reset_states = reset_states
        self.events = events
        self.update_nets_interval = 100
        self.dfa_state = dfa_state
        self.time_start = time.time()
        self.log_path = log_path

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.model.policy.set_training_mode(True)
        # Update optimizer learning rate
        self.model._update_learning_rate(self.model.policy.optimizer)
        # Compute current clip range
        clip_range = self.model.clip_range(self.model._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.model.clip_range_vf is not None:
            clip_range_vf = self.model.clip_range_vf(self.model._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.model.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.model.rollout_buffer.get(self.model.batch_size):
                actions = rollout_data.actions
                if isinstance(self.model.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.model.use_sde:
                    self.model.policy.reset_noise(self.model.batch_size)

                values, log_prob, entropy = self.model.policy.evaluate_actions(rollout_data.observations, actions)
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

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
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
                self.model.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.model.policy.parameters(), self.model.max_grad_norm)
                self.model.policy.optimizer.step()

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
    )-> SelfPPO:
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
            continue_training = self.model.collect_rollouts(self.model.env, self.model.rollout_buffer, self.model.n_steps, reset_states=self.reset_states,
                                                      events=self.events, dfa_state=self.dfa_state)

            if continue_training is False:
                break

            self.model._update_current_progress_remaining(self.model.num_timesteps, total_timesteps)
            self.train()
            train_time += 1

            for i in range(self.model.classify_num):
                if i == self.dfa_state:
                    self.model.policies[self.dfa_state] = self.model.policy
                    self.policy_queue.put((self.model.policy, self.dfa_state))
                else:
                    self.policy_queue.put((None, i))
                    self.model.policies[i] = self.policy_result_queues[i].get()

            if self.dfa_state == 0 and (train_time+1) % self.eval_interval == 0:
                p = EvaluateProgress(self.model.num_timesteps, self.eval_env, self.eval_time, self.model.policies,
                                     self.time_start, self.model.policy, self.model.num_per_class, self.log_path)
                p.start()
        return self

    def run(self):
        self.learn(self.total_timesteps)



class EvaluateProgress(Process):
    def __init__(self, timestep, eval_env, eval_time, policies, time, policy, num_per_class, log_path):
        super().__init__()
        self.timestep = timestep
        self.eval_env = eval_env
        self.eval_time = eval_time
        self.policies = policies
        self.time_start = time
        self.policy = policy
        self.num_per_class = num_per_class
        self.log_path = log_path

    def run(self):
        mean_reward, mean_length = 0, 0
        for times in range(self.eval_time):
            state = self.eval_env.reset()
            done = False
            while not done:
                dfa_state = int(state['as'].item() / self.num_per_class)
                action, _ = self.policies[dfa_state].predict(state, deterministic=True)
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
            [self.timestep+1, hour, mean_reward / self.eval_time, mean_length / self.eval_time],
        ]
        with open(self.log_path, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(data)
        print('time:', hours, "h", minutes, "min", second, 's')
        print('n_step:', self.timestep + 1, 'mean_reward:', mean_reward / self.eval_time, 'mean_length:',
              mean_length / self.eval_time)


class UpdateProgress(Process):
    def __init__(self, net, classify_num, queue, result_queues):
        super().__init__()
        self.nets = [copy.deepcopy(net) for _ in range(classify_num)]
        self.queue = queue
        self.result_queues = result_queues

    def run(self):
        while True:
            data, dfa_state = self.queue.get()

            if data is None:
                self.result_queues[dfa_state].put(self.nets[dfa_state])
            else:
                self.nets[dfa_state] = data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-log', type=str, default='task2_1.csv', help='log path')
    parser.add_argument('-i', type=str, default='inst21', help='inst name')
    parser.add_argument('-r', type=str, default='random_1', help='inst name')
    parser.add_argument('-n', type=int, default=800000, help='timestep')
    args = parser.parse_args()
    domain = 'benchmarks/waterworld/' + args.r + '.rddl'
    eval_domain = domain[:-5] + '_eval.rddl'
    instance = 'benchmarks/waterworld/' + args.i + '.rddl'
    log_path = 'log/' + args.log
    data = [
        ['n_steps', 'time', 'mean_reward', 'mean_length'],
    ]
    directory = os.path.dirname(log_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    with open(log_path, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(data)
    classify_num = 6
    term_num = 2
    env = FlattenAction(RDDLEnv.RDDLEnv(domain=domain,instance=instance))
    model = MyPPO("MultiInputPolicy", env, device='cpu')
    manager = Manager()
    process_list = []
    policy_queue = Queue()
    policy_result_queues = [Queue() for _ in range(classify_num)]
    policy_update = UpdateProgress(model.policy, classify_num, policy_queue, policy_result_queues)
    policy_update.start()

    reset_states = [manager.list() for _ in range(classify_num)]
    events = [Event() for _ in range(classify_num)]
    del model
    for i in range(classify_num):
        p = MyProcess(domain, eval_domain, instance, i, policy_queue, policy_result_queues, reset_states, events, args.n, classify_num, term_num, log_path)
        process_list.append(p)
        p.start()

    process_list[0].join()
    for p in process_list:
        p.terminate()
    policy_update.terminate()
    print('finish.')
