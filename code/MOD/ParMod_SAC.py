import argparse
import collections
import copy
import os
import re
import sys
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
from MySAC import MySAC
import time
from multiprocess import Process, Manager, Event, Queue, Pipe
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
from MyEnv import MyEnv

class MyProcess(Process): #继承Process类
    def __init__(self, domain, eval_domain, instance, dfa_state, actor_queue, actor_result_queues, critic_queue, critic_result_queues,
                 reset_states, events, total_timesteps, classify_num, term_num, log_path, model_class=MySAC, action_noise=None):
        super(MyProcess,self).__init__()
        if dfa_state == 0:
            self.env = FlattenAction(RDDLEnv.RDDLEnv(domain=domain, instance=instance))
            eval_env = FlattenAction(RDDLEnv.RDDLEnv(domain=eval_domain, instance=instance))
            self.eval_env = BaseAlgorithm._wrap_env(maybe_make_env(eval_env, 1), 1, True)
            self.eval_interval = 1000
            self.eval_time = 5
        else:
            self.env = FlattenAction(MyEnv(domain, instance, dfa_state, reset_states, events[dfa_state]))
        # policy_kwargs = dict(n_critics=2, n_quantiles=25)
        self.model = model_class("MultiInputPolicy", self.env, verbose=1, learning_starts=1000,
                                learning_rate=3e-4, batch_size=256, train_freq=1, action_noise=action_noise,
                                buffer_size=100000, classify_num=classify_num, term_num=term_num, device='cpu')
        self.thread_terminal = False
        self.actor_queue = actor_queue
        self.actor_result_queues = actor_result_queues
        self.critic_queue = critic_queue
        self.critic_result_queues = critic_result_queues
        self.total_timesteps = total_timesteps
        self.reset_states = reset_states
        self.events = events
        self.update_nets_interval = 100
        self.dfa_state = dfa_state
        self.time_start = time.time()
        self.log_path = log_path

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.model.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.model.actor.optimizer, self.model.critic.optimizer]
        if self.model.ent_coef_optimizer is not None:
            optimizers += [self.model.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self.model._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data, nums = self.model.replay_buffer.sample(batch_size,
                                                                env=self.model._vec_normalize_env)  # type: ignore[union-attr]
            # We need to sample because `log_std` may have changed between two gradient steps
            if self.model.use_sde:
                self.model.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.model.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.model.ent_coef_optimizer is not None and self.model.log_ent_coef is not None:
                ent_coef = th.exp(self.model.log_ent_coef.detach())
                ent_coef_loss = -(self.model.log_ent_coef * (log_prob + self.model.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.model.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None and self.model.ent_coef_optimizer is not None:
                self.model.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.model.ent_coef_optimizer.step()

            target_q_values = th.tensor([], device=self.model.device)
            index = 0
            for i in range(self.model.classify_num):
                if nums[i] == 0:
                    continue
                next_observations = {}
                for key, value in replay_data.next_observations.items():
                    next_observations[key] = value[index:index + nums[i]]
                with th.no_grad():
                    # Select action according to policy
                    next_actions, next_log_prob = self.model.actor_targets[i].action_log_prob(next_observations)
                    # Compute the next Q values: min over all critics targets
                    next_q_values = th.cat(self.model.critic_targets[i](next_observations, next_actions), dim=1)
                    next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                    # add entropy term
                    next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                    # td error + entropy term
                    target_q_values1 = replay_data.rewards[index:index + nums[i]] + (
                                1 - replay_data.dones[index:index + nums[i]]) * self.model.gamma * next_q_values
                    target_q_values = th.cat((target_q_values, target_q_values1), dim=0)
                index += nums[i]

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.model.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)  # for type checker
            critic_losses.append(critic_loss.item())  # type: ignore[union-attr]

            # Optimize the critic
            self.model.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.model.critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Min over all critic networks
            q_values_pi = th.cat(self.model.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.model.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.model.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.model.target_update_interval == 0:
                polyak_update(self.model.critic.parameters(), self.model.critic_targets[self.dfa_state].parameters(),
                              self.model.tau)
                polyak_update(self.model.actor.parameters(),
                              self.model.actor_targets[self.dfa_state].parameters(), self.model.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.model.batch_norm_stats, self.model.batch_norm_stats_target, 1.0)

        self.model._n_updates += gradient_steps

    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 100,
            tb_log_name: str = "run",
            reset_num_timesteps: bool = True,
            progress_bar: bool = False,
    ):
        total_timesteps, callback = self.model._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )
        while self.model.num_timesteps < total_timesteps:
            # print(self.model.num_timesteps)
            rollout = self.model.collect_rollouts(self.model.env, callback=callback,
                                                      train_freq=self.model.train_freq,
                                                      replay_buffer=self.model.replay_buffer,
                                                      action_noise=self.model.action_noise,
                                                      learning_starts=self.model.learning_starts,
                                                      log_interval=log_interval, reset_states=self.reset_states,
                                                      events=self.events, dfa_state=self.dfa_state)

            if self.model.num_timesteps > 0 and self.model.num_timesteps > self.model.learning_starts:
                gradient_steps = self.model.gradient_steps if self.model.gradient_steps >= 0 else rollout.episode_timesteps
                # Special case when the user passes `gradient_steps=0`
                if gradient_steps > 0:
                    self.train(batch_size=self.model.batch_size, gradient_steps=gradient_steps)

            if self.dfa_state == 0 and (self.model.num_timesteps+1) % self.eval_interval == 0:
                p = EvaluateProgress(self.model.num_timesteps, self.eval_env, self.eval_time, self.model.actor_targets, self.time_start, self.model.policy, self.model.num_per_class, self.log_path)
                p.start()

            if (self.model.num_timesteps+1) % 300 == 0:
                for i in range(self.model.classify_num):
                    if i == self.dfa_state:
                        self.actor_queue.put((self.model.actor_targets[self.dfa_state], self.dfa_state))
                        self.critic_queue.put((self.model.critic_targets[self.dfa_state], self.dfa_state))
                    else:
                        self.actor_queue.put((None, i))
                        self.model.actor_targets[i] = self.actor_result_queues[i].get()
                        self.critic_queue.put((None, i))
                        self.model.critic_targets[i] = self.critic_result_queues[i].get()



    def network_update(self, net1, net2):
        polyak_update(net1.parameters(), net2.parameters(), 1.0)
        return net2


    def run(self):
        self.learn(self.total_timesteps)


def compare_models(model1, model2):
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()

    # 检查两个模型的参数数量是否相同
    if len(state_dict1) != len(state_dict2):
        return False

    # 检查每个参数张量是否相等
    for key in state_dict1:
        if not th.allclose(state_dict1[key], state_dict2[key]):
            return False

    return True

class EvaluateProgress(Process):
    def __init__(self, timestep, eval_env, eval_time, actor_targets, time, policy, num_per_class, log_path):
        super().__init__()
        self.timestep = timestep
        self.eval_env = eval_env
        self.eval_time = eval_time
        self.actor_targets = actor_targets
        self.time_start = time
        self.policy = policy
        self.num_per_class = num_per_class
        self.log_path = log_path

    def predict(
            self,
            observation: Union[np.ndarray, Dict[str, np.ndarray]],
            state: Optional[Tuple[np.ndarray, ...]] = None,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False,
            dfa_state: int = -1,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:

        observation, vectorized_env = self.policy.obs_to_tensor(observation)

        with th.no_grad():
            actions = self.actor_targets[dfa_state](observation, deterministic)
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

    def run(self):
        mean_reward, mean_length = 0, 0
        for times in range(self.eval_time):
            state = self.eval_env.reset()
            done = False
            while not done:
                dfa_state = int(state['as'].item() / self.num_per_class)
                action, _ = self.predict(state, deterministic=True, dfa_state=dfa_state)
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
    parser.add_argument('-log', type=str, default='task1.csv', help='log path')
    parser.add_argument('-i', type=str, default='inst21', help='inst name')
    parser.add_argument('-r', type=str, default='waterworld2', help='inst name')
    parser.add_argument('-c', type=int, default=6, help='process num')
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
    classify_num = args.c
    term_num = 2
    env = FlattenAction(RDDLEnv.RDDLEnv(domain=domain,instance=instance))
    model = MySAC("MultiInputPolicy", env, verbose=1, learning_starts=1000,
                       learning_rate=3e-4, batch_size=256, train_freq=1, action_noise=None,
                       buffer_size=100000, classify_num=classify_num, term_num=term_num, device='cpu')
    manager = Manager()
    process_list = []
    actor_queue = Queue()
    actor_result_queues = [Queue() for _ in range(classify_num)]
    actor_update = UpdateProgress(model.actor, classify_num, actor_queue, actor_result_queues)
    actor_update.start()
    critic_queue = Queue()
    critic_result_queues = [Queue() for _ in range(classify_num)]
    critic_update = UpdateProgress(model.critic_target, classify_num, critic_queue, critic_result_queues)
    critic_update.start()

    reset_states = manager.list([collections.deque(maxlen=10) for _ in range(classify_num)])

    events = [Event() for _ in range(classify_num)]
    del model
    for i in range(classify_num):
        p = MyProcess(domain, eval_domain, instance, i, actor_queue, actor_result_queues, critic_queue, critic_result_queues, reset_states, events, 150000, classify_num, term_num, log_path)
        process_list.append(p)
        p.start()
    process_list[0].join()
    for p in process_list:
        p.terminate()
    actor_update.terminate()
    critic_update.terminate()
    print('finish.')
