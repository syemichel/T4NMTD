import random
import time
from copy import copy

from stable_baselines3 import SAC
import gymnasium as gym


import numpy as np
from DFATransformer8 import DFATransformer
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box, Dict, Discrete
from stable_baselines3 import SAC

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}


class HalfCheetahEnv_wo_dfa(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20,
    }

    def __init__(
        self,
        forward_reward_weight=1.0,
        ctrl_cost_weight=0.1,
        reset_noise_scale=0,
        exclude_current_positions_from_observation=False,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            forward_reward_weight,
            ctrl_cost_weight,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs,
        )
        self.current_step = 0
        self.points = {'a': -8, 'b': -4, 'c': 0, 'd': 4, 'e': 8}
        self.horizon = 1000
        # initialize dfa
        self.dfa = DFATransformer()
        self.automata_states_num = int(self.dfa.accepting_state) + 1

        self._forward_reward_weight = forward_reward_weight

        self._ctrl_cost_weight = ctrl_cost_weight

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(18,), dtype=np.float64
        )

        MujocoEnv.__init__(
            self,
            "half_cheetah.xml",
            5,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )



    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = -(x_position_after - x_position_before) / self.dt
        ctrl_cost = self.control_cost(action)
        forward_reward = self._forward_reward_weight * x_velocity
        observation = self._get_obs()

        # get props
        props = {'a': False, 'b': False, 'c': False, 'd': False, 'e': False}
        for k, v in self.points.items():
            if abs(v - x_position_after) < 0.3:
                props[k] = True
                break
        last_dfa_state = self.dfa.dfa_state
        terminated, if_success, if_failure, _ = self.dfa.step(props)

        # reward shaping
        reshaped_reward = 0
        if not terminated:
            reshaped_reward = 100 / (self.automata_states_num - int(self.dfa.dfa_state)) - 100 / (
                    self.automata_states_num - int(last_dfa_state))
        elif if_failure:
            reshaped_reward = 100 / (self.automata_states_num + 1) - 100 / (
                    self.automata_states_num - int(last_dfa_state))
        elif if_success:
            reshaped_reward = 100 + 100 / self.automata_states_num - 100 / (self.automata_states_num - int(last_dfa_state))
        reward = reshaped_reward

        info = {
            "x_position": x_position_after,
            "x_velocity": x_velocity,
            "reward_run": forward_reward,
            "reward_ctrl": -ctrl_cost,
            'is_success': if_success
        }

        if self.render_mode == "human":
            self.render()

        self.current_step += 1
        truncated = False
        if self.current_step == self.horizon:
            self.current_step = 0
            truncated = True

        info['props'] = props

        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        # calculate distance from each subgoal
        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        # DFA setting
        self.dfa.reset()
        return observation

class HalfCheetahEnv(gym.ObservationWrapper):
    def __init__(
            self,
            forward_reward_weight=1.0,
            ctrl_cost_weight=0.1,
            reset_noise_scale=0.1,
            exclude_current_positions_from_observation=False,
            **kwargs,
    ):

        env = HalfCheetahEnv_wo_dfa(forward_reward_weight=forward_reward_weight,
                                    ctrl_cost_weight=ctrl_cost_weight,
                                    reset_noise_scale=reset_noise_scale,
                                    exclude_current_positions_from_observation=exclude_current_positions_from_observation, **kwargs)
        super().__init__(env)
        self.observation_space_original = Box(
            low=-np.inf, high=np.inf, shape=(18,), dtype=np.float64
        )
        self.observation_space = Dict({"obs": self.observation_space_original, "as": Box(
            low=0, high=env.get_wrapper_attr('automata_states_num')-1, shape=(1,), dtype=np.int64
        )})

    def observation(self, obs):
        return {"obs": obs, "as": int(self.env.get_wrapper_attr('dfa').dfa_state)}



'''# 创建一个 MuJoCo 环境
# env = gym.make('HalfCheetah-v4', render_mode='human')  # 你可以选择其他的 MuJoCo 环境，如 'Ant-v2', 'HalfCheetah-v2' 等
env = ObservationWithDFA(HalfCheetahEnv(render_mode='human'))

model = SAC("MultiInputPolicy", env, verbose=1, batch_size=256)
model.learn(total_timesteps=1000000, log_interval=10)
# 重置环境以开始
state = env.reset()'''
'''
if __name__ == '__main__':
    text_path = 'dfa_text/task1.txt'
    with open(text_path, 'r', encoding='utf-8') as file:
        text = file.read()
    env = HalfCheetahEnv(dfa_text=text, accepting_state='5', render_mode='human')
    model = SAC("MultiInputPolicy", env, verbose=1, learning_starts=1000,
                                learning_rate=3e-4, batch_size=256, train_freq=1,
                                buffer_size=100000,  device='cpu')
    model.learn(total_timesteps=1000000, log_interval=10)'''