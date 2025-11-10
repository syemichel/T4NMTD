import random
import time
from copy import copy

from stable_baselines3 import SAC
import gymnasium as gym
import ray

import numpy as np
from DFATransformer7 import DFATransformer
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box, Dict, Discrete


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
        state_list,
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

        self.state_list = state_list
        self.start = False
        self.get_state_interval = 5
        self.current_reset_time = 0

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
        if not terminated:
            reward = 100 / (self.automata_states_num - int(self.dfa.dfa_state)) - 100 / (
                    self.automata_states_num - int(last_dfa_state))
        elif if_failure:
            reward = 100 / (self.automata_states_num + 1) - 100 / (self.automata_states_num - int(last_dfa_state))
        elif if_success:
            reward = 100 + 100 / self.automata_states_num - 100 / (self.automata_states_num - int(last_dfa_state))
        # self.reshaped_reward = (last_dfa_state != self.dfa.dfa_state) * 10 - 100 * if_failure
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
        self.total_reward = 0
        self.currentH = 0
        obs = None
        if len(self.state_list) > 0:
            obs = self.state_list[0]
            self.state_list.pop()
            original_obs = obs['obs']
            self.dfa.dfa_state = str(obs['as'].item())
            self.set_state(original_obs[0][0][0:9], original_obs[0][0][9:18])
            observation = self._get_obs()

        else:
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
        return observation


class HalfCheetahEnvOX(gym.ObservationWrapper):
    def __init__(
            self,
            state_list,
            forward_reward_weight=1.0,
            ctrl_cost_weight=0.1,
            reset_noise_scale=0.1,
            exclude_current_positions_from_observation=False,
            **kwargs,
    ):
        env = HalfCheetahEnv_wo_dfa(state_list, forward_reward_weight, ctrl_cost_weight, reset_noise_scale, exclude_current_positions_from_observation, **kwargs)
        super().__init__(env)
        self.observation_space_original = Box(
            low=-np.inf, high=np.inf, shape=(18,), dtype=np.float64
        )
        self.observation_space = Dict({"obs": self.observation_space_original, "as": Box(
            low=0, high=env.get_wrapper_attr('automata_states_num')-1, shape=(1,), dtype=np.int64
        )})

    def observation(self, obs):
        return {"obs": obs, "as": int(self.env.get_wrapper_attr('dfa').dfa_state)}


