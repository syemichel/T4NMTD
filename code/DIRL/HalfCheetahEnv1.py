import random
import time
from copy import copy

from ray.util.client import ray
from stable_baselines3 import SAC
import gymnasium as gym
import numpy as np
from utils import *
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box, Dict, Discrete
from stable_baselines3 import SAC

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}

dfa_text1 = '''if(ds==@q1 ^ (~p2 & ~p4)) then @q1
else if(ds==@q1 ^ (p4)) then @q3
else if(ds==@q1 ^ (p2 & ~p4)) then @q2
else if(ds==@q3 ^ (~p3 & ~p5)) then @q3
else if(ds==@q3 ^ (p5 & ~p3)) then @q2
else if(ds==@q3 ^ (p3)) then @q4
else if(ds==@q2 ^ (True)) then @q2
else if(ds==@q4 ^ (True)) then @q4'''

class DFATransformer:
    def __init__(self, dfa_text):
        self.dfa_text = dfa_text1
        self.dfa = get_dfa(self.dfa_text)
        self.dfa_state = '@q1'
        self.accepting_state = "@q" + str(self.dfa.number_of_nodes())
        self.error_state = "@q2"

    def reset(self):
        self.dfa_state = '@q1'

    def evaluate_logic_formula(self, props:Dict, formula):
        # 创建一个字典，将命题名称映射到它们的值
        '''props = {'p1': p1,'p2': p2,'p3': p3}'''

        # 替换公式中的命题名称为对应的布尔值

        for var, value in props.items():
            formula = re.sub(r'\b' + re.escape(var) + r'\b', str(value), formula)
        formula = formula.replace('~', 'not ').replace('&', 'and ').replace('|', 'or ')
        # 评估公式
        try:
            result = eval(formula)
            return result
        except Exception as e:
            print(f"评估公式时出错: {e}")
            return None

    # return terminate, if_success and if_failure
    def step(self, props):
        out_edges = self.dfa.out_edges(str(self.dfa_state), data=True)
        for edge in out_edges:
            if self.evaluate_logic_formula(props, formula=edge[2]['formula']):
                self.dfa_state = edge[1]
                break
        if self.dfa_state == self.error_state:
            return True, False, True
        if self.dfa_state == self.accepting_state:
            return True, True, False
        return False, False, False

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
        dfa_text='',
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
        self.points = {'p1': -8, 'p2': -4, 'p3': 0, 'p4': 4, 'p5': 8}
        self.horizon = 1500
        # initialize dfa
        self.dfa = DFATransformer(dfa_text)
        self.props = {'p1': False, 'p2': False, 'p3': False, 'p4': False, 'p5': False}
        self.automata_states_num = self.dfa.dfa.number_of_nodes()

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
        self.props = {'p1': False, 'p2': False, 'p3': False, 'p4': False, 'p5': False}
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = -(x_position_after - x_position_before) / self.dt
        ctrl_cost = self.control_cost(action)
        forward_reward = self._forward_reward_weight * x_velocity
        observation = self._get_obs()

        # get props
        for k, v in self.points.items():
            if abs(v - x_position_after) < 0.3:
                self.props[k] = True
                break
        last_dfa_state = int(self.dfa.dfa_state.split('q')[1])
        terminated, if_success, if_failure = self.dfa.step(self.props)
        dfa_state = int(self.dfa.dfa_state.split('q')[1])

        # reward shaping
        if not terminated:
            reward = 100 / (self.automata_states_num - dfa_state) - 100 / (
                    self.automata_states_num - last_dfa_state)
        elif if_failure:
            reward = -10
        elif if_success:
            reward = 100 + 100 / self.automata_states_num - 100 / (self.automata_states_num - last_dfa_state)

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
        self.props = {'p1': False, 'p2': False, 'p3': False, 'p4': False, 'p5': False}
        return observation

class HalfCheetahEnv(gym.ObservationWrapper):
    def __init__(
            self,
            forward_reward_weight=1.0,
            ctrl_cost_weight=0.1,
            reset_noise_scale=0.1,
            exclude_current_positions_from_observation=False,
            dfa_text="",
            **kwargs,
    ):

        env = HalfCheetahEnv_wo_dfa(forward_reward_weight=forward_reward_weight,
                                    ctrl_cost_weight=ctrl_cost_weight,
                                    reset_noise_scale=reset_noise_scale,
                                    exclude_current_positions_from_observation=exclude_current_positions_from_observation,
                                    dfa_text=dfa_text, **kwargs)
        super().__init__(env)
        self.observation_space_original = Box(
            low=-np.inf, high=np.inf, shape=(18,), dtype=np.float64
        )
        self.observation_space = Dict({"obs": self.observation_space_original, "ds": Box(
            low=0, high=env.get_wrapper_attr('automata_states_num')-1, shape=(1,), dtype=np.int64
        )})

    def observation(self, obs):
        return {"obs": obs, "ds": int(self.env.get_wrapper_attr('dfa').dfa_state.split('q')[1])-1}

if __name__ == '__main__':
    text_path = 'dfa_text/halfcheetah/halfcheetah1.txt'
    with open(text_path, 'r', encoding='utf-8') as file:
        text = file.read()
    env = HalfCheetahEnv(dfa_text=text, render_mode='human')
    model = SAC("MultiInputPolicy", env, verbose=1, learning_starts=1000,
                                learning_rate=3e-4, batch_size=256, train_freq=1,
                                buffer_size=100000,  device='cpu')
    model.learn(total_timesteps=1000000, log_interval=10)