import argparse
import time
from collections import defaultdict
import re
import sympy as sp
import networkx as nx
import copy
from collections import OrderedDict
import gymnasium as gym
import gymnasium.spaces as spaces
from gymnasium.spaces import *
from abc import ABCMeta, abstractmethod
import random
import numpy as np
from collections import deque

class BaseAgent(metaclass=ABCMeta):

    @abstractmethod
    def take_action(self, state):
        pass


class Agent(BaseAgent):
    def __init__(self, action_space, num_actions=1):
        self.action_space = action_space
        self.num_actions = num_actions

    def take_action(self, s, state=None):
        action = {}
        selected_actions = random.sample(list(s), self.num_actions)  # problem!!!!!!!
        for sample in selected_actions:
            if isinstance(self.action_space[sample], gym.spaces.Box):
                action[sample] = s[sample][0].item()
            elif isinstance(self.action_space[sample], gym.spaces.Discrete):
                action[sample] = s[sample]
        return action

class FlattenAction(gym.ActionWrapper):

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.action_space = spaces.flatten_space(env.action_space)

        print(self.action_space)

    def action(self, act):
        act = spaces.unflatten(self.env.action_space, act)
        agent = Agent(action_space=self.env.action_space, num_actions=self.env.numConcurrentActions)
        return agent.take_action(act)

class UpperDiscreteEnv(gym.ActionWrapper):
    def __init__(self, env: gym.Env, option_num):
        super().__init__(env)
        self.action_space = Box(-0.5, 0.4999, (1, ), dtype=np.float32)

        print(self.action_space)

    def action(self, act):
        return {'select': act}

class UpperDiscreteEnv1(gym.ActionWrapper):
    def __init__(self, env: gym.Env, option_num):
        super().__init__(env)
        self.action_space = Discrete(option_num)
        print(self.action_space)

    def action(self, act):
        return {'select': act}

class HalfCheetahObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space_original = Box(
            low=-np.inf, high=np.inf, shape=(18,), dtype=np.float64
        )
        self.observation_space = Dict({"obs": self.observation_space_original, "ds": Box(
            low=0, high=8, shape=(1,), dtype=np.int64
        )})

    def observation(self, obs):
        return {"obs": obs, "ds": int(self.env.get_wrapper_attr('dfa').dfa_state.split('q')[1]) - 1}

class AddDFAStateObs(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        image_observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.agent_view_size, self.agent_view_size, 3),
            dtype="uint8",
        )
        # 新 observation_space 是原本的 image、direction 加 dfa_state
        self.observation_space = spaces.Dict({
            'image': image_observation_space,
            'direction': Box(low=0, high=5, shape=(1, ), dtype=np.float32),
            'ds': Box(low=0, high=4, shape=(1, ), dtype=np.int16)
        })

    def observation(self, obs):
        # 移除 mission 字段（如果有）
        obs.pop('mission', None)
        return {
            'image': obs['image'],
            'direction': obs['direction'],
            'ds': obs['ds']
        }

class MinigridActionEnv(gym.ActionWrapper):
    def __init__(self, env: gym.Env, option_num=4):
        super().__init__(env)
        self.opiton_num = 4
        self.action_space = Box(0, 0.999, (1, ), dtype=np.float32)
        print(self.action_space)

    def action(self, act):
        return int(act * self.opiton_num)


class FrozenLakeActionWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.action_space = Box(0, 4, (1, ), dtype=np.int32)

        print(self.action_space)

    def action(self, act):
        return int(act.item())


class FrozenLakeObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space_original = env.observation_space
        self.observation_space = Dict({"obs": self.observation_space_original, "ds": Box(
            low=0, high=8, shape=(1,), dtype=np.int64
        )})

    def observation(self, obs):
        return {"obs": obs, "ds": self.dfa.dfa_state, "carrying": self.carrying_passenger_id, "passanger_loc1": self.passengers[1], "passanger_loc3": self.passengers[2], "passanger_loc3": self.passengers[3]}