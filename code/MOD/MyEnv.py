import re
import time

from PRG_SB3 import *
import numpy as np
from pyRDDLGym import RDDLEnv
class MyEnv(RDDLEnv.RDDLEnv):

    def __init__(self, domain, instance, dfa_state, reset_states, event):
        super(MyEnv, self).__init__(domain, instance)
        self.last_reset = None
        self.reset_states = reset_states
        self.event = event
        self.dfa_state = dfa_state

    def reset(self):
        # print(len(self.reset_states[self.dfa_state]), 'len')
        if len(self.reset_states[self.dfa_state]) == 0:
            self.event.clear()
            self.event.wait()
            print(self.dfa_state, 'new process')
        index = random.randint(0, len(self.reset_states[self.dfa_state])-1)
        obs = self.reset_states[self.dfa_state][index]
        self.total_reward = 0
        self.currentH = 0
        self.obs_to_init_values(obs)
        obs, self.done = self.sampler.reset()
        self.state = self.sampler.states
        # print(obs)
        return obs, {}

    def obs_to_init_values(self, obs):
        last_key = ""
        for key, value in obs.items():
            init_key = re.sub("__.*", "", key)
            if init_key != last_key:
                i = 0
            init_value = self.sampler.init_values[init_key]
            if isinstance(init_value, np.ndarray):
                shape = init_value.shape
                index = np.unravel_index(i, shape)
                init_value[index] = value.item()
            else:
                value1 = value.item()
                if isinstance(self.sampler.init_values[init_key], bool):
                    value1 = bool(value1)
                self.sampler.init_values[init_key] = value1
            i += 1
            last_key = init_key

