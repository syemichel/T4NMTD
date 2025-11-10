import copy
import re
import time
import ray
import random
import numpy as np
from pyRDDLGym import RDDLEnv
class MyEnv(RDDLEnv.RDDLEnv):

    def __init__(self, domain, instance, option_index, reset_ps):
        super(MyEnv, self).__init__(domain, instance)
        self.option_index = option_index
        self.start = False
        self.reset_ps = reset_ps
        self.end_states = None
        self.buffer_action = None
        self.initial_state = None
        self.current_used_state = False
        self.value = None
        self.log_prob = None

    def reset(self):
        reset_state = ray.get(self.reset_ps.get_state.remote(self.option_index))

        while not self.start:
            if reset_state is None:
                time.sleep(5)
                reset_state = ray.get(self.reset_ps.get_state.remote(self.option_index))
            else:
                self.start = True
                print(self.option_index, "start!!!")
                break
        info = reset_state
        obs, self.end_states = info
        self.initial_state = obs
        self.total_reward = 0
        self.currentH = 0
        self.obs_to_init_values(obs)
        obs, self.done = self.sampler.reset()
        self.state = self.sampler.states
        return obs, {}

    def obs_to_init_values(self, obs):
        last_key = ""
        for key, value in obs.items():
            init_key = re.sub("__.*", "", key)
            if init_key != last_key:
                i = 0
            init_value = self.sampler.init_values[init_key]
            init_value = copy.deepcopy(init_value)
            if isinstance(init_value, np.ndarray):
                shape = init_value.shape
                index = np.unravel_index(i, shape)
                init_value[index] = value.item()
                self.sampler.init_values[init_key] = init_value
            else:
                value1 = value.item()
                if isinstance(self.sampler.init_values[init_key], bool):
                    value1 = bool(value1)
                self.sampler.init_values[init_key] = value1
            i += 1
            last_key = init_key