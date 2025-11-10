from stable_baselines3 import A2C
from stable_baselines3.common.base_class import maybe_make_env, BaseAlgorithm
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from utils import *
import argparse
from pyRDDLGym import RDDLEnv
class LowerEnv(RDDLEnv.RDDLEnv):

    def __init__(self, domain, instance):
        super(LowerEnv, self).__init__(domain, instance)

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

