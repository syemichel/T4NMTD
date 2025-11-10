from __future__ import annotations

from .MyRDDLEnv import *
from env.halfcheetah.HalfCheetahEnv1 import HalfCheetahEnv1
from env.halfcheetah.HalfCheetahEnvOX1 import HalfCheetahEnvOX1
from env.halfcheetah.HalfCheetahEnv2 import HalfCheetahEnv2
from env.halfcheetah.HalfCheetahEnvOX2 import HalfCheetahEnvOX2
from env.minigird.MinigridEnv import MiniGridEnv1
from env.minigird.MinigridEnvOX import MiniGridEnvOX1
from env.frozenlake.frozen_lake1 import FrozenLakeEnv1
from env.frozenlake.frozen_lakeOX1 import FrozenLakeEnvOX1
from env.frozenlake.frozen_lake2 import FrozenLakeEnv2
from env.frozenlake.frozen_lakeOX2 import FrozenLakeEnvOX2
from env.frozenlake.frozen_lake3 import FrozenLakeEnv3
import gymnasium as gym
from typing import Any
from gymnasium.core import ObsType
from util.wrapper import *



def GetLowerEnv(name, option_index=None, reset_ps=None, render_mode=None):
    if name.startswith("waterworld") or name.startswith("racecar"):
        env = GetLowerRDDLEnv(name, option_index, reset_ps)
    elif name == "halfcheetah1" and reset_ps is None:
        env = HalfCheetahEnv1()
    elif name == "halfcheetah1" and reset_ps is not None:
        env = HalfCheetahObsWrapper(HalfCheetahEnvOX1(option_index=option_index, reset_ps=reset_ps))
    elif name == "halfcheetah2" and reset_ps is None:
        env = HalfCheetahEnv2()
    elif name == "halfcheetah2" and reset_ps is not None:
        env = HalfCheetahObsWrapper(HalfCheetahEnvOX2(option_index=option_index, reset_ps=reset_ps))
    elif name == "minigrid1" and reset_ps is None:
        env = MinigridActionEnv(AddDFAStateObs(MiniGridEnv1()))
    elif name == "minigrid1" and reset_ps is not None:
        env = MinigridActionEnv(AddDFAStateObs(MiniGridEnvOX1(option_index=option_index, reset_ps=reset_ps)))
    elif name == "frozenlake1" and reset_ps is None:
        env = FrozenLakeEnv1(desc=None, map_name="8x8", is_slippery=False, render_mode=render_mode)
    elif name == "frozenlake1" and reset_ps is not None:
        env = FrozenLakeEnvOX1(desc=None, map_name="8x8", is_slippery=False, option_index=option_index, reset_ps=reset_ps, render_mode=render_mode)
    elif name == "frozenlake2" and reset_ps is None:
        env = FrozenLakeEnv2(desc=None, map_name="8x8", is_slippery=False, render_mode=render_mode)
    elif name == "frozenlake2" and reset_ps is not None:
        env = FrozenLakeEnvOX2(desc=None, map_name="8x8", is_slippery=False, option_index=option_index, reset_ps=reset_ps, render_mode=render_mode)
    elif name == "frozenlake3" and reset_ps is None:
        env = FrozenLakeEnv3(desc=None, map_name="8x8", is_slippery=False, render_mode=render_mode)
    else:
        raise ValueError(f"Unsupported environment name: {name}")

    return env



def GetUpperEnv(name, option_num):
    if name.startswith("waterworld") or name.startswith("racecar"):
        env = GetUpperRDDLEnv(name, option_num)
    elif name.startswith("halfcheetah1"):
        env = UpperDiscreteEnv(HalfCheetahEnv1(), option_num)
    elif name.startswith("halfcheetah2"):
        env = UpperDiscreteEnv(HalfCheetahEnv2(), option_num)
    elif name.startswith("minigrid1"):
        env = UpperDiscreteEnv(AddDFAStateObs(MiniGridEnv1()), option_num)
    elif name.startswith("frozenlake1"):
        env = UpperDiscreteEnv1(FrozenLakeEnv1(desc=None, map_name="8x8", is_slippery=False), option_num)
    elif name.startswith("frozenlake2"):
        env = UpperDiscreteEnv1(FrozenLakeEnv2(desc=None, map_name="8x8", is_slippery=False), option_num)
    else:
        raise ValueError(f"Unsupported environment name: {name}")
    return env
