import copy
import re
import time

import numpy as np

from util.wrapper import FlattenAction, UpperDiscreteEnv
from pyRDDLGym import RDDLEnv
from .BaseRDDLEnv import MyEnv

import json
import os

def GetLowerRDDLEnv(name, option_index=None, reset_ps=None):
    with open('../env/env_configs.json') as f:
        config = json.load(f)[name]
    domain = config['domain']
    instance = config['instance']
    if reset_ps is None:
        env = FlattenAction(RDDLEnv.RDDLEnv(domain=domain, instance=instance))
    else:
        env = FlattenAction(MyEnv(domain, instance, option_index, reset_ps))
    return env


def GetUpperRDDLEnv(name, option_num):
    with open('../env/env_configs1.json') as f:
        config = json.load(f)[name]

    domain = config['domain']
    instance = config['instance']
    return UpperDiscreteEnv(RDDLEnv.RDDLEnv(domain=domain, instance=instance), option_num=option_num)

