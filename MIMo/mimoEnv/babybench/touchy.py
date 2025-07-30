""" 
Custom MIMo env for babybench with env
"""
import os
import numpy as np
import mujoco

from mimoEnv.babybench.base import BabyBenchEnv, DEFAULT_SIZE, SCENE_XML
from mimoEnv.envs.mimo_env import MIMoEnv, SCENE_DIRECTORY
from mimoActuation.actuation import SpringDamperModel
import mimoEnv.utils as env_utils
import mimoEnv.babybench.utils as bb_utils

class BabyBenchTouchEnv(BabyBenchEnv):
    pass