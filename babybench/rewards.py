import torch
import torchrl
from tensordict import TensorDict
from tensordict.nn import InteractionType
import numpy as np
import gymnasium as gym


class MagnitudeTouchReward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def compute_intrinsic_reward(self, obs):
        intrinsic_reward = np.sum(obs['touch'] > 1e-6) / len(obs['touch'])
        return intrinsic_reward

    def step(self, action):
        obs, extrinsic_reward, terminated, truncated, info = self.env.step(action)
        intrinsic_reward = self.compute_intrinsic_reward(obs)
        total_reward = intrinsic_reward + extrinsic_reward # extrinsic reward is always 0  
        return obs, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    
class SurpriseTouchReward(MagnitudeTouchReward):
    ...