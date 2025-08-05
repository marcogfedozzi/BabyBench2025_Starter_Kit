import numpy as np
import os
import gymnasium as gym
import time
import argparse
import mujoco
import yaml

import mimoEnv
from mimoEnv.envs.mimo_env import MIMoEnv
import mimoEnv.utils as env_utils
import babybench.utils as bb_utils
import babybench.eval as bb_eval

import torch
from tensordict import TensorDict
from tensordict.nn import InteractionType

from torchrl.envs.libs.gym import GymEnv, GymWrapper, set_gym_backend
from torchrl.envs import (
    Compose,
    SelectTransform,
    NoopResetEnv,
    ObservationNorm,
    RewardSum,
    StepCounter,
    ToTensorImage,
    TransformedEnv,
    ParallelEnv
)
from torchrl.envs.utils import step_mdp
from torchrl.envs.utils import ExplorationType, set_exploration_type


from torchrl.modules import Actor, ActorCriticOperator, ProbabilisticActor, ValueOperator
from tensordict.nn import (TensorDictModule, TensorDictSequential, 
                        ProbabilisticTensorDictModule, 
                        ProbabilisticTensorDictSequential)
from torchrl.objectives import SACLoss, SoftUpdate, ValueEstimators

import babybench.utils as bb_utils

from torchrl.modules import ConvNet, MLP
from torch.nn import Linear
from torchrl.modules import NormalParamExtractor, TanhNormal
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.record.loggers import generate_exp_name, get_logger

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='examples/config_test_installation.yml', type=str,
                        help='The configuration file to set up environment variables')
    parser.add_argument('--render', default=True,  type=bool,
                        help='Renders a video for each episode during the evaluation.')
    parser.add_argument('--duration', default=1000, type=int,
                        help='Total timesteps per evaluation episode')
    parser.add_argument('--episodes', default=10, type=int,
                        help='Number of evaluation episode')
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)

    env = bb_utils.make_env(config, training=False)
    torchrl_env = GymWrapper(env)
    env.reset()

    # Initialize evaluation object
    evaluation = bb_eval.EVALS[config['behavior']](
        env=env,
        duration=args.duration,
        render=args.render,
        save_dir=config['save_dir'],
    )

    # Preview evaluation of training log
    evaluation.eval_logs()

    ###

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hidden_size = 1024

    backbone = TensorDictModule(
         MLP(
            in_features=torchrl_env.observation_spec["touch"].shape[-1],
            out_features=hidden_size,
            num_cells=[64, 64],
            activation_class=torch.nn.ReLU,
            device=device
        ),
        in_keys=["touch"], out_keys=["hidden"]
    )

    actor = ProbabilisticActor(
        module=TensorDictModule(
            torch.nn.Sequential(
                MLP(
                    in_features=hidden_size,
                    out_features=torchrl_env.action_spec.shape[-1] * 2,  # mean and std
                    num_cells=[64, 128],
                    activation_class=torch.nn.ReLU,
                    device=device
                ),
                NormalParamExtractor() # specify that the output has to be split; by default the "scale" is mapped onto positive range with a softplus
            ),
            in_keys=["hidden"], out_keys=["loc", "scale"] #
        ),
        spec=torchrl_env.action_spec,
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=TanhNormal, # we use a scaled Tanh to map into the correct action values
        distribution_kwargs={
            "low": torchrl_env.action_spec.space.low,
            "high": torchrl_env.action_spec.space.high,
            "tanh_loc": False
        },
        default_interaction_type=InteractionType.RANDOM,
        return_log_prob=False
    )
    

    qvalue = ValueOperator(
        MLP(
            out_features=1,
            num_cells=[64, 32],
            activation_class=torch.nn.ReLU,
            device=device
        ),
        in_keys=["hidden", "action"] # it's a Q network, so the input is the (state, action) pair
    )

    # Notice that we could simply use a TensorDictSequential, as the data flow
    # is dictated by the in and out keys of each module.
    ac_module = ActorCriticOperator(
        backbone,
        actor,
        qvalue,
    ).to(device)

    ac_module.load_state_dict(torch.load(config['model_path'], map_location=device))

    policy_module = ac_module.get_policy_operator()
    qvalue_module = ac_module.get_critic_operator()

    print("Loaded policy and Q-value modules from", config['model_path'])


    ###
    with torch.no_grad():
        for ep_idx in range(args.episodes):
            print(f'Running evaluation episode {ep_idx+1}/{args.episodes}')

            # Reset environment and evaluation
            obs, _ = env.reset()
            evaluation.reset()

            for t_idx in range(args.duration):

                # Select action
                #action = env.action_space.sample()

                # ---------------------------------------------------# 
                #                                                    #
                # TODO REPLACE WITH CALL TO YOUR TRAINED POLICY HERE #
                # action = policy(obs)                               #
                #                                                    #
                # ---------------------------------------------------#

                for k, v in obs.items():
                    if isinstance(v, np.ndarray):
                        try:
                            obs[k] = torch.tensor(v, device=device, dtype=torch.float32)
                        except ValueError:
                            obs[k] = torch.tensor(v.copy(), device=device, dtype=torch.float32)
                obs = TensorDict(obs, device=device)

                action = policy_module(obs)["action"].cpu().numpy()

                # Perform step in simulation
                obs, _, _, _, info = env.step(action)

                # Perform evaluations of step
                evaluation.eval_step(info)
                
            evaluation.end(episode=ep_idx)

if __name__ == '__main__':
    main()
