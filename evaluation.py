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


from babybench import rl_utils as rlu
from hydra import compose, initialize
import hydra
import os
import omegaconf

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
	parser.add_argument('--run', default=None, type=str,
						help='Name of the run')
	args = parser.parse_args()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	torch.manual_seed(42)

	
	with open(args.config) as f:
		eval_config = yaml.safe_load(f)

	run_dir = os.path.join("models", "run_"+args.run)

	cfg = _get_hydra_config(run_dir)

	with omegaconf.open_dict(cfg):
		cfg.eval.seed = -1
	eval_config = rlu.update_config_savedir(eval_config, args.run)

	# Env
	env = rlu.make_env(cfg, eval_config,is_eval=True)


	# Initialize evaluation object
	evaluation = bb_eval.EVALS[eval_config['behavior']](
		env=env,
		duration=args.duration,
		render=args.render,
		save_dir="run_"+args.run,
	)

	# Preview evaluation of training log
	evaluation.eval_logs()

	###

	# Module

	agent = rlu.make_agent(cfg, env)

	agent_file = os.path.join(run_dir, "actor_module.pth")
	agent.load_state_dict(torch.load(agent_file, map_location=device))

	print("Loaded policy and Q-value modules from", agent_file)

	###
	with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():

		for ep_idx in range(args.episodes):
			print(f'Running evaluation episode {ep_idx+1}/{args.episodes}')

			# Reset environment and evaluation
			obs = env.reset()
			evaluation.reset()

			td = env.rollout(args.duration, agent)

			for t_idx in range(args.duration):
				# Note: there's really nothing useful in the info dict
				info = {
					"terminated": td["terminated"][t_idx],
					"truncated": td["truncated"][t_idx],
					"done": td["done"][t_idx]
				}

				# Perform evaluations of step
				evaluation.eval_step(info)
				
			evaluation.end(episode=ep_idx)


def _get_hydra_config(config_path):

	hydra.core.global_hydra.GlobalHydra.instance().clear() # Clear the global hydra instance to avoid conflicts
	cfg = None
	with initialize(version_base="1.3.2", config_path=config_path, job_name="config"):
		try:
			cfg = compose(config_name="config")
		except hydra.errors.MissingConfigException as e:
			print(f"No config file found for {config_path}")
			raise e
	
	return cfg

if __name__ == '__main__':
	
	rlu.register_script_resolvers()
	main()
