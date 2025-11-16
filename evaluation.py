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
import babybench.rewards as bb_rewards

from tensordict.nn import TensorDictModule
from tensordict import TensorDict

import torch
from torchrl.envs.utils import ExplorationType, set_exploration_type

from babybench import rl_utils as rlu
from hydra import compose, initialize
import hydra
import os
import omegaconf
from functools import partial
import cv2 as cv

class AgentActionScalerModule(TensorDictModule):
	def __init__(self, agent_module: TensorDictModule, scale: float = 1.0, action_key='action'):
			# keep in/out keys consistent with wrapped module when possible
			in_keys = getattr(agent_module, "in_keys", [])
			out_keys = getattr(agent_module, "out_keys", [])
			super().__init__(agent_module, in_keys=in_keys, out_keys=out_keys)
			self.agent_module = agent_module
			self.scale = scale
			self.action_key = action_key

	def forward(self, tensordict: TensorDict) -> TensorDict:
			out = self.agent_module(tensordict)
			if self.action_key in out.keys():
					a = out.get(self.action_key)
					out.set(self.action_key, a * self.scale)
			return out

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

	#with omegaconf.open_dict(cfg):
	#	cfg.eval.seed = -1
	eval_config = rlu.update_config_savedir(eval_config, args.run)

	# Predictor

	# Env
	_throwaway_env = rlu.make_env(cfg, eval_config, is_eval=True) # dumb but quick way to set needed resolvers
	# think instead about passing the env to the predictor

	predictor = rlu.make_predictor(cfg)

	# Intrinsic Reward Wrapper
	# using the predictor reward
	pred_reward = partial(
		bb_rewards.PredictorTouchReward,
		predictor=predictor,
		run_dir="models/run_"+args.run
	)

	# Env
	env = rlu.make_env(cfg, eval_config, is_eval=True, reward_wrapper=pred_reward)
	

	# Initialize evaluation object
	evaluation = bb_eval.EVALS[eval_config['behavior']](
		env=env,
		duration=args.duration,
		render=args.render,
		save_dir="models/run_"+args.run,
	)

	# Preview evaluation of training log
	evaluation.eval_logs()

	###

	# Module

	agent = rlu.make_agent(cfg, env)

	#agent = AgentActionScalerModule(agent, scale=100)

	"""
	agent_file = os.path.join(run_dir, "actor_module.pth")

	# load saved state dict
	loaded = None
	try:
		loaded = torch.load(agent_file, map_location=device)
		agent.load_state_dict(loaded)
		print("Loaded policy and Q-value modules from", agent_file)
	except Exception as e:
		print(f"ERROR loading state dict from {agent_file}:", e)
		# re-raise so caller sees the failure
		raise
	"""

	def show_rand_imgs(eval, n_imgs, k):
		
		from random import sample
		from math import sqrt, floor, ceil

		imgs = sample(eval._images, n_imgs)
		n_cols = floor(sqrt(n_imgs))
		n_rows = ceil(n_imgs/n_cols)

		H, W, D = imgs[0].shape # H, W, D

		img_collection = np.zeros((H*n_rows, W*n_cols, D))

		for i in range(n_rows):
			for j in range(n_cols):
				img_collection[i*H:(i+1)*H, j*W:(j+1)*W] = imgs[i*n_rows+j]
		
		print(f"{n_imgs} images samples, now showing: ")
		cv.imwrite(f"img_collection_{k}.jpg", img_collection)
		print(f"Images saved 'img_collection_{k}.jpg'")


	###
	with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
	#with set_exploration_type(ExplorationType.RANDOM), torch.no_grad():

		for ep_idx in range(args.episodes):
			print(f'Running evaluation episode {ep_idx+1}/{args.episodes}')

			# Reset environment and evaluation
			#obs = env.reset()
			evaluation.reset()

			td = env.rollout(args.duration, agent, auto_cast_to_device=True)
			#print("Touch")
			#print(torch.linalg.vector_norm(td["touch"], dim=-1))
			#print(torch.max(td["touch"], dim=-1))
			#print(torch.min(td["touch"], dim=-1))
			print("Action")
			print(td["action"])
			#print(torch.linalg.vector_norm(td["action"], dim=-1))
			#print(torch.max(td["action"], dim=-1))
			#print(torch.min(td["action"], dim=-1))
			#print("Reward")
			#print(td[("next", "reward")])
			print("------------------------")

			for t_idx in range(args.duration):
				# Note: there's really nothing useful in the info dict
				info = {
					"terminated": td["terminated"][t_idx],
					"truncated": td["truncated"][t_idx],
					"done": td["done"][t_idx]
				}

				# Perform evaluations of step
				evaluation.eval_step(info)

				
			print(f"Found {len(evaluation._images)} images")
			show_rand_imgs(evaluation, 4, ep_idx)
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
