
import hydra
import torchrl.envs.transforms as T
from omegaconf import DictConfig, open_dict
from typing import List, Dict, Optional
import logging
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from typing import Type, Tuple

from torchrl.envs import TransformedEnv


class DuplicateFilter:
	"""
	Filters away duplicate log messages.
	Modified version of: https://stackoverflow.com/a/31953563/965332
	"""

	def __init__(self, logger):
		self.msgs = set()
		self.logger = logger

	def filter(self, record):
		msg = str(record.msg)
		is_duplicate = msg in self.msgs
		if not is_duplicate:
			self.msgs.add(msg)
		return not is_duplicate

	def __enter__(self):
		self.logger.addFilter(self)

	def __exit__(self, exc_type, exc_val, exc_tb):
		self.logger.removeFilter(self)

def instantiate_transforms(transform_cfg: DictConfig, input_td: Optional[TensorDict] = None) -> T.Compose:
	"""Instantiates transforms from config.

	:param transform_cfg: A DictConfig object containing transform configurations.
	:return: A list of instantiated transforms.
	"""
	transforms: List[T.Transform] = []

	if not transform_cfg:
		logging.warning("No transform configs found! Skipping..")
		return transforms

	if not isinstance(transform_cfg, DictConfig):
		raise TypeError("Callbacks config must be a DictConfig!")

	with DuplicateFilter(logger=logging.getLogger()):
		for _, cb_conf in transform_cfg.items():
			if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:

				logging.info(f"Instantiating transform <{cb_conf._target_}>")

				if "ObservationNorm" in cb_conf._target_ and (cb_conf.get("loc") is None or cb_conf.get("scale") is None):
					
					logging.warning("ObservationNorm with None loc and/or scale detected. Setting them to 0 and 1 for now, call 'init_transforms' to properly initialize them later")
					
					with open_dict(cb_conf):
						cb_conf.loc = 0
						cb_conf.scale = 1
				
				transforms.append(hydra.utils.instantiate(cb_conf))

	return T.Compose(*transforms)

def init_stats(env: TransformedEnv, num_iter: int = 1000):
	
	for trsf in env.transform:
		if isinstance(trsf, T.ObservationNorm):
			logging.info(f"Initializing transform {trsf} with {num_iter} iterations")
			trsf.init_stats(env, num_iter=num_iter)

def register_agent_resolvers(env: TransformedEnv) -> None:
	"""
	Register custom resolvers for the environment.
	
	:param env: The environment to register resolvers for.
	"""

	from omegaconf import OmegaConf

	OmegaConf.register_resolver("obs_shape", lambda key: env.observation_spec[key].shape)
	OmegaConf.register_resolver("act_shape", lambda key: env.action_spec.shape)
	OmegaConf.register_resolver("act_space_low", lambda key: env.action_spec.space.low)
	OmegaConf.register_resolver("act_space_high", lambda key: env.action_spec.space.high)

def make_agent(cfg: DictConfig, env: TransformedEnv) -> TensorDictModule:
	"""
	"""

	register_agent_resolvers(env)

	ac_module = hydra.utils.instantiate(cfg.module)
	
	return ac_module

def make_loss(cfg: DictConfig, module: TensorDictModule) -> Type[torch.nn.Module]:
	"""
	Instantiates a loss function for the agent.
	
	:param module: The agent module.
	:param cfg: The configuration for the loss function.
	:return: An instance of the loss function.
	"""

	if str(cfg.loss.type).upper() == "SAC":
		from torchrl.objectives import SACLoss
		loss = SACLoss(module, **cfg.loss.kwargs)