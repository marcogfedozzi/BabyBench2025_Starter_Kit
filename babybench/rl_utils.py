import hydra
from omegaconf import OmegaConf
import torch
import torchrl.envs.transforms as T
from omegaconf import DictConfig, open_dict
from typing import List, Dict, Optional
import logging
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from typing import Type, Tuple

from torchrl.envs import TransformedEnv
from torchrl.objectives import LossModule, TargetNetUpdater
from torchrl.envs.libs.gym import GymEnv, GymWrapper
from torchrl.record.loggers.utils import get_logger, generate_exp_name
from typing import Any
from babybench import utils as bb_utils
from torchrl.record.loggers.common import Logger


def _check_key(cfg: DictConfig, key: str):
	return key in cfg and getattr(cfg, key) is not None


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

def instantiate_transforms(transform_cfg: DictConfig) -> T.Compose:
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

def register_script_resolvers():
	"""
	Register custom resolvers for the script.
	"""

	OmegaConf.register_resolver("get_device", lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))


def make_env(cfg: OmegaConf, bbench_config: Any):

	reward_wrapper = hydra.utils.instantiate(cfg.reward, _partial_=True)

	env = reward_wrapper(bb_utils.make_env(bbench_config, training=True))

	if _check_key(cfg.env, "transform"):
		transforms = instantiate_transforms(cfg.env.transforms)

		env = TransformedEnv(
                GymWrapper(env),
                transform=transforms
        )
	else:
			env = GymWrapper(env)

	if _check_key(cfg, "seed"):
		env.set_seed(cfg.seed)

	return env


def register_env_resolvers(env: TransformedEnv) -> None:
	"""
	Register custom resolvers for the environment..
	
	:param env: The environment to register resolvers for.
	"""

	OmegaConf.register_resolver("env_obs_shape", lambda key: env.observation_spec[key].shape)
	OmegaConf.register_resolver("env_act_shape", lambda key: env.action_spec.shape)
	OmegaConf.register_resolver("env_act_space_low", lambda key: env.action_spec.space.low)
	OmegaConf.register_resolver("env_act_space_high", lambda key: env.action_spec.space.high)

def register_agent_resolvers(agent: TensorDictModule) -> None:
	"""
	Register custom resolvers for the policy agent.
	
	:param env: The agent to register resolvers for.
	"""

	OmegaConf.register_resolver("agent_policy_operator", lambda: agent.get_policy_operator())
	OmegaConf.register_resolver("agent_critic_operator", lambda: agent.get_critic_operator())

def make_agent(cfg: DictConfig, env: TransformedEnv) -> TensorDictModule:
	"""
	"""

	register_env_resolvers(env)

	ac_module = hydra.utils.instantiate(cfg.module)

	"""	
	# Initialize the lazy modules
	with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        td = env.fake_tensordict()
        td = td.to(device)
        for net in agent:
            net(td)
	"""
	
	return ac_module

def make_loss(cfg: DictConfig, module: TensorDictModule) -> Tuple[LossModule, Optional[TargetNetUpdater]]:
	"""
	Instantiates a loss function for the agent.
	
	:param module: The agent module.
	:param cfg: The configuration file.
	:return: An instance of the loss function and an optional instance of the target net updater, if required.
	"""

	register_agent_resolvers(module)
	loss_module: LossModule = hydra.utils.instantiate(cfg.loss.module)

	loss_module.make_value_estimator(cfg.loss.value_estimator)

	target_net_updater = None
	if _check_key(cfg.loss, "target_net_updater"):
		target_net_updater: TargetNetUpdater = hydra.utils.instantiate(cfg.loss.target_net_updater, loss_module=loss_module)


	return loss_module, target_net_updater

def register_loss_resolvers(module: TensorDict, loss_module: LossModule):
	OmegaConf.register_resolver("loss_log_alpha", lambda: loss_module.log_alpha)
	OmegaConf.register_resolver("agent_policy_parameters", lambda: module.get_policy_operator().parameters())
	OmegaConf.register_resolver("agent_critic_parameters", lambda: module.get_critic_operator().parameters())

def make_optimizers(cfg: DictConfig, module: TensorDict, loss_module: LossModule) -> List[torch.optim.Optimizer]:

	register_loss_resolvers(cfg, module, loss_module)

	optim = []

	for optimizer in cfg.optimizers:
		optim.append(hydra.utils.instantiate(optim))

	return optim

def make_collector_rb(cfg: DictConfig, env: GymEnv, agent: TensorDictModule):

	collector = hydra.utils.instantiate(cfg.collector, create_env_fn=env, policy=agent)

	replay_buffer = hydra.utils.instantiate(cfg.replay_buffer, 
		transform=lambda data: data.to(agent.device, non_blocking=True) if data.device != agent.device else data.clone()
	)

	return collector, replay_buffer

def make_logger(cfg: DictConfig) -> Logger:
	"""
	Create the logger
	"""
	
	if _check_key(cfg.logger, "wandb_kwargs"):
		kwargs = {"wandb_kwargs": cfg.logger.wandb_kwargs}

	return get_logger(logger_type=cfg.logger.logger_type,
									 logger_name=cfg.logger.logger_name,
									 experiment_name=generate_exp_name(
										 model_name=cfg.logger.model_name,
										 experiment_name=cfg.logger.experiment_name
									 ),
									 **kwargs)