import hydra
from omegaconf import OmegaConf
import torch
import torchrl.envs.transforms as T
from omegaconf import DictConfig, open_dict
from typing import List, Dict, Optional
import logging
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from typing import Type, Tuple
from functools import partial

from torchrl.collectors import DataCollectorBase
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.envs import TransformedEnv
from torchrl.objectives import LossModule, TargetNetUpdater
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.envs.libs.gym import GymEnv, GymWrapper
from torchrl.record.loggers.utils import get_logger, generate_exp_name
from typing import Any
from babybench import utils as bb_utils
from torchrl.record.loggers.common import Logger
import os


from torchrl.collectors import SyncDataCollector


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

	logging.info(f"Instantiating Transforms: {transform_cfg.keys()}")

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

				"""Unnecessary right now
				if "ObservationNorm" in cb_conf._target_ and (cb_conf.get("loc") is None or cb_conf.get("scale") is None):
					
					logging.warning("ObservationNorm with None loc and/or scale detected. Setting them to 0 and 1 for now, call 'init_transforms' to properly initialize them later")
					
					with open_dict(cb_conf):
						cb_conf.loc = 0
						cb_conf.scale = 1
				"""
				
				transforms.append(hydra.utils.instantiate(cb_conf))

	return T.Compose(*transforms)

def init_stats(env: TransformedEnv, num_iter: int = 1000):
	"""Init stats"""
	
	for trsf in env.transform:
		if isinstance(trsf, T.ObservationNorm):
			logging.info(f"Initializing transform {trsf} with {num_iter} iterations")
			trsf.init_stats(num_iter=num_iter)

def register_script_resolvers():
	"""
	Register custom resolvers for the script.
	"""

	OmegaConf.register_new_resolver("get_device", lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))
	OmegaConf.register_new_resolver("type",  lambda x: hydra.utils.get_object(x))
	OmegaConf.register_new_resolver("cls",   lambda x: hydra.utils.get_class(x))

def make_env(cfg: OmegaConf, bbench_config: Any, seed_mod: int = 0) -> GymEnv:

	# TODO: add the possibility to specify loc and scale as a list of values, 
	# and instead of init_stats assign them directly to ObsNorm after computing the
	# mean of each one (in case of multiple training env and a single eval env)

	reward_wrapper = hydra.utils.instantiate(cfg.reward, _partial_=True)

	env = reward_wrapper(bb_utils.make_env(bbench_config, training=True))

	_trsf = cfg.env.get("transform", None)
	if _trsf is not None:
		transforms = instantiate_transforms(_trsf)

		env = TransformedEnv(
                GymWrapper(env),
                transform=transforms
        )
	else:
		env = GymWrapper(env)


	_seed = cfg.get("seed", -1)
	if _seed > 0:
		env.set_seed(cfg.seed + seed_mod)

	if isinstance(env, TransformedEnv):
		init_stats(env, num_iter=cfg.env.init_stats_iter)

	register_env_resolvers(env)

	return env


def register_env_resolvers(env: TransformedEnv) -> None:
	"""
	Register custom resolvers for the environment..
	
	:param env: The environment to register resolvers for.
	"""

	OmegaConf.register_new_resolver("env_obs_shape", lambda key: env.observation_spec[key].shape[-1], replace=True)
	OmegaConf.register_new_resolver("env_act_shape", lambda k: env.action_spec.shape[-1]*k, replace=True)
	OmegaConf.register_new_resolver("env_act_space_low", lambda: env.action_spec.space.low, replace=True)
	OmegaConf.register_new_resolver("env_act_space_high", lambda: env.action_spec.space.high, replace=True)

def register_agent_resolvers(agent: TensorDictModule) -> None:
	"""
	Register custom resolvers for the policy agent.
	
	:param env: The agent to register resolvers for.
	"""

	OmegaConf.register_new_resolver("agent_policy_operator", lambda: agent.get_policy_operator(), replace=True)
	OmegaConf.register_new_resolver("agent_critic_operator", lambda: agent.get_critic_operator(), replace=True)

def make_agent(cfg: DictConfig, env: GymEnv) -> TensorDictModule:
	"""
	"""

	ac_module = hydra.utils.instantiate(cfg.module)

	register_agent_resolvers(ac_module)

	# Initialize the lazy modules
	with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
		td = env.fake_tensordict()
		ac_module(td)
	
	return ac_module

def register_loss_resolvers(module: TensorDict, loss_module: LossModule):
	OmegaConf.register_new_resolver("loss_log_alpha", lambda: [loss_module.log_alpha], replace=True)
	OmegaConf.register_new_resolver("agent_policy_parameters", lambda: module.get_policy_operator().parameters(), replace=True)
	OmegaConf.register_new_resolver("agent_critic_parameters", lambda: module.get_critic_operator().parameters(), replace=True)

def make_loss(cfg: DictConfig, module: TensorDictModule) -> Tuple[LossModule, Optional[TargetNetUpdater]]:
	"""
	Instantiates a loss function for the agent.
	
	:param module: The agent module.
	:param cfg: The configuration file.
	:return: An instance of the loss function and an optional instance of the target net updater, if required.
	"""

	loss_module: LossModule = hydra.utils.instantiate(cfg.loss.module)

	loss_module.make_value_estimator(**cfg.loss.value_estimator)

	loss_module.set_vmap_randomness("same")

	target_net_updater = cfg.loss.get("target_net_updater", None)
	if target_net_updater is not None:
		target_net_updater: TargetNetUpdater = hydra.utils.instantiate(cfg.loss.target_net_updater, loss_module=loss_module)

	register_loss_resolvers(module, loss_module)

	return loss_module, target_net_updater

def make_optimizers(cfg: DictConfig) -> Dict[str, torch.optim.Optimizer]:

	optim = {}

	for name, optimizer in cfg.optimizers.items():
		optim[name] = hydra.utils.instantiate(optimizer)

	return optim

def _to_device_transform(data: TensorDict, device):
	return data.to(device, non_blocking=True) if data.device != device else data.clone()

def _rand_init_replay_buffer(env: GymEnv, replay_buffer: ReplayBuffer, rand_steps: int):
	"""
	Insert random rollout if necessary to warm up later training.
	"""

	replay_buffer.extend(env.rollout(rand_steps))

def make_collector_rb(cfg: DictConfig, env: GymEnv, agent: TensorDictModule, bbench_config: Any = None) -> Tuple[DataCollectorBase, ReplayBuffer]:

	_tdtr = partial(_to_device_transform, device=agent.device)
	replay_buffer = hydra.utils.instantiate(cfg.replay_buffer, transform=_tdtr)

	_irf = cfg.get("init_rand_frames", 0)
	if _irf > 0:
		logging.info(f"Warming up Replay Buffer with {_irf} frames")
		
		_rand_init_replay_buffer(env, replay_buffer, _irf)
	# Check if the collector expects a list of env-generator functions
	collector_type = cfg.collector._target_.split('.')[-1]
	if "Multi" in collector_type:
		num_envs = cfg.env.num_envs
		env = []

		# If more envs are expected generate them
		assert bbench_config is not None, \
			f"Expected bbench config file to generate multiple copies of the environment, as the Collector is of type: {collector_type}"
	
		for i in range(num_envs):
			env.append(partial(make_env, cfg=cfg, bbench_config=bbench_config, seed_mod=i))

	# Random Warm Up


	collector = hydra.utils.instantiate(cfg.collector, create_env_fn=env, policy=agent, replay_buffer=replay_buffer)

	return collector, replay_buffer

def make_logger(cfg: DictConfig) -> Logger:
	"""
	Create the logger
	"""
	
	wandb_kwargs = cfg.logger.get("wandb_kwargs", {})
	if wandb_kwargs:
		kwargs = {"wandb_kwargs": wandb_kwargs}

	return get_logger(logger_type=cfg.logger.logger_type,
									 logger_name=cfg.logger.logger_name,
									 experiment_name=generate_exp_name(
										 model_name=cfg.logger.model_name,
										 experiment_name=cfg.logger.experiment_name
									 ),
									 **kwargs)

def step_optimizers(optimizers: Dict[str, torch.optim.Optimizer], losses: TensorDict) -> TensorDict:
	"""
	Backward steps through the losses for which an optimizer is specified.

	Returns a *detached* TensorDict with only the paired losses selected.
	
	Note that the key of the optimizers
	is important to determine the correct pairing with the loss tensor.
	It should follow the convention:
	optimizer_<lossname>
	where lossname is(are) the name of the item(s) generated from the loss module chosen, minus the "loss_" prefix.
	For SACLoss the losses are: loss_actor, loss_qvalue, loss_alpha -> actor, qvalue, alpha -> optimizer_actor, ...
	"""

	lossnames_l = []

	for optim_name, optim in optimizers.items():
		loss_name = optim_name.replace("optimizer", "loss")

		if not loss_name in losses:
			logging.warning(f"Loss item {loss_name} not found. Available entries are {losses.keys()}.")
			continue

		lossnames_l.append(loss_name)
			
		loss: torch.Tensor = losses[loss_name]
		optim.zero_grad()
		loss.backward()
		optim.step()
	
	return losses.select(*lossnames_l).detach()



def save_model(cfg: DictConfig, save_dir: str, logger: Logger, loss_module: LossModule, optimizers: Dict[str, torch.optim.Optimizer]):
	run_name = logger.experiment.name
	run_id  = logger.experiment.id
	dir_name = os.path.join(save_dir, "run_"+str(run_id))
	os.makedirs(dir_name, exist_ok=True)
	logging.info(f"Saving the models of the run in {dir_name}")

	#torch.save(policy_module.state_dict(), os.path.join(dir_name, "policy_module.pth"))
	#torch.save(qvalue_module.state_dict(), os.path.join(dir_name, "qvalue_module.pth"))
	torch.save(loss_module.state_dict(), os.path.join(dir_name, "loss_module.pth"))
	for optim_name, optim in optimizers.items():
			torch.save(optim.state_dict(), os.path.join(dir_name, f"{optim_name}.pth"))

	with open(os.path.join(dir_name, "config.yaml"), "w") as f:
			OmegaConf.save(cfg, f)