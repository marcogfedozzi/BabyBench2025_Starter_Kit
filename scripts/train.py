import torch
from torchrl.envs.utils import ExplorationType, set_exploration_type

import time
from tqdm import tqdm
import yaml
import hydra
import logging
from babybench import rl_utils as rlu

from omegaconf import DictConfig

@hydra.main(version_base="1.3.2", config_path="./config", config_name="default") # default
def main(cfg: DictConfig):

	# Seeding and config loading
	
	torch.manual_seed(cfg.seed)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	logging.info(f"Using device: {device}")

	with open('examples/config_selftouch_simple.yml') as f:
		train_config = yaml.safe_load(f)

	# Logger

	logger = rlu.make_logger(cfg)
	logging.info("Logger created")

	train_config = rlu.update_config_savedir(train_config, logger.experiment.id)
	
	# Env

	logging.info("Making env")
	env = rlu.make_env(cfg, train_config)
	logging.info("Env created")

	# Eval Env

	eval_every = None
	eval_for = None
	if cfg.eval.get("is_on", False):

		logging.info("Making Eval env")
		eval_env = rlu.make_env(cfg, train_config, is_eval=True)
		logging.info("Eval env created")

		eval_every = cfg.eval.get("every", None)
		eval_for = cfg.eval.get("for", None)

	# Module

	agent = rlu.make_agent(cfg, env)
	logging.info("Agent created")

	# Predictor

	predictor = rlu.make_predictor(cfg)
	logging.info("Predictor created")

	# Loss

	loss_module, target_net_updater = rlu.make_loss(cfg, agent)
	logging.info("Loss created")

	# Optimizers

	optimizers, clip_grad_func = rlu.make_optimizers(cfg, loss_module)
	optimizers.update(predictor.optim)
	clip_grad_func.update(predictor.clip_grad)
	logging.info("Optimizers created")

	# Collector and ReplayBuffer
	
	collector, replay_buffer = rlu.make_collector_rb(cfg, env, agent, bbench_config=train_config)
	logging.info("Collector and Replay Buffer created")

	# Training

	pbar = tqdm(total=collector.total_frames)
	pbar_every = cfg.get("pbar_every", 1)

	collected_obs = 0
	prec_wc = 0
	
	def update_write_count(replay_buffer, prec_wc):
		collected_frames = replay_buffer.write_count - prec_wc
		prec_wc = replay_buffer.write_count
		pbar.update(collected_frames)

		return collected_frames, prec_wc

	collection_start = time.time()
	
	collector.start()

	if cfg.init_rand_frames <= 0:
		logging.info("Warming up the replay buffer")
		time.sleep(10)
	
	logging.info("--- Training starting ---")

	for train_step in range(cfg.train_steps):
		metrics_to_log = {}

		collection_time = time.time() - collection_start

		collector.update_policy_weights_() # Needed for aSync collection
		collected_frames, prec_wc = update_write_count(replay_buffer, prec_wc)

		if train_step % pbar_every == 0:
			pbar.set_description(f"Training Step: {train_step}")

		metrics_to_log["replay_buffer/collected_frames"] = collected_frames
		metrics_to_log["replay_buffer/write_count"] = replay_buffer.write_count

		collected_obs += collected_frames
		training_start_time = time.time()

		# Sample from the replay buffer
		td = replay_buffer.sample()

		rlu.log_info_keys(cfg, td, metrics_to_log)

		# Compute the loss
		loss_td = loss_module(td)
		td, loss_td = predictor(td, loss_td) # extra computation for intrinsic reward or else

		# Update Networks

		rlu.compute_grads(optimizers, loss_td, clip_grad_func)

		if train_step % cfg.get('grad_log_every', 100) == 0:
			rlu.log_model(loss_module, logger, train_step, "agent")
			rlu.log_model(predictor, logger, train_step, "predictor")

		rlu.step_optimizers(optimizers)

		if target_net_updater is not None:            
			target_net_updater.step() # Polyak update
		
		training_time = time.time() - training_start_time

		episode_end = td["next", "done"] if td["next", "done"].any() else td["next", "truncated"]

		episode_rewards = td["next", "reward"][episode_end]

		# log the norm of the action vector, averaged across the batch dim
		metrics_to_log["train/action_magnitude_mean"] = torch.linalg.vector_norm(td["action"], dim=-1).mean()
		metrics_to_log["train/action_magnitude_std"] 	= torch.linalg.vector_norm(td["action"], dim=-1).std()

		# Logging

		if len(episode_rewards) > 0:
			metrics_to_log["train/reward"] = td["next", "reward"].mean().item()
			#metrics_to_log["train/reward"] = episode_rewards.mean().item()
			if ("next", "step_count") in td and ("next", "episode_reward") in td:
				episode_length = td["next", "step_count"][episode_end]
				metrics_to_log["train/episode_reward"] = td["next", "episode_reward"].mean().item()
				metrics_to_log["train/episode_length"] = episode_length.sum().item()/len(episode_length)
		
		if collected_obs >= collector.init_random_frames:
			for k, v in loss_td.items():
				metrics_to_log[f"train/{k}"] = v.detach().item()
			metrics_to_log["train/collection_time"] = collection_time
			metrics_to_log["train/training_time"] = training_time

		# Evaluation
		# Notice that for now we're evauating using the same environment used for training,
		# not ideal, to be changed
		
		if eval_every is not None and train_step % eval_every == 0:
			logging.info(f"Evaluation @ {train_step} -- Init")
			with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
				eval_start = time.time()

				eval_rollout = eval_env.rollout(
					eval_for, agent, 
					auto_cast_to_device=True, 
					break_when_any_done=True
				)

				eval_loss_td = loss_module(eval_rollout.to(agent.device))

				eval_rollout, eval_loss_td = predictor(eval_rollout, eval_loss_td)

				eval_time = time.time() - eval_start
				eval_reward = eval_rollout["next", "reward"].sum(-2).mean().item()
				metrics_to_log["eval/reward"] = eval_reward
				metrics_to_log["eval/time"] = eval_time
				
				metrics_to_log["eval/action_magnitude_mean"] = torch.linalg.vector_norm(td["action"], dim=-1).mean()
				metrics_to_log["eval/action_magnitude_std"] = torch.linalg.vector_norm(td["action"], dim=-1).std()
				
				for k, v in eval_loss_td.items():
					metrics_to_log[f"eval/{k}"] = v.detach().item()

				logging.info(f"Evaluation @ {train_step} -- End")

				del eval_rollout, eval_loss_td

		if logger is not None:
			for metric_name, metric_value in sorted(metrics_to_log.items()):
				logger.log_scalar(metric_name, metric_value, collected_frames)
	
	logging.info("--- Training completed ---")
	end_time = time.time()
	execution_time = end_time - collection_start
	logging.info(f"Training took {execution_time:.2f} seconds to finish")
	
	# Save the model
	
	if cfg.save_models:
		rlu.save_model(
			cfg,
			cfg.save_dir,
			logger,
			agent,
			loss_module,
			optimizers,
			predictor=predictor
		)

	logging.info("Shutting down")

	try:
		collector.async_shutdown()
	except RuntimeError as e:
		logging.warning(f"Collector shutdown failed with {e}: skipping")
	# Add this back in when making eval env
	#if not eval_env.is_closed:
	#    eval_env.close()
	if not env.is_closed:
		env.close()
	
if __name__ == "__main__":
	rlu.register_script_resolvers()
	main()