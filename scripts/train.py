import torch
import torchrl
from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type

import time
from tqdm import tqdm
import yaml
import hydra
import logging
from babybench import rl_utils as rlu

from omegaconf import DictConfig, OmegaConf

import tensordict

torchrl



@hydra.main(version_base="1.3.2", config_path="./config", config_name="default")
def main(cfg: DictConfig):

    # Seeding and config loading


    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    with open('examples/config_training.yml') as f:
        train_config = yaml.safe_load(f)

    # Logger

    logger = rlu.make_logger(cfg)
    logging.info("Logger created")
    
    # Env

    logging.info("Making env")
    env = rlu.make_env(cfg, train_config)
    logging.info("Env created")

    # SAC Module


    agent = rlu.make_agent(cfg, env)
    logging.info("Agent created")

    # Loss
    #
    # Already managed multiple copies of the Q net, the loss function, and all the rest

    loss_module, target_net_updater = rlu.make_loss(cfg, agent)
    logging.info("Loss created")

    # Optimizers

    optimizers = rlu.make_optimizers(cfg, agent, loss_module)
    logging.info("Optimizers created")

    # Collector and ReplayBuffer
    
    collector, replay_buffer = rlu.make_collector_rb(cfg, env, agent)
    logging.info("Collector and Replay Buffer created")

    # Training

    pbar = tqdm(total=collector.total_frames)

    collected_obs = 0

    eval_iter = 1000

    print("--- Training starting ---")

    collection_start = time.time()

    for i, td in enumerate(collector):

        collection_time = time.time() - collection_start

        collector.update_policy_weights_() # Needed for aSync collection
        collected_frames = td.numel()
        pbar.update(collected_frames)

        replay_buffer.extend(td) # put sequence in the buffer

        collected_obs += collected_frames
        training_start_time = time.time()

        if collected_obs >= collector.init_random_frames:

            losses = TensorDict({}, batch_size=[collector.frames_per_batch])
            sample_start = time.time()
            sample_time = 0

            for j in range(collector.frames_per_batch):
                # Sample from the replay buffer
                sampled_td = replay_buffer.sample()
                # print(sampled_td["next", "reward"])
                # print(sampled_td["action"])
                sample_time += time.time() - sample_start

                # Compute the loss
                loss_td = loss_module(sampled_td)

                # Update Networks

                detached_losses = rlu.step_optimizers(optimizers, loss_td)

                losses[j] = detached_losses

                if target_net_updater is not None:            
                    target_net_updater.step() # Polyak update

        traning_time = time.time() - training_start_time


        episode_end = td["next", "done"] if td["next", "done"].any() else td["next", "truncated"]

        episode_rewards = td["next", "reward"][episode_end]


        # Logging
        metrics_to_log = {}

        if len(episode_rewards) > 0:
            metrics_to_log["train/reward"] = episode_rewards.mean().item()
            if ("next", "step_count") in td and ("next", "episode_reward") in td:
                episode_length = td["next", "step_count"][episode_end]
                metrics_to_log["train/episode_reward"] = td["next", "episode_reward"].mean().item()
                metrics_to_log["train/episode_length"] = episode_length.sum().item()/len(episode_length)
        
        if collected_obs >= collector.init_random_frames:
            metrics_to_log["train/q_loss"] = losses.get("loss_qvalue").mean().item()
            metrics_to_log["train/actor_loss"] = losses.get("loss_actor").mean().item()
            metrics_to_log["train/alpha_loss"] = losses.get("loss_alpha").mean().item()
            metrics_to_log["train/alpha"] = loss_td["alpha"].item()
            metrics_to_log["train/entropy"] = loss_td["entropy"].item()
            metrics_to_log["train/collection_time"] = collection_time
            metrics_to_log["train/training_time"] = traning_time
            metrics_to_log["train/sampling_time"] = sample_time/collector.frames_per_batch

        # Evaluation
        # Notice that for now we're evauating using the same environment used for training,
        # not ideal, to be changed
        if abs(collected_frames % eval_iter) < collector.frames_per_batch:
            print("Eval")
            with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                eval_start = time.time()

                eval_rollout = env.rollout(
                    1000, agent, 
                    auto_cast_to_device=True, 
                    break_when_any_done=True
                )

                eval_time = time.time() - eval_start
                eval_reward = eval_rollout["next", "reward"].sum(-2).mean().item()
                metrics_to_log["eval/reward"] = eval_reward
                metrics_to_log["eval/time"] = eval_time

                del eval_rollout

        if logger is not None:
            for metric_name, metric_value in metrics_to_log.items():
                logger.log_scalar(metric_name, metric_value, collected_frames)

    
    collector.shutdown()
    # Add this back in when making eval env
    #if not eval_env.is_closed:
    #    eval_env.close()
    if not env.is_closed:
        env.close()
    end_time = time.time()
    execution_time = end_time - collection_start
    logging.info(f"Training took {execution_time:.2f} seconds to finish")
    
    # Save the model
    
    if cfg.save_models:
        rlu.save_model(
            cfg,
            cfg.save_dir,
            logger,
            loss_module,
            optimizers
        )

if __name__ == "__main__":
    rlu.register_script_resolvers()
    main()