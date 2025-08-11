import torch
import torchrl
from tensordict import TensorDict
from tensordict.nn import InteractionType
import yaml

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

import numpy as np
import gymnasium as gym

import time
from tqdm import tqdm
import hydra


from babybench import rewards as bb_rewards
from babybench import rl_utils as rlu

from omegaconf import DictConfig

@hydra.main(version_base="1.3.2", config_path="../config/simulation", config_name="dataset_cnlzd")
def main(cfg: DictConfig):

    # An intro on TensorDicts
    #
    # https://docs.pytorch.org/tensordict/stable/index.html
    #
    # TorchRL is designed to be used with TensorDicts (TD), a data type native of pytorch "tensordict" library
    # TDs functionally act like dictionaries, making it super easy to pass complex aggregated data, but with
    # (almost) the same speed and efficiency as regular Tensors (there's a slight overhead, but waaaay less than
    # it would be by using regular tensors, and saves you the headache of having to manage multiple different data types).
    # It makes it really easy to build complex networks, as you can specify which keys of the TD go as input and which come out
    # of each module. In this way most networks with multiple parallel branches can be defined as simple sequential nets,
    # as the data flow will be dictated by the in and out keys! (see example on SAC module later).

    # Seeding and config loading

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    with open('examples/config_training.yml') as f:
        config = yaml.safe_load(f)

    # Logger

    logger = rlu.get_logger(cfg)
    
    # Env

    print("Making env")
    env = rlu.make_env(cfg)

    # SAC Module

    agent = rlu.make_agent(cfg, env)

    # test it doable
    for net in agent:
        print(net)

    # Loss
    #
    # Already managed multiple copies of the Q net, the loss function, and all the rest

    loss_module, target_net_updater = rlu.make_loss(cfg, agent)

    # Optimizers

    optimizers = rlu.make_optimizers(cfg, agent, loss_module)

    # Collector and ReplayBuffer
    
    collector, replay_buffer = rlu.make_collector_rb(cfg, env, agent)

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
                sample_time += time.time() - sample_start

                # Compute the loss
                loss_td = loss_module(sampled_td)

                # TODO: change this to be policy agnostic
                # How: make the optimizers a dct with "actor", "qvalue", "alpha"
                # pass the loss_td and search for the keys "loss_"+key in optim

                loss_actor = loss_td["loss_actor"]
                loss_critic = loss_td["loss_qvalue"]
                loss_alpha = loss_td["loss_alpha"]


                # Update Actor
                optimizer_actor.zero_grad()
                loss_actor.backward()
                optimizer_actor.step()

                # Update Critic
                optimizer_critic.zero_grad()
                loss_critic.backward()
                optimizer_critic.step()

                # Update Alpha
                optimizer_alpha.zero_grad()
                loss_alpha.backward()
                optimizer_alpha.step()

                losses[j] = loss_td.select("loss_actor", "loss_qvalue", "loss_alpha").detach()

                target_net_updater.step() # Polyak update

        traning_time = time.time() - training_start_time


        episode_end = td["next", "done"] if td["next", "done"].any() else td["next", "truncated"]

        episode_rewards = td["next", "reward"][episode_end]


        # Logging
        metrics_to_log = {}

        if len(episode_rewards) > 0:
            metrics_to_log["train/reward"] = episode_rewards.mean().item()
            if TO_TRANSFORM:
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
                    1000, policy_module, 
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
    
    # Save the model

    torch.save(ac_module.state_dict(), config['model_path'])



if __name__ == "__main__":
    bbrl_utils.register_script_resolvers()
    main()