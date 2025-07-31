import torch
import torchrl
from tensordict import TensorDict
from tensordict.nn import InteractionType
import yaml

from torchrl.envs.libs.gym import GymEnv, GymWrapper, set_gym_backend
from torchrl.envs import (
    Compose,
    NoopResetEnv,
    ObservationNorm,
    StepCounter,
    ToTensorImage,
    TransformedEnv,
    ParallelEnv
)
from torchrl.envs.utils import step_mdp
from torchrl.envs.utils import ExplorationType, set_exploration_type


from torchrl.modules import Actor, ActorCriticOperator, ProbabilisticActor
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

class IMWrapper(gym.Wrapper):
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


def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    with open('examples/config_test_installation.yml') as f:
        config = yaml.safe_load(f)


    print("Making env")
    env = IMWrapper(bb_utils.make_env(config, training=True))
    # Wrap env

    logger = get_logger(
        logger_type='wandb',
        logger_name='sac_mimo_logging',
        experiment_name=generate_exp_name('SAC', 'test_touch'),
        wandb_kwargs={
            'mode': 'online',
            'project': 'torchrl_sac_mimo',
            'group': None
        }
    )


    # We need to convert the environment in a TorchRL compatible format
    # luckily, since the env is a Gym env under the hood, we can use the GymWrapper
    print("Converting to GymWrapper")
    env = TransformedEnv(
            GymWrapper(env),
            transform=ObservationNorm(in_keys=["touch"]),

    )

    env.transform.init_stats(num_iter=100, reduce_dim=0, cat_dim=0)

    env.set_seed(42)
    data = env.reset()

    print(data)

    hidden_size = 1024

    # Test to create a architecture reminiscent of a Soft Actor Critic (SAC) policy

    # MLP backbone (input: touch, output: hidden)

    # MLP actor (input: hidden, output: action mean and std) -> probabilistic actor
    # MLP value (input: hidden, action, output: value)


    backbone = TensorDictModule(
         MLP(
            in_features=env.observation_spec["touch"].shape[-1],
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
                    out_features=env.action_spec.shape[-1] * 2,  # mean and std
                    num_cells=[128, 364],
                    activation_class=torch.nn.ReLU,
                    device=device
                ),
                NormalParamExtractor()
            ),
            in_keys=["hidden"], out_keys=["loc", "scale"]
        ),
        spec=env.action_spec,
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": env.action_spec.space.low,
            "high": env.action_spec.space.high,
            "tanh_loc": False
        },
        default_interaction_type=InteractionType.RANDOM,
        return_log_prob=False
    )
    

    qvalue = TensorDictModule(
        MLP(
            out_features=1,
            num_cells=[400, 32],
            activation_class=torch.nn.ReLU,
            device=device
        ),
        in_keys=["hidden", "action"], out_keys=["value"]
    )

    ac_module = ActorCriticOperator(
        backbone,
        actor,
        qvalue,
    ).to(device)

    policy_module = ac_module.get_policy_operator()
    qvalue_module = ac_module.get_critic_operator()


    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        td = env.fake_tensordict()
        td = td.to(device)
        for net in [policy_module, qvalue_module]:
            net(td)

    # Loss

    loss_module = SACLoss(
        actor_network=policy_module,
        qvalue_network=qvalue_module,
    )

    #loss_module.make_value_estimator(ValueEstimators.GAE, gamma=0.99, lmbda=0.95)
    loss_module.make_value_estimator(gamma=0.99)

    target_net_updater = SoftUpdate(
        loss_module=loss_module,
        eps=0.995 # polyak update value
    )

    # Optimizers

    optimizer_actor     = torch.optim.Adam(params=policy_module.parameters(), lr=3e-4)
    optimizer_critic    = torch.optim.Adam(params=qvalue_module.parameters(), lr=3e-4)
    optimizer_alpha     = torch.optim.Adam(params=[loss_module.log_alpha], lr=3e-4)

    # Collector

    _fpb = 16

    collector = SyncDataCollector(
        env,
        policy_module,
        frames_per_batch=_fpb,
        total_frames=10_000,
        device=device,
        init_random_frames=500,
    )

    # TODO: check running async collector https://docs.pytorch.org/rl/main/reference/collectors.html#running-the-collector-asynchronously

    replay_buffer = ReplayBuffer(
        batch_size=16,
        storage=LazyTensorStorage(max_size=5_000),
        sampler=SamplerWithoutReplacement(),
        transform=lambda data: data.to(device, non_blocking=True) if data.device != device else data.clone(),
    )


    # Now onto training!

    # First step: reward := avg number of active touch sensors

    pbar = tqdm(total=collector.total_frames)

    collected_obs = 0

    eval_iter = 1000

    # Training

    print("--- Training starting ---")

    collection_start = time.time()

    for i, td in enumerate(collector):

        collection_time = time.time() - collection_start


        collector.update_policy_weights_()
        collected_frames = td.numel()
        pbar.update(collected_frames)

        replay_buffer.extend(td)

        collected_obs += td.numel()
        training_start_time = time.time()


        if collected_obs >= collector.init_random_frames:

            losses = TensorDict({}, batch_size=[collector.frames_per_batch])
            sample_start = time.time()
            sample_time = 0

            for j in range(collector.frames_per_batch):
                # Sample from the replay buffer
                sampled_td = replay_buffer.sample()
                sample_time += time.time() - sample_start

                print(sampled_td)


                # Compute the loss
                loss_td = loss_module(sampled_td)

                print(loss_td)

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

                losses[i] = loss_td.select("loss_actor", "loss_qvalue", "loss_alpha").detach()

                target_net_updater.step()

        traning_time = time.time() - training_start_time


        episode_end = td["next", "done"] if td["next", "done"].any() else td["next", "truncated"]

        episode_rewards = td["next", "reward"][episode_end]


        # Logging
        metrics_to_log = {}

        if len(episode_rewards) > 0:
            episode_length = td["next", "step_count"][episode_end]
            metrics_to_log["train/reward"] = episode_rewards.mean().item()
            metrics_to_log["train/episode_length"] = episode_length.sum().item()/len(episode_length)
        
        if collected_frames >= collector.init_random_frames:
            metrics_to_log["train/q_loss"] = losses.get("loss_qvalue").mean().item()
            metrics_to_log["train/actor_loss"] = losses.get("loss_actor").mean().item()
            metrics_to_log["train/alpha_loss"] = losses.get("loss_alpha").mean().item()
            metrics_to_log["train/alpha"] = loss_td["alpha"].item()
            metrics_to_log["train/entropy"] = loss_td["entropy"].item()
            metrics_to_log["train/collection_time"] = collection_time
            metrics_to_log["train/training_time"] = traning_time
            metrics_to_log["train/sampling_time"] = sample_time/collector.frames_per_batch

        

        # Evaluation
        if abs(collected_frames % eval_iter) < collector.frames_per_batch:
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



if __name__ == "__main__":
    main()