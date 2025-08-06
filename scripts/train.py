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

@hydra.main(version_base="1.3.2", config_path="../config/simulation", config_name="dataset_cnlzd")
def main():

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

    # using wandb for tracking experiment progress across multiple runs
    # especially good for later hyperparams tuning
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
    
    print("Making env")
    env = bb_rewards.MagnitudeTouchReward(bb_utils.make_env(config, training=True))
    print("Converting to GymWrapper")

    TO_TRANSFORM = False

    if TO_TRANSFORM:
        env = TransformedEnv(
                GymWrapper(env),
                transform=Compose(
                    ObservationNorm(in_keys=["touch"]),
                    SelectTransform("touch"), # only keep the "touch" modality in the observation
                    StepCounter(),
                    RewardSum()
                )

        )
        # Many "transforms" can be applied to an environment, here we're just using one
        # to normalize the touch observation. Normalization can be done by passing the desired
        # loc and scale or, as done here, by runnning some steps of the MDP and computing the
        # distribution parameters based on the observation.

        env.transform[0].init_stats(num_iter=100, reduce_dim=0, cat_dim=0)
    else:
        env = GymWrapper(env)


    env.set_seed(42)
    data = env.reset()

    print(data)

    # SAC Module

    # Why SAC? It is designed to handle continuous action spaces (like joint activation here!)
    # with high dimensional input spaces (like touch here!).
    #
    # The net is comprised of an Actor and a Critic module, taking as input an embedding of the
    # observation, obtained via compression by a common backbone
    #
    # MLP backbone (input: touch, output: hidden)
    #
    # MLP actor (input: hidden, output: action mean and std) -> probabilistic actor
    # MLP value (input: hidden, action, output: value)

    hidden_size = 1024

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


    # our actor will output a probabilistic action of dim = num of actuated joints
    # instead of the deterministic activation
    actor = ProbabilisticActor(
        module=TensorDictModule(
            torch.nn.Sequential(
                MLP(
                    in_features=hidden_size,
                    out_features=env.action_spec.shape[-1] * 2,  # mean and std
                    num_cells=[64, 128],
                    activation_class=torch.nn.ReLU,
                    device=device
                ),
                NormalParamExtractor() # specify that the output has to be split; by default the "scale" is mapped onto positive range with a softplus
            ),
            in_keys=["hidden"], out_keys=["loc", "scale"] #
        ),
        spec=env.action_spec,
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=TanhNormal, # we use a scaled Tanh to map into the correct action values
        distribution_kwargs={
            "low": env.action_spec.space.low,
            "high": env.action_spec.space.high,
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

    policy_module = ac_module.get_policy_operator()
    qvalue_module = ac_module.get_critic_operator()

    # Initialize the lazy modules
    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        td = env.fake_tensordict()
        td = td.to(device)
        for net in [policy_module, qvalue_module]:
            net(td)

    # Loss
    #
    # Already managed multiple copies of the Q net, the loss function, and all the rest

    loss_module = SACLoss(
        actor_network=policy_module,
        qvalue_network=qvalue_module,
    )

    # Can select the type of value estimator: TD0, TD1, TDgamma
    loss_module.make_value_estimator(gamma=0.99)

    # Specify that the target Q net(s) will be updated continuously with a Polyak update
    target_net_updater = SoftUpdate(
        loss_module=loss_module,
        eps=0.995 # polyak update value
    )

    # Optimizers

    optimizer_actor     = torch.optim.Adam(params=policy_module.parameters(), lr=3e-4)
    optimizer_critic    = torch.optim.Adam(params=qvalue_module.parameters(), lr=3e-4)
    optimizer_alpha     = torch.optim.Adam(params=[loss_module.log_alpha], lr=3e-4)

    # Collector
    #
    # The collector is what manages the run of the environment. You specify a maximum number of steps to run it
    # for (also infinite) and how many steps to return in a batch (sequence). It is also possible to run the
    # environment with random actions for a specific amount of steps, to collect data before training.
    #
    # The simplest collector is the Sync one, that runs one instance of the environment and of the policy
    # in sequence. More complex collectors exist, allowing one to parallelize environment execution and
    # policy training, as well as running multiple environments in parallel (https://docs.pytorch.org/rl/main/reference/collectors.html#running-the-collector-asynchronously).

    _fpb = 16

    collector = SyncDataCollector(
        env,
        policy_module,
        frames_per_batch=_fpb,
        total_frames=5_000,
        device=device,
        init_random_frames=100,
    )

    # Replay Buffer
    #
    # This is especially useful for offpolicy learning (like SAC), as it can be filled with sequences 
    # extracted from the collector, and can later be sampled for random transitions.

    replay_buffer = ReplayBuffer(
        batch_size=16,
        storage=LazyTensorStorage(max_size=5_000),
        sampler=SamplerWithoutReplacement(),
        transform=lambda data: data.to(device, non_blocking=True) if data.device != device else data.clone(),
    )


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
    main()