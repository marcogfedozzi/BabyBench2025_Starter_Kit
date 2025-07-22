import torch
import torchrl
from tensordict import TensorDict
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

from torchrl.modules import Actor
from tensordict.nn import (TensorDictModule, TensorDictSequential, 
                        ProbabilisticTensorDictModule, 
                        ProbabilisticTensorDictSequential)

import babybench.utils as bb_utils

from torchrl.modules import ConvNet, MLP
from torch.nn import Linear
from torchrl.modules import NormalParamExtractor, TanhNormal


def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    with open('examples/config_test_installation.yml') as f:
        config = yaml.safe_load(f)


    print("Making env")
    env = bb_utils.make_env(config, training=False)
    print("Converting to GymWrapper")
    env = GymWrapper(env)
    env.set_seed(42)
    data = env.reset()

    #print(env.observation_spec)
    #print(env.action_spec)

    print(data)

    """
    actor = Actor(

        MLP(
            in_features=env.observation_spec["touch"].shape[-1],
            out_features=env.action_spec.shape[-1],
            num_cells=[64, 64],
            activation_class=torch.nn.ReLU,
            device=device
        ),
        in_keys=["touch"],
    )
    """

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

    actor = ProbabilisticTensorDictSequential(
        TensorDictModule(
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
        ProbabilisticTensorDictModule(in_keys=["loc", "scale"],
                                      out_keys=["action"],
                                      distribution_class=TanhNormal,
                                      return_log_prob=False,)
    )

    actor = Actor(
        MLP(
            in_features=hidden_size,
            out_features=env.action_spec.shape[-1],
            num_cells=[128, 364],
            activation_class=torch.nn.ReLU,
            device=device
        ),
        in_keys=["hidden"],
    )

    value = TensorDictModule(
        MLP(
            out_features=1,
            num_cells=[400, 32],
            activation_class=torch.nn.ReLU,
            device=device
        ),
        in_keys=["hidden", "action"], out_keys=["value"]
    )

    print(value)


    sequence = TensorDictSequential(
        backbone,
        actor,
        value,
    ).to(device)

    MAX_EPS = 10

    data_stack = TensorDict(batch_size=[MAX_EPS])

    for epx in range(MAX_EPS):
        data = sequence(data.to(device))
        data_stack[epx] = env.step(data)

        if data["done"].any():
            print(f"Episode {epx} finished")
            break
        data = step_mdp(data)

    print("Collected data:", data_stack)

    env.reset()
    tensordict_rollout = env.rollout(policy=sequence, max_steps=MAX_EPS, auto_cast_to_device=True)
    print(tensordict_rollout)


if __name__ == "__main__":
    main()