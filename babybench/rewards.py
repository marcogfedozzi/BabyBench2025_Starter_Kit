import torch
import torchrl
from tensordict.nn import InteractionType, TensorDictModule
import numpy as np
import gymnasium as gym
from torchrl.modules import MLP
import torch.nn.functional as F


class MagnitudeTouchReward(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

    def compute_intrinsic_reward(self, obs, action):
        intrinsic_reward = np.sum(obs['touch'] > 1e-6) / len(obs['touch'])
        return intrinsic_reward, {}

    def step(self, action):
        obs, extrinsic_reward, terminated, truncated, info = self.env.step(action)
        intrinsic_reward, extra_info = self.compute_intrinsic_reward(obs, action)
        total_reward = intrinsic_reward + extrinsic_reward # extrinsic reward is always 0
        info.update(extra_info)
        return obs, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    

# Deprecated
class SurpriseTouchReward(MagnitudeTouchReward):
    # TODO:
    # add three networks:
    # - feat extractor -> could be provided from the outside (e.g. the SAC one)
    # - forward model
    # - inverse model

    # Compare it to original
    # https://github.com/pathak22/noreward-rl/blob/master/src/model.py

    def __init__(self, env: gym.Env, 
                 #feat_extractor: Module, 
                 #forward_model: Module, 
                 #inverse_model: Module,
                 device: str = "cpu",
                 eta: float = 1.0,
                 beta: float = 1.0
                 ):
        self.env = env
        self.device = device

        feat_size = 256

        self.feat_ext = MLP(
            in_features=env.observation_space["touch"].shape[-1],
            out_features=feat_size,
            num_cells=[1024,512],
            activation_class=torch.nn.ReLU,
            activate_last_layer=False,
            dropout=0.0,
            device=self.device
        )
                
        self.fwd_mod = MLP(
            in_features=feat_size+env.action_space.shape[-1],
            out_features=feat_size,
            num_cells=[512,1024],
            activation_class=torch.nn.ReLU,
            activate_last_layer=False, # mapped as the feat extractor output
            dropout=0.0,
            device=self.device
        )
        
        self.inv_mod = MLP(
            in_features=feat_size+feat_size,
            out_features=env.action_space.shape[-1],
            num_cells=[512,1024],
            activation_class=torch.nn.ReLU,
            dropout=0.0,
            device=self.device
        )

        self.fwd_lossfn = torch.nn.CosineEmbeddingLoss(margin=0.0, reduction="mean")
        self.inv_lossfn = torch.nn.MSELoss(reduction="mean")

        self.fwd_optim = torch.optim.Adam(
            list(self.feat_ext.parameters()) + list(self.fwd_mod.parameters()), 
            lr=1e-3
        )
        self.inv_optim = torch.optim.Adam(
            list(self.feat_ext.parameters())+list(self.inv_mod.parameters()), 
            lr=1e-3
        )

        self._action_low    = torch.as_tensor(self.env.action_space.low).to(device)
        self._action_high   = torch.as_tensor(self.env.action_space.high).to(device)

        self._eta = eta

        self.feats_t = torch.zeros(feat_size, device=self.device) # no grad needed

    def compute_intrinsic_reward(self, obs, action):
        """
        Note: obs is the one at time t+1, post action.
        """
        with torch.enable_grad():  # override collector's no_grad
            obs_t = torch.as_tensor(obs["touch"], dtype=torch.float32, device=self.device)
            act_t = torch.as_tensor(action, dtype=torch.float32, device=self.device)

            # Feat extractor
            feats_t_next = self.feat_ext(obs_t)                # -> (..., feat_size)

            # Inverse
            inv_in = torch.cat([self.feats_t, feats_t_next], dim=-1)
            action_pred = self.inv_mod(inv_in)                 # -> (..., action_dim)

            # Forward
            fwd_in = torch.cat([self.feats_t, act_t], dim=-1)
            feats_t_next_pred = self.fwd_mod(fwd_in)           # -> (..., feat_size)
            action_pred = torch.sigmoid(action_pred) * (self._action_high - self._action_low) + self._action_low

            # Losses

            if feats_t_next.ndim == 1:
                feats_t_next = feats_t_next.unsqueeze(0)
                feats_t_next_pred = feats_t_next_pred.unsqueeze(0)

            # L2-normalize per-sample for cosine loss
            feats_t_next_n = F.normalize(feats_t_next, p=2, dim=-1, eps=1e-8)
            feats_t_next_pred_n = F.normalize(feats_t_next_pred, p=2, dim=-1, eps=1e-8)

            # cosine target: 1 means "pull together"
            cos_target = feats_t_next_n.new_ones(feats_t_next_n.size(0))

            L_fwd: torch.Tensor = self._beta * self.fwd_lossfn(feats_t_next_n, feats_t_next_pred_n, cos_target)
            L_inv: torch.Tensor = (1 - self._beta) * self.inv_lossfn(act_t, action_pred)

            # Step Losses 
            self.fwd_optim.zero_grad()
            self.inv_optim.zero_grad()
            L_fwd.backward(retain_graph=True)  # feat_ext shared
            L_inv.backward()
            self.fwd_optim.step()
            self.inv_optim.step()

            # Reward
            intrinsic_reward = self._eta / 2 * torch.linalg.vector_norm(feats_t_next_pred - feats_t_next, dim=-1).detach().cpu()

            # Update feats_t for next step (break graph)
            self.feats_t = feats_t_next.detach().squeeze()

        return intrinsic_reward, {"loss_forward": L_fwd.detach().cpu(), "loss_inverse": L_inv.detach().cpu()}
