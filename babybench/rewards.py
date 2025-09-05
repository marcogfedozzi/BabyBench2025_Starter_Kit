import torch
import torchrl
from tensordict.nn import InteractionType, TensorDictModule
import numpy as np
import gymnasium as gym
from torchrl.modules import MLP
import torch.nn.functional as F

import os
from tensordict import TensorDict


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


class PredictorTouchReward(gym.Wrapper):
    """
    Gym wrapper that loads a trained predictor (ForwardInverseSurprisePredictor)
    from a run directory and uses it to compute an intrinsic reward signal.

    Usage:
        env = PredictorTouchReward(env, run_dir='models/run_4wr9eh6t', device='cpu')

    The wrapper keeps the previous observation (from reset or previous step)
    and on each step constructs a small TensorDict with:
      - keys in predictor._in_keys for the previous obs
      - ('next', key) for the next obs returned by the environment
      - 'action' and ('next','reward') (the extrinsic reward)

    It then calls predictor(td, loss_td) and extracts the intrinsic component
    added by the predictor (predictor overwrites ('next','reward') with
    extrinsic + intrinsic). The intrinsic reward is returned to the environment
    caller (total_reward = extrinsic + intrinsic).
    """

    def __init__(self, env: gym.Env, predictor: TensorDictModule, run_dir: str = "models/run_final", predictor_path: str = None, device: str = None):
        """
        PredictorTouchReward requires an externally-created `predictor` instance
        (for example produced by `rlu.make_predictor(cfg)`). The wrapper will
        optionally load a state dict from `predictor_path` into that predictor
        if a path is provided.
        """
        super().__init__(env)

        if predictor is None:
            raise ValueError("PredictorTouchReward requires a predictor instance. Create it with rlu.make_predictor(cfg) and pass it to this wrapper.")

        self._td_type = TensorDict
        self._run_dir = run_dir
        self._device = torch.device(device) if device is not None else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        # resolve predictor path
        if predictor_path is None:
            predictor_path = os.path.join(run_dir, "predictor_module.pth")
        self._predictor_path = predictor_path

        # use provided predictor instance
        self.predictor = predictor

        # attempt to load weights into the provided predictor if a state dict exists
        try:
            if os.path.exists(self._predictor_path):
                sd = torch.load(self._predictor_path, map_location=str(self._device))
                try:
                    self.predictor.load_state_dict(sd)
                except Exception:
                    if isinstance(sd, dict) and "state_dict" in sd:
                        self.predictor.load_state_dict(sd["state_dict"])
        except Exception:
            # ignore load errors and continue with provided predictor as-is
            pass

        # move predictor to device and set eval()
        try:
            self.predictor.to(self._device)
            self.predictor.eval()
        except Exception:
            pass

        # storage for previous observation (to compute transition)
        self._prev_obs = None

    def reset(self, **kwargs):
        resp = self.env.reset(**kwargs)
        # gymnasium may return (obs, info)
        if isinstance(resp, tuple) and len(resp) == 2:
            obs, info = resp
        else:
            obs = resp
            info = {}

        self._prev_obs = obs
        return resp

    def step(self, action):
        # perform environment step
        obs_next, extrinsic_reward, terminated, truncated, info = self.env.step(action)

        # default: no intrinsic reward
        intrinsic_value = 0.0

        if self.predictor is not None and self._prev_obs is not None:
            try:
                from tensordict import TensorDict

                td = TensorDict({}, batch_size=())
                # populate in_keys and next keys
                in_keys = getattr(self.predictor, "_in_keys", ["touch"]) or ["touch"]
                for k in in_keys:
                    # previous obs value
                    prev_val = None
                    if isinstance(self._prev_obs, dict):
                        prev_val = self._prev_obs.get(k)
                    else:
                        try:
                            # tensordict-like
                            prev_val = self._prev_obs.get(k)
                        except Exception:
                            prev_val = None

                    if prev_val is None:
                        continue

                    t_prev = torch.as_tensor(prev_val, dtype=torch.float32, device=self._device)
                    t_next = torch.as_tensor(obs_next[k], dtype=torch.float32, device=self._device)
                    td.set(k, t_prev)
                    td.set(("next", k), t_next)

                # action: ensure tensor
                t_act = torch.as_tensor(action, dtype=torch.float32, device=self._device)
                td.set("action", t_act)

                # set next reward (extrinsic) as a 1x1 tensor so shapes match predictor
                r_t = torch.as_tensor([[extrinsic_reward]], dtype=torch.float32, device=self._device)
                td.set(("next", "reward"), r_t)

                loss_td = TensorDict({}, batch_size=())

                # call predictor in inference mode
                with torch.no_grad():
                    td_out, _ = self.predictor(td, loss_td)

                # predictor overwrites ('next','reward') with extrinsic+intrinsic
                pred_reward = td_out.get(("next", "reward"))
                # compute intrinsic as difference
                # pred_reward may be tensor shape (1,1) or (1,)
                if isinstance(pred_reward, torch.Tensor):
                    intrinsic_tensor = (pred_reward - r_t).detach().cpu()
                    intrinsic_value = float(intrinsic_tensor.flatten().mean().item())
            except Exception:
                intrinsic_value = 0.0

        total_reward = extrinsic_reward + intrinsic_value

        # update prev obs
        self._prev_obs = obs_next

        # info may be updated with predictor metrics in future
        return obs_next, total_reward, terminated, truncated, info
