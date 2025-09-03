import torch
import torchrl
from tensordict import TensorDict
from tensordict.nn import InteractionType, TensorDictModule
import torch
import torch.nn.functional as F
from typing import List, Optional


class ForwardInverseSurprisePredictor(TensorDictModule):
    """
    TensorDictModule wrapper around feat_extractor, forward and inverse predictors.
    Expects a tensordict with observation keys in `in_keys` and an 'action' key.
    On forward it writes 'intrinsic_reward' (CPU tensor) into the tensordict and
    returns it.

    NOTE: making this a TensorDictModule registers submodules and makes .to(device)
    behave correctly. It does NOT automatically share parameters across processes —
    for MultiAsync you must either centralize training or share parameter storage
    explicitly and avoid optimizer.step() in worker processes.
    """

    def __init__(
        self,
        feat_extractor: torch.nn.Module,
        forward_model: torch.nn.Module,
        inverse_model: torch.nn.Module,
        forward_loss_fn: torch.nn.Module,
        inverse_loss_fn: torch.nn.Module,
        optim: torch.optim.Optimizer,
        in_keys: List[str] | str,
        feat_size: int,
        action_low: float = -1.0,
        action_high: float = 1.0,
        beta: float = 0.5,
        eta: float = 1.0,
        dtype = torch.float32
    ):
        super().__init__(module=torch.nn.Identity(), in_keys=in_keys, 
                         out_keys=[("next", "reward"), "intrinsic_loss_fwd", "intrinsic_loss_inv"],
                         )

        assert 0.0 <= beta <= 1.0, "beta must be in [0,1]"
        self._beta = beta
        self._eta = eta

        # register submodules by assignment
        self.feat_ext   = feat_extractor.to(dtype)
        self.fwd_mod    = forward_model.to(dtype)
        self.inv_mod    = inverse_model.to(dtype)

        # loss/optim
        self.fwd_loss_fn = forward_loss_fn
        self.inv_loss_fn = inverse_loss_fn

        self._optim = optim(
            list(self.feat_ext.parameters()) + 
            list(self.fwd_mod.parameters()) + 
            list(self.inv_mod.parameters())
        )
    

        try:
            len(in_keys)
        except:
            self._in_keys = [in_keys]
        else:
            self._in_keys = in_keys

        self._action_low = torch.as_tensor(action_low, device=self.device)
        self._action_high = torch.as_tensor(action_high, device=self.device)

    def forward(self, td: TensorDict, loss_td: TensorDict) -> TensorDict:
        # gather observation parts and move to predictor device
        obs_parts, obs_next_parts = [], []
        
        for k in self._in_keys:
            if k not in td.keys():
                raise KeyError(f"Expected key '{k}' in tensordict")
            obs_parts.append(td.get(k).to(self.device))
            obs_next_parts.append(td.get(("next", k)).to(self.device))

        obs_in = torch.cat(obs_parts, dim=-1)
        obs_next_in = torch.cat(obs_next_parts, dim=-1)

        # next features
        feats_t = self.feat_ext(obs_in)  # (B, feat_size) or (feat_size,)
        feats_t_next = self.feat_ext(obs_next_in)  # (B, feat_size) or (feat_size,)
        
        if feats_t.ndim == 1:
            feats_t = feats_t.unsqueeze(0)
        if feats_t_next.ndim == 1:
            feats_t_next = feats_t_next.unsqueeze(0)

        # inverse: (feats_t, feats_t_next) -> action_pred
        inv_in = torch.cat([feats_t, feats_t_next], dim=-1)
        action_pred = self.inv_mod(inv_in)

        # forward: (feats_t, action) -> feats_t_next_pred
        actions = td.get("action").to(self.device)
        if actions.ndim == 1:
            actions = actions.unsqueeze(0)
        fwd_in = torch.cat([feats_t, actions], dim=-1)
        feats_t_next_pred = self.fwd_mod(fwd_in)

        # map action_pred to action range (if needed)
        action_pred = torch.sigmoid(action_pred) * (self._action_high - self._action_low) + self._action_low

        # cosine: normalize
        feats_t_next_n = F.normalize(feats_t_next, p=2, dim=-1, eps=1e-8)
        feats_t_next_pred_n = F.normalize(feats_t_next_pred, p=2, dim=-1, eps=1e-8)
        cos_target = feats_t_next_n.new_ones(feats_t_next_n.size(0))

        # losses
        L_fwd = self.fwd_loss_fn(feats_t_next_n, feats_t_next_pred_n, cos_target)
        L_inv = self.inv_loss_fn(actions, action_pred)

        L_all = self._beta * L_fwd + (1.0 - self._beta) * L_inv

        loss_td.set("loss_predictor", L_all)
        
        # intrinsic reward (detach, move to CPU)
        surprise = (self._eta / 2.0) * torch.linalg.vector_norm(feats_t_next_pred - feats_t_next, dim=-1)
        surprise = surprise.unsqueeze(1)

        # update previous feature (detach)
        self.feats_t = feats_t_next.detach().squeeze().to(self.device)

        reward = td["next", "reward"]

        td.set(("next", "reward"), surprise.to(reward.device) + reward)
        # losses already set above

        return td, loss_td
    
    @property
    def optim(self):
        return {"optimizer_predictor": self._optim}