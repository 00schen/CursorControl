import torch as th
from .encdec_policy import EncDecPolicy
import rlkit.torch.pytorch_util as ptu
import numpy as np


class CalibrationPolicy(EncDecPolicy):
    def __init__(self, *args, env, prev_vae=None, sample=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.env = env
        self.prev_vae = prev_vae
        self.sample = sample

    def get_action(self, obs):
        with th.no_grad():
            raw_obs = obs['raw_obs']
            features = [obs['goal_obs']]

            if self.prev_vae is not None:
                if self.incl_state:
                    features.append(raw_obs)
                pred_features = self.prev_vae.sample(th.Tensor(np.concatenate(features)).to(ptu.device)).detach()
            else:
                pred_features = self.target

            if th.is_tensor(pred_features):
                obs['latents'] = pred_features.cpu().numpy()
            else:
                obs['latents'] = pred_features

            policy_input = [raw_obs, pred_features]

            return self.policy.get_action(*policy_input)
