import torch as th
from .encdec_policy import EncDecPolicy
from .encdec_sac_policy import EncDecSACPolicy
import torch.nn.functional as F
import rlkit.torch.pytorch_util as ptu
import numpy as np


class CalibrationPolicy(EncDecPolicy):
    def __init__(self, *args, env, prev_vae=None, sample=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.env = env
        self.prev_vae = prev_vae
        self.sample = sample
        self.target = None

    def get_action(self, obs):
        with th.no_grad():
            raw_obs = obs['raw_obs']

            # only use initial switch positions, keep constant for entire traj
            if self.target is None:
                self.target = self.env.base_env.target_pos[self.env.base_env.target_index]

            features = [self.target]

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

            q_values, ainfo = self.qf.get_action(raw_obs, pred_features)
            q_values = ptu.tensor(q_values)
            action = F.one_hot(q_values.argmax().long(), 6).flatten()

        return ptu.get_numpy(action), ainfo

    def reset(self):
        self.target = None


class CalibrationSACPolicy(EncDecSACPolicy):
    def __init__(self, *args, env, prev_vae=None, sample=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.env = env
        self.prev_vae = prev_vae
        self.sample = sample
        self.target = None

    def get_action(self, obs):
        with th.no_grad():
            raw_obs = obs['raw_obs']

            # only use initial switch positions, keep constant for entire traj
            if self.target is None:
                self.target = self.env.base_env.target_pos[self.env.base_env.target_index]

            features = [self.target]

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

            return self.policy.get_action(raw_obs, pred_features)

    def reset(self):
        self.target = None
