import torch as th
from .encdec_policy import EncDecQfPolicy
from .encdec_sac_policy import EncDecPolicy
import torch.nn.functional as F
import rlkit.torch.pytorch_util as ptu
import numpy as np


class CalibrationDQNPolicy(EncDecQfPolicy):
    def __init__(self, *args, env, prev_vae=None, sample=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.env = env
        self.prev_vae = prev_vae
        self.sample = sample
        self.target = None

    def get_action(self, obs):
        with th.no_grad():
            raw_obs = obs['raw_obs']
            goal_set = obs.get('goal_set')

            # only use initial switch positions, keep constant for entire traj
            if self.target is None:
                self.target = self.env.base_env.target_pos[self.env.base_env.target_index]

            features = [self.target]

            if self.prev_vae is not None:
                if self.incl_state:
                    features.append(raw_obs)
                    if goal_set is not None:
                        features.append(goal_set.ravel())
                pred_features = self.prev_vae.sample(th.Tensor(np.concatenate(features)).to(ptu.device)).detach()
            else:
                pred_features = self.target

            if th.is_tensor(pred_features):
                obs['latents'] = pred_features.cpu().numpy()
            else:
                obs['latents'] = pred_features

            qf_input = [raw_obs, pred_features]
            if goal_set is not None:
                qf_input.insert(1, goal_set.ravel())

            q_values, ainfo = self.qf.get_action(*qf_input)
            q_values = ptu.tensor(q_values)
            action = F.one_hot(q_values.argmax().long(), 6).flatten()

        return ptu.get_numpy(action), ainfo

    def reset(self):
        self.target = None


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
            goal_set = obs.get('goal_set')

            # only use initial switch positions, keep constant for entire traj
            if self.target is None:
                self.target = self.env.base_env.target_pos[self.env.base_env.target_index]

            features = [self.target]

            if self.prev_vae is not None:
                if self.incl_state:
                    features.append(raw_obs)
                    if goal_set is not None:
                        features.append(goal_set.ravel())
                pred_features = self.prev_vae.sample(th.Tensor(np.concatenate(features)).to(ptu.device)).detach()
            else:
                pred_features = self.target

            if th.is_tensor(pred_features):
                obs['latents'] = pred_features.cpu().numpy()
            else:
                obs['latents'] = pred_features

            policy_input = [raw_obs, pred_features]
            if goal_set is not None:
                policy_input.insert(1, goal_set.ravel())

            return self.policy.get_action(*policy_input)

    def reset(self):
        self.target = None
