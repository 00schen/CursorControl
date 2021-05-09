import torch as th
from .encdec_policy import EncDecPolicy
import torch.nn.functional as F
import rlkit.torch.pytorch_util as ptu
from rlkit.torch.distributions import OneHotCategorical as TorchOneHot


class CalibrationPolicy(EncDecPolicy):
    def __init__(self, *args, env, prev_encoder=None, sample=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.env = env
        self.prev_encoder = prev_encoder
        self.sample = sample

    def get_action(self, obs):
        with th.no_grad():
            raw_obs = obs['raw_obs']
            target = self.env.base_env.target_pos[self.env.base_env.target_index]
            if self.prev_encoder is not None:
                pred_features = self.prev_encoder.sample(th.Tensor(target).to(ptu.device)).detach()
                obs['latents'] = pred_features.cpu().numpy()
            else:
                pred_features = target
                obs['latents'] = pred_features

            q_values, ainfo = self.qf.get_action(raw_obs, pred_features[:3])
            q_values = ptu.tensor(q_values)
            if self.logit_scale != -1:
                action = TorchOneHot(logits=self.logit_scale * q_values).sample().flatten().detach()
            else:
                action = F.one_hot(q_values.argmax().long(), 6).flatten()

        return ptu.get_numpy(action), ainfo
