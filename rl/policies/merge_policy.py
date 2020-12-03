from CursorControl.rl.utils import RunningMeanStd
import numpy as np
from scipy.stats import norm as normal
from .follower_policy import FollowerPolicy

class ComparisonMergePolicy:
	def __init__(self, rng, base_policy):
		self.follower = FollowerPolicy()
		self.base_policy = base_policy
		self.rng = rng

	def get_action(self, obs):
		recommend = obs[:-6]
		cumulative = 1-normal.cdf(self.alpha,self.noop_rms.mean,np.sqrt(self.noop_rms.var))
		if self.rng.random() < cumulative:
			self.noop_rms.update(np.array([not np.count_nonzero(recommend)]))
			action,ainfo = self.base_policy.get_action(obs)
			return action.cpu().numpy(), {"alpha": cumulative}
		else:
			action,ainfo = self.follower.get_action(obs)
			ainfo['alpha'] = cumulative
			return action,ainfo

	def reset(self):
		self.noop_rms = RunningMeanStd()
		self.base_policy.reset()
		self.follower.reset()