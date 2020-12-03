from .base_oracles import UserModelOracle

class StraightLineOracle(UserModelOracle):
	def _query(self,obs,info):
		criterion = info['cos_error'] < self.threshold
		target_pos = info['target_pos']
		return criterion, target_pos
