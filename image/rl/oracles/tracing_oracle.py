from .base_oracles import UserModelOracle

class TracingOracle(UserModelOracle):
	def _query(self,obs,info):
		criterion = info['distance_to_target'] > self.threshold
		target_pos = info['target_pos']
		return criterion, target_pos
        