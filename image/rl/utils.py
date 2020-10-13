import numpy as np
from envs import overhead_factory,rng

from railrl.demos.source.dict_to_mdp_path_loader import DictToMDPPathLoader
class PathAdaptLoader(DictToMDPPathLoader):
	def load_path(self, path, replay_buffer, obs_dict=None):
		if path['env_infos'][0].get('adapt',True):
			replay_buffer.env.adapt_path(path)
		super().load_path(path, replay_buffer, obs_dict)

def action_adapt(self,action,info,ainfo):
	if self.action_type in ['target']:
		action = info.get('target_pred',info['target_pos'])
	elif self.action_type in ['joint']:
		action = info['joint_action']
	elif self.action_type in ['disc_traj']:
		action = np.zeros(6)
		index = ainfo['action_index']
		action[index] = 1
	return action

def adapt_factory(base,adapt_funcs):
	class PathAdapter(base):
		def step(self,action):
			obs,r,done,info = super().step(action)
			info['adapt'] = False
			return obs,r,done,info
		def adapt_path(self,path):
			tran_iter = zip(list(path['observations'][1:])+[path['next_observations'][-1]],
							list(path['actions']),
							list(path['rewards']),
							list(path['terminals']),
							list(path['env_infos']),
							list(path['agent_infos']),
							)

			processed_trans = []
			obs = self.adapt_reset(path['observations'][0])
			for next_obs,action,r,done,info,ainfo in tran_iter:
				next_obs,r,done,info = self.adapt_step(next_obs,r,done,info)
				action = self.action_adapt(action,info,ainfo)
				processed_trans.append((obs,next_obs,action,r,done,info,ainfo))
				obs = next_obs

			new_path = dict(zip(
				['observations','next_observations','actions','rewards','terminals','env_infos','agent_infos'],
				list(zip(*processed_trans)),))
			path.update(new_path)
			path['env_infos'][0]['adapt'] = False
			return path
	return PathAdapter

def adapt_factory(config):
	if config['adapt_tran']:
		config['factories'] += [adapt_factory]
	return overhead_factory(config)
