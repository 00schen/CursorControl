import numpy as np
from rlkit.util.io import load_local_or_remote_file

class SimplePathLoader:
	def __init__(self,demo_path,demo_path_proportion,replay_buffers):
		self.demo_path = demo_path
		self.demo_path_proportion = demo_path_proportion
		self.replay_buffers = replay_buffers
		self.n_trans = []
		self.n_noop = []
	
	def load_demos(self):
		if type(self.demo_path) is not list:
			self.demo_path = [self.demo_path]
		for demo_path,proportion,buffer in zip(self.demo_path,self.demo_path_proportion,self.replay_buffers):
			trans = 0
			noop = 0
			data = load_local_or_remote_file(demo_path)
			print("using", len(data), "paths for training")
			for i, path in enumerate(data[:proportion]):
				infos = path['env_infos']
				trans += len(infos)
				noop += sum([x['noop'] for x in infos])
				self.load_path(path, buffer)
			self.n_trans.append(trans)
			self.n_noop.append(noop)
		noop_rates = [y / x for x, y in zip(self.n_trans, self.n_noop) if x != 0]

	def load_path(self, path, replay_buffer, obs_dict=None):
		tran_iter = zip(list(path['observations'][1:])+[path['next_observations'][-1]],
						list(path['actions']),
						list(path['rewards']),
						list(path['terminals']),
						list(path['env_infos']),
						list(path['agent_infos'])
					)

		env = replay_buffer.env
		processed_trans = []
		obs = path['observations'][0]
		obs = env.adapt_reset(obs)

		for next_obs,action,r,done,info,ainfo in tran_iter:
			info.update(ainfo)
			action = info.get(env.action_type,action)
			next_obs,r,done,info = env.adapt_step(next_obs,r,done,info)

			# if data_aug:
			# 	r, done = -1, False

			# if end_early:
			# 	if r < 0:
			# 		int_count += 1
			# 		if int_count >= end_thresh:
			# 			processed_trans = processed_trans[:-int_count + 2]
			# 			prev = processed_trans[-1]
			# 			prev = prev[0], prev[1], prev[2], -1, True, prev[5]
			# 			processed_trans[-1] = prev
			# 			break
			# 	else:
			# 		int_count = 0
			#
			# if adjust_r:
			# 	if r >= 0 and curr_r < 0:
			# 		prev = processed_trans[index]
			# 		int_done = True if self.int_dones else prev[4]
			# 		prev = prev[0], prev[1], prev[2], -1, int_done, prev[5]
			# 		processed_trans[index] = prev
			# 		curr_r = 0
			# 	elif r < 0:
			# 		if curr_r == 0:
			# 			index = len(processed_trans)
			# 		curr_r += r
			#
			# 	processed_trans.append((obs,next_obs,action,0,done,info))

				# if r >= 0:
				# 	processed_trans.append((obs, next_obs, action, r, done, info))
				# 	interv = False
				# else:
				# 	if not interv:
				# 		processed_trans.append((obs, next_obs, action, r, True, info))
				# 	interv = True

			processed_trans.append((obs,next_obs,action,r,done,info))
			obs = next_obs

		# if not end_early or int_count < end_thresh:
		# if adjust_r and curr_r < 0:
		# 	prev = processed_trans[index]
		# 	int_done = True if self.int_dones else prev[4]
		# 	prev = prev[0], prev[1], prev[2], -1, int_done, prev[5]
		# 	processed_trans[index] = prev
		# print([x[3] for x in processed_trans])
		new_path = dict(zip(
			['observations','next_observations','actions','rewards','terminals','env_infos'],
			list(zip(*processed_trans))
			))
		# print(np.mean(np.array(new_path['observations'])[:,-6:],axis=0),np.mean(new_path['rewards']))
		# print(new_path['actions'][10:15])
		path.update(new_path)
		path['observations'] = np.array(path['observations'])
		path['next_observations'] = np.array(path['next_observations'])
		path['actions'] = np.array(path['actions'])
		path['rewards'] = np.array(path['rewards'])[:,np.newaxis]
		path['terminals'] = np.array(path['terminals'])[:,np.newaxis]
		replay_buffer.add_path(path)
