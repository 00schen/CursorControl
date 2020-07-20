import argparse
import ray
from tqdm import tqdm
from copy import deepcopy

from utils import *
from envs import *

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

parser = argparse.ArgumentParser(description='Sequence Modeling - Velocity Controlled 2D Simulation')
parser.add_argument('--workers', default=10, type=int, help='Number of environments to run in parallel')
parser.add_argument('--reward_type', default=4, type=int, help='1: rollout, 2: distance+action_size')
args = parser.parse_args()

@ray.remote(num_cpus=1,num_gpus=.2)
class Sampler:
	def sample(self,config,seed,count):
		env = make_vec_env(lambda: PreviousN(config))
		env.seed(seed)
		rng = np.random.default_rng(seed)

		data = []
		for i in tqdm(range(count)):
			eps_data = []

			obs = env.reset()
			obs = obs[0]

			count = 0
			done = False
			while not done:
				action = [env.envs[0].env.env.target_pos-env.envs[0].env.env.tool_pos]\
						 if rng.random() > config['pr'] else [env.action_space.sample()]
				new_obs,r,done,info = env.step(action) 
				new_obs,r,done,info = new_obs[0],r[0],done[0],info[0]
				r += info['diff_distance']
				done = count >= 200
				if info["distance_to_target"] < .025:
					r += 100
					eps_data.append((obs,new_obs,action,r,done,info))
					break
				eps_data.append((obs,new_obs,action,r,done,info))
				obs = new_obs

				count += 1

			obs,new_obs,action,r,done,info = zip(*eps_data)
			data.append([obs,new_obs,action,r,done,info])

		return data

if __name__ == "__main__":
	ray.init(temp_dir='/tmp/ray_exp')
	
	# envs = ['ScratchItch']
	# success_rates = []
	# min_distances = []
	# for env in envs:
	# 	curriculum_default_config = {'num_obs': 5, 'oracle': 'trajectory', 'coop': False,
	# 								'action_type': 'joint', 'step_limit': np.inf}
	# 	env_config = deepcopy(default_config)
	# 	env_config.update(curriculum_default_config)
	# 	env_config.update(env_map[env])
	# 	print(env)
	# 	print(env_config['env_name'])

	# 	configs = [{**deepcopy(env_config),**{'pr': pr}} for pr in [0,.25,.5,.75]]
	# 	success_rate = []
	# 	min_distance = []
	# 	for config in configs:
	# 		print(config['env_name'])
	# 		samplers = [Sampler.remote() for i in range(args.workers)]
	# 		samples = [samplers[i].sample.remote(config,1000+i,500//args.workers) for i in range(args.workers)]
	# 		samples = [ray.get(sample) for sample in samples]
	# 		# samples = [Sampler().sample(config,1000+i,100//args.workers) for i in range(args.workers)]

	# 		samples = list(zip(*sum(samples,[])))
			
	# 		success_rate.append(np.mean([np.any(np.array(sample)>50) for sample in samples[3]]))
	# 		success_rates.append(success_rate)

	# 		min_distance.append(np.mean([np.amin([info["distance_to_target"] for info in sample]) for sample in samples[5]]))
	# 		min_distances.append(min_distance)

	# np.savez_compressed('S.success_rates',**dict(list(zip(envs,success_rates))))
	# np.savez_compressed('S.min_distances',**dict(list(zip(envs,min_distances))))

	envs = ['Reach']
	for env in envs:
		num_obs = 1 if args.reward_type==OBS1 else 2 if args.reward_type==OBS2 else 5
		curriculum_default_config = {'oracle': 'trajectory', 'step_limit': np.inf}
		env_config = deepcopy(default_config)
		env_config.update(curriculum_default_config)
		env_config.update(action_map['trajectory'])
		env_config.update(env_map[env])
		env_config['pr'] = .5

		samplers = [Sampler.remote() for i in range(args.workers)]
		samples = [samplers[i].sample.remote(env_config,1000+i,5000//args.workers) for i in range(args.workers)]
		samples = [ray.get(sample) for sample in samples]

		print("sampling done")
		samples = list(zip(*sum(samples,[])))
		samples = [sum(component,()) for component in samples]
		print(len(samples[0]))

		if args.reward_type == NORM:
			samples[3] = (samples[3]-np.mean(samples[3],axis=0))/np.std(samples[3],axis=0)
			stats_path = os.path.join(os.path.abspath(''),"replays",f"replay.{env[0]}.stats")
			np.savez_compressed(stats_path,mean=np.mean(samples[3],axis=0),std=np.std(samples[3],axis=0))
		buffer = ReplayBuffer(int(1e6),
							spaces.Box(-np.inf,np.inf,(
									(env_config["sa_obs_size"]+env_config['oracle_size'])*env_config['num_obs'],)),
							env_config['action_space'],device='cuda')
		buffer.extend(*samples)

		reward_name = SAMPLE_NAME[args.reward_type]
		path = os.path.join(os.path.abspath(''),"replays",f"replay.{env[0]}.traj")
		with open(path, "wb") as file_handler:
			pickle.dump(buffer, file_handler)

	
