import os,sys
import argparse
from copy import deepcopy
from types import MethodType
import torch
import numpy as np
import pybullet as p

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs import *
from video_recorder import *
dirname = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--env_name',)
parser.add_argument('--exp_name', default='a-test')
args, _ = parser.parse_known_args()

def demonstration_factory(base):
	class DemonstrationEnv(base):
		def __init__(self,config):
			super().__init__(config)
			self.target_index = 0
			self.target_count = 0

		def new_target(self):
			self.target_index += 1
			if self.target_index == self.env.num_targets:
				self.target_index %= self.env.num_targets
				self.target_count += 1
		def reset(self):
			target_index = self.target_index
			def generate_target(self,index):
				nonlocal target_index
				self.__class__.generate_target(self,target_index)
			self.env.generate_target = MethodType(generate_target,self.env)
			return super().reset()
	return DemonstrationEnv

if __name__ == "__main__":
	config = deepcopy(default_config)
	config.update(env_map[args.env_name])
	config.update(dict(
		oracle_size=6,
		oracle='rad_discrete_traj',
		num_obs=10,
		num_nonnoop=10,
		threshold=.3,
		input_penalty=.1,
		action_type='basis_target',
		action_penalty=0,
		include_target=True,
		# target_delay=80,
		video_config = dict(
			video_path=os.path.join(dirname,'videos'),
			video_name='test_lightswitch',
			video_episodes=8,
		)
	))

	import pkgutil
	egl = pkgutil.get_loader('eglRenderer')
	import pybullet_data

	# env = default_class(config['env_name'])(config)
	env = video_factory(demonstration_factory(default_class(config['env_name'])))(config)
	agent = torch.load("test27c.pkl",map_location=torch.device("cpu"))['trainer/policy']

	# env.render('human')
	p.setAdditionalSearchPath(pybullet_data.getDataPath())
	plugin = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
	print("plugin=", plugin)
	
	obs = env.reset()

	i = 0
	while i < 8:
		action = agent(torch.tensor(obs).float()).sample().numpy()
		obs,r,done,info = env.step(action)
		if done:
			i+=1
			done = False
			env.new_target()
			obs = env.reset()
			print(env.env.target_index)
			print("episode",i)
	env.close()

	p.unloadPlugin(plugin)
