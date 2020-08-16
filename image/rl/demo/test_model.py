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
from railrl.torch.core import PyTorchModule
import torch.nn.functional as F

# from discrete_experiment import ArgmaxDiscretePolicy

parser = argparse.ArgumentParser()
parser.add_argument('--env_name',)
args, _ = parser.parse_known_args()

class ArgmaxDiscretePolicy(PyTorchModule):
	def __init__(self, qf):
		super().__init__()
		self.qf = qf

	def get_action(self, obs):
		if isinstance(obs,np.ndarray):
			obs = torch.from_numpy(obs).float()
		if next(self.qf.parameters()).is_cuda:
			obs = obs.cuda()
		q_values = self.qf(obs)
		action = F.one_hot(q_values.argmax(0,keepdim=True),list(q_values.size())[0]).cpu().flatten().detach().numpy()
		return action, {}

	def reset(self):
		pass

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
		# oracle='rad_discrete_traj',
		oracle='user',
		num_obs=10,
		num_nonnoop=10,
		threshold=.3,
		input_penalty=.1,
		action_type='cat_target',
		action_penalty=0,
		include_target=True,
		# target_delay=80,
		video_config = dict(
			video_path=os.path.join(dirname,'videos'),
			video_name='test_lightswitch',
			video_episodes=8,
		)
	))

	# import pkgutil
	# egl = pkgutil.get_loader('eglRenderer')
	# import pybullet_data

	env = default_class(config['env_name'])(config)
	# env = video_factory(demonstration_factory(default_class(config['env_name'])))(config)
	print(torch.load("params.pkl",map_location=torch.device("cpu")))
	agent = ArgmaxDiscretePolicy(torch.load("params.pkl",map_location=torch.device("cpu"))['trainer/qf']) 
	env.seed(45)

	# p.setAdditionalSearchPath(pybullet_data.getDataPath())
	# plugin = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
	# print("plugin=", plugin)
	env.render('human')
	
	obs = env.reset()

	i = 0
	while i < config['video_config']['video_episodes']:
		# action = agent(torch.tensor(obs).float()).sample().numpy()
		action = agent.get_action(obs)[0]
		obs,r,done,info = env.step(action)
		if done:
			i+=1
			done = False
			# env.new_target()
			obs = env.reset()
			print(env.env.target_index)
			print("episode",i)
	env.close()

	# p.unloadPlugin(plugin)
