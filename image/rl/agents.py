import numpy as np
import pybullet as p
from numpy.linalg import norm
import torch as th
import torch.nn.functional as F
from railrl.torch.core import PyTorchModule
from railrl.torch.distributions import Distribution
from railrl.torch.distributions import OneHotCategorical as TorchOneHot
from assistive_gym.envs import JacoReference
import os,sys
# import ppo.a2c_ppo_acktr
# sys.modules['a2c_ppo_acktr'] = ppo.a2c_ppo_acktr
# sys.path.append('a2c_ppo_acktr')

class Agent:
	def __init__(self):
		self.size = 6
	def reset(self):
		pass

"""Oracle Agents"""
import pygame as pg
SCREEN_SIZE = 300
class UserInputOracle(Agent):
	def __init__(self,env):
		super().__init__()
		self.env = env
	def get_input(self):
		pass
	def get_action(self,obs,info=None):
		user_info = self.get_input()
		action = {
			'left': 	np.array([0,1,0,0,0,0]),
			'right': 	np.array([1,0,0,0,0,0]),
			'forward':	np.array([0,0,1,0,0,0]),
			'backward':	np.array([0,0,0,1,0,0]),
			'up':		np.array([0,0,0,0,0,1]),
			'down':		np.array([0,0,0,0,1,0]),
			'noop':		np.array([0,0,0,0,0,0])
		}[self.action]
		print(self.action)
		return action, user_info

class KeyboardOracle(UserInputOracle):
	def get_input(self):
		keys = p.getKeyboardEvents()
		inputs = {
			p.B3G_LEFT_ARROW: 	'left',
			p.B3G_RIGHT_ARROW: 	'right',
			ord('r'):		 	'forward',
			ord('f'):		 	'backward',
			p.B3G_UP_ARROW:		'up',
			p.B3G_DOWN_ARROW:	'down'
		}
		self.action = 'noop'
		for key in inputs:
			if key in keys and keys[key]&p.KEY_WAS_TRIGGERED:
				self.action = inputs[key]
			
		return {"action": self.action}

class MouseOracle(UserInputOracle):
	def get_input(self):
		mouse_pos = pg.mouse.get_pos()
		new_mouse_pos = np.array(mouse_pos)-np.array([SCREEN_SIZE//2,SCREEN_SIZE//2])
		# new_mouse_pos = np.array(mouse_pos)-np.array(self.mouse_pos)
		self.mouse_pos = mouse_pos
		radians = (np.arctan2(*new_mouse_pos) - (np.pi/3) + (2*np.pi)) % (2*np.pi)
		index = np.digitize([radians],np.linspace(0,2*np.pi,7,endpoint=True))[0]
		inputs = {
			1:	'right',
			4:	'left',
			2:	'forward',
			5:	'backward',
			3:	'up',
			6:	'down',
		}
		if norm(new_mouse_pos) > 50:
			self.action = inputs[index]
		else:
			self.action = 'noop'
		return {"mouse_pos": mouse_pos, "action": self.action}

	def reset(self):
		self.mouse_pos = pg.mouse.get_pos()

class UserModelOracle(Agent):
	def __init__(self,env,threshold=.5,epsilon=0,blank=1):
		super().__init__()
		self.get_base_env = env.get_base_env
		self.rng = env.rng
		self.threshold = threshold
		self.epsilon = epsilon
		self.blank = blank
	def get_action(self,obs,info=None):
		rng = self.rng
		base_env = self.get_base_env()
		# if self.prev_noop:
		# 	prob = self.blank*(info['cos_error'] < self.threshold)
		# 	# prob = (info['cos_error'] < self.threshold)*(1-info['cos_error']/(self.threshold+1)+1e-8)
		# else:
		# 	prob = .8
		prob = self.blank*(info['frachet'] < self.threshold)
		action = np.zeros(self.size)
		if rng.random() < prob:
			traj = base_env.target_pos-base_env.tool_pos
			axis = np.argmax(np.abs(traj))
			if rng.random() > self.epsilon:
				index = 2*axis+(traj[axis]>0)
			else:
				index = rng.integers(6)
			action[index] = 1

		self.prev_noop = not np.count_nonzero(action)
		return action, {}
	def reset(self):
		self.prev_noop = True

class PPOModelOracle(Agent):
	def __init__(self,env,):
		super().__init__()
		self.get_base_env = env.get_base_env
		self.ppo, self.ob_rms = th.load(os.path.join(os.path.abspath(''),'trained_models','ppo', env.env_name + ".pt"))
		self.reference_arm = JacoReference(frame_skip=self.get_base_env().frame_skip,time_step=self.get_base_env().time_step)
		
	def get_action(self,obs,info=None):
		obs = info.pop('recreate_full_obs')(obs)
		self.ob_rms.update(obs)
		obs = (obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + 1e-8)
		obs = th.tensor(obs).float()

		_, action, _, _ = self.ppo.act(obs, th.zeros(1, self.ppo.recurrent_hidden_state_size), th.zeros(1, 1), deterministic=True)
		action = action.squeeze()[0].detach().numpy()
		self.reference_arm.step(action,info.pop('joint_pos'))
		traj = self.reference_arm.tool_pos - self.get_base_env().tool_pos
		axis = np.argmax(np.abs(traj))
		index = 2*axis+(traj[axis]>0)
		action = np.zeros(self.size)
		action[index] = 1
		return action, {}

"""Policies"""
class TranslationPolicy(Agent):
	def __init__(self,env,policy,**kwargs):
		super().__init__()
		self.policy = policy
		def joint(action,ainfo={}):
			ainfo['joint'] = action
			return action,ainfo	
		def target(coor,ainfo={}):
			base_env = env.get_base_env()
			ainfo['target'] = coor
			joint_states = p.getJointStates(base_env.robot, jointIndices=base_env.robot_left_arm_joint_indices, physicsClientId=base_env.id)
			joint_positions = np.array([x[0] for x in joint_states])

			link_pos = p.getLinkState(base_env.robot, 13, computeForwardKinematics=True, physicsClientId=base_env.id)[0]
			new_pos = np.array(coor)+np.array(link_pos)-base_env.tool_pos

			new_joint_positions = np.array(p.calculateInverseKinematics(base_env.robot, 13, new_pos, physicsClientId=base_env.id))
			new_joint_positions = new_joint_positions[:7]
			action = new_joint_positions - joint_positions

			clip_by_norm = lambda traj,limit: traj/max(1e-4,norm(traj))*np.clip(norm(traj),None,limit)
			action = clip_by_norm(action,.25)
			return joint(action, ainfo)
		def trajectory(traj,ainfo={}):
			ainfo['trajectory'] = traj
			return target(env.get_base_env().tool_pos+traj,ainfo)
		def disc_traj(onehot,ainfo={}):
			ainfo['disc_traj'] = onehot
			index = np.argmax(onehot)
			traj = [
				np.array((-1,0,0)),
				np.array((1,0,0)),
				np.array((0,-1,0)),
				np.array((0,1,0)),
				np.array((0,0,-1)),
				np.array((0,0,1)),
			][index]*kwargs['traj_len']
			return trajectory(traj,ainfo)
		self.translate = {
			# 'target': target,
			'trajectory': trajectory,
			'joint': joint,
			'disc_traj': disc_traj,
		}[kwargs['action_type']]
	
	def get_action(self,obs):
		return self.translate(*self.policy.get_action(obs))

	def reset(self):
		self.policy.reset()

class OneHotCategorical(Distribution,TorchOneHot):
	def rsample_and_logprob(self):
		s = self.sample()
		log_p = self.log_prob(s)
		return s, log_p

class ArgmaxDiscretePolicy(PyTorchModule):
	def __init__(self, qf1, qf2):
		super().__init__()
		self.qf1 = qf1
		self.qf2 = qf2
		self.action_dim = qf1.output_size

	def get_action(self, obs):
		if isinstance(obs,np.ndarray):
			obs = th.from_numpy(obs).float()
		if next(self.qf1.parameters()).is_cuda:
			obs = obs.cuda()

		with th.no_grad():
			q_values = th.min(self.qf1(obs),self.qf2(obs))
			action = F.one_hot(q_values.argmax(0,keepdim=True),self.action_dim).flatten().detach()
		return action.cpu().numpy(), {}

	def reset(self):
		pass

class BoltzmannPolicy(PyTorchModule):
	def __init__(self, qf1, qf2, logit_scale=100):
		super().__init__()
		self.qf1 = qf1
		self.qf2 = qf2
		self.logit_scale = logit_scale

	def get_action(self, obs):
		if isinstance(obs,np.ndarray):
			obs = th.from_numpy(obs).float()
		if next(self.qf1.parameters()).is_cuda:
			obs = obs.cuda()

		with th.no_grad():
			q_values = th.min(self.qf1(obs),self.qf2(obs))
			action = OneHotCategorical(logits=self.logit_scale*q_values).sample().flatten().detach()
		return action.cpu().numpy(), {}

	def reset(self):
		pass

class OverridePolicy:
	def __init__(self,env,policy,max_overrides):
		self.policy = policy
		self.rng = env.rng
		self.env = env
		self.max = max_overrides
	def get_action(self,obs):
		recommend = self.env.recommend
		action,info = self.policy.get_action(obs)
		if np.count_nonzero(recommend):
			self.count += 1
			if self.count <= self.max:
				return recommend,info
			else:
				return action, info
		else:
			self.count = 0
			return action, info
	def reset(self):
		self.count = 0
		self.policy.reset()

from utils import RunningMeanStd
class ComparisonMergeArgPolicy:
	def __init__(self, env, qf1, qf2, alpha=1):
		super().__init__()
		self.env = env
		self.follower = FollowerPolicy(env)
		self.qf1 = qf1
		self.qf2 = qf2
		self.action_dim = qf1.output_size
		self.alpha = alpha

	def get_action(self, obs):
		if isinstance(obs,np.ndarray):
			obs = th.from_numpy(obs).float()
		if next(self.qf1.parameters()).is_cuda:
			obs = obs.cuda()

		with th.no_grad():
			q_values = th.min(self.qf1(obs),self.qf2(obs))
			action = F.one_hot(q_values.argmax(0,keepdim=True),self.action_dim).flatten().detach()

		# self.noop_rms.update(np.array([not np.count_nonzero(self.env.recommend)]))
		# alpha = self.env.rng.normal(self.noop_rms.mean,np.sqrt(self.noop_rms.var))
		# condition = alpha < .6
		alpha = self.alpha
		condition = self.env.rng.random() < self.alpha
		if condition:
			return action.cpu().numpy(), {"alpha": alpha}
		else:
			action,ainfo = self.follower.get_action(obs)
			ainfo['alpha'] = alpha
			return action,ainfo

	def reset(self):
		self.noop_rms = RunningMeanStd()
		self.follower.reset()

class BayesianMergeArgPolicy:
	def __init__(self, env, qf1, qf2):
		super().__init__()
		self.follower = FollowerPolicy(env)
		self.qf1 = qf1
		self.qf2 = qf2
		self.action_dim = qf1.output_size

	def get_action(self, obs):
		if isinstance(obs,np.ndarray):
			obs = th.from_numpy(obs).float()
		if next(self.qf1.parameters()).is_cuda:
			obs = obs.cuda()

		with th.no_grad():
			q1,q2,da = self.qf1(obs),self.qf2(obs),th.tensor(self.follower.get_action(obs)[0])
			q1 /= -th.std(q1).item()
			q2 /= -th.std(q2).item()
			q1 = self.qf1.alpha*q1 + (1-self.qf1.alpha)*da
			q2 = self.qf2.alpha*q2 + (1-self.qf2.alpha)*da
			q_values = th.min(q1,q2)
			action = F.one_hot(q_values.argmax(0,keepdim=True),self.action_dim).flatten().detach()
		return action.cpu().numpy(), {}

	def reset(self):
		self.follower.reset()

"""Demonstration Policies"""
class FollowerPolicy:
	def __init__(self,env):
		self.env = env
	def get_action(self,obs):
		recommend = self.env.recommend
		if np.count_nonzero(recommend):
			self.action_index = np.argmax(recommend)
		action = np.zeros(6)
		action[self.action_index] = 1
		return action,{}
	def reset(self):
		self.action_index = 0

class EpsilonPolicy:
	def __init__(self,env,epsilon=.25):
		self.epsilon = epsilon
		self.get_base_env = env.get_base_env
		self.rng = env.rng
	def get_action(self,obs):
		rng = self.rng
		base_env = self.get_base_env()
		real_traj = base_env.target_pos-base_env.tool_pos
		real_index = np.argmax(np.abs(real_traj))
		real_index = real_index*2 + real_traj[real_index] > 0
		if self.action_index == real_index:
			self.action_index = rng.choice(6)
		else:
			self.action_index = self.action_index if rng.random() > self.epsilon else rng.choice(6)
		action = np.zeros(6)
		action[self.action_index] = 1
		return action,{}
	def reset(self):
		self.action_index = 0

class DemonstrationPolicy:
	def __init__(self,env,lower_p=.5,upper_p=1):
		self.polcies = [FollowerPolicy(env),EpsilonPolicy(env,epsilon=1/10)]
		self.lower_p = lower_p
		self.upper_p = upper_p
		self.rng = env.rng
	def get_action(self,obs):
		p = [self.p,1-self.p]
		actions = [policy.get_action(obs) for policy in self.polcies]
		act_tuple = self.rng.choice(actions,p=p)
		return act_tuple
	def reset(self):
		self.p = self.rng.random()*(self.upper_p-self.lower_p) + self.lower_p
		# self.p = 1
		for policy in self.polcies:
			policy.reset()
