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
from collections import deque
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
		self.epsilon = epsilon
		self.blank = blank
		self.oracle = {
			"Feeding": StraightLineOracle,
			"Laptop": StraightLineOracle,
			"LightSwitch": LightSwitchOracle,
			"Circle": ReachOracle,
			"Sin": ReachOracle,
		}[env.env_name](self.get_base_env,threshold)
	def get_action(self,obs,info=None):
		rng = self.rng	
		base_env = self.get_base_env()
		criterion,target_pos = self.oracle.query(obs,info)
		action = np.zeros(self.size)	
		if rng.random() < self.blank*criterion:
			traj = target_pos-base_env.tool_pos
			axis = np.argmax(np.abs(traj))
			index = 2*axis+(traj[axis]>0)
			if rng.random() < self.epsilon:
				index = rng.integers(6)
			action[index] = 1
		self.prev_noop = not np.count_nonzero(action)
		return action, {}
	def reset(self):
		self.prev_noop = True
		self.oracle.reset()

class StraightLineOracle(Agent):
	def __init__(self,get_base_env,threshold):
		self.get_base_env =get_base_env
		self.threshold = threshold
	def query(self,obs,info):
		base_env = self.get_base_env()
		criterion = info['cos_error'] < self.threshold
		target_pos = base_env.target_pos
		return criterion, target_pos

class ReachOracle(Agent):
	def __init__(self,get_base_env,threshold):
		self.get_base_env =get_base_env
		self.threshold = threshold
	def query(self,obs,info):
		base_env = self.get_base_env()
		criterion = info['distance_to_target'] > self.threshold
		target_pos = base_env.target_pos
		return criterion, target_pos

class LightSwitchOracle(Agent):
	def __init__(self,get_base_env,threshold):
		super().__init__()
		self.get_base_env = get_base_env
		self.threshold = threshold
		# file_name = os.path.join(os.path.abspath(''),'li_oracle_params.pkl')
		# self.policy = ArgmaxDiscretePolicy(
		# 	qf1=th.load(file_name,map_location=th.device("cpu"))['trainer/qf1'],
		# 	qf2=th.load(file_name,map_location=th.device("cpu"))['trainer/qf2'],
		# )

	def query(self,obs,info):
		base_env = self.get_base_env()

		target_indices = np.nonzero(np.not_equal(base_env.target_string,base_env.current_string))[0]
		switches = np.array(base_env.switches)[target_indices]
		target_poses1 = np.array(base_env.target_pos1)[target_indices]
		target_poses = np.array(base_env.target_pos)[target_indices]

		bad_contact =  np.any(np.logical_and(info['angle_dir'] != 0,
							np.logical_or(np.logical_and(info['angle_dir'] < 0, base_env.target_string == 1),
								np.logical_and(info['angle_dir'] > 0, base_env.target_string == 0))))
		info['bad_contact'] = bad_contact
		ineff_contact = (info['ineff_contact'] and np.all(np.abs(info['angle_dir']) < .005))\
						and min(norm(np.array(base_env.target_pos)-base_env.tool_pos,axis=1)) < .2
		info['ineff_contact'] = ineff_contact
		self.ineff_contacts.append(ineff_contact)
		self.bad_contacts.append(bad_contact)

		if len(target_indices) == 0:
			info['cos_error'] = 1
			info['distance_to_target'] = 0
			return False, np.zeros(3)

		
		if np.sum(self.bad_contacts) > 1:
			tool_pos = base_env.tool_pos
			on_off = base_env.target_string[target_indices[0]]

			_,switch_orient = p.getBasePositionAndOrientation(base_env.wall, physicsClientId=base_env.id)
			target_pos2 = [0,.2,.1] if on_off == 0 else [0,.2,-.1]
			target_pos2 = np.array(p.multiplyTransforms(target_poses[0], switch_orient, target_pos2, [0, 0, 0, 1])[0])

			sphere_collision = -1
			sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=.03, rgbaColor=[0, 1, 1, 1], physicsClientId=base_env.id)
			target = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=sphere_collision, baseVisualShapeIndex=sphere_visual, basePosition=target_pos2,
								useMaximalCoordinates=False, physicsClientId=base_env.id)

			if ((base_env.tool_pos[2] < target_poses[0][2] and on_off == 0)
				or (base_env.tool_pos[2] > target_poses[0][2] and on_off == 1)):
				threshold = .5
				target_pos = target_pos2
			elif ((base_env.tool_pos[2] > target_poses[0][2] + .15 and on_off == 0)
				or (base_env.tool_pos[2] < target_poses[0][2] - .15 and on_off == 1)):
				threshold = 0
				target_pos = target_poses1[0]
			else:
				threshold = .5
				target_pos = tool_pos + np.array([0,0,1]) if on_off == 0 else tool_pos + np.array([0,0,-1])
		elif np.sum(self.ineff_contacts) > 2:
			on_off = base_env.target_string[target_indices[0]]
			_,switch_orient = p.getBasePositionAndOrientation(base_env.wall, physicsClientId=base_env.id)
			target_pos = np.array(p.multiplyTransforms(target_poses[0], switch_orient, [0,1,0], [0, 0, 0, 1])[0])
			threshold = .5
		elif norm(base_env.tool_pos-target_poses1,axis=1)[0] < .25:
			if norm(base_env.tool_pos-target_poses1,axis=1)[0] > .12:
				threshold = self.threshold
				target_pos = target_poses1[0]
			else:
				threshold = 0
				target_pos = target_poses[0]
		else:
			threshold = .5
			target_pos = target_poses1[0]

			
		old_traj = target_pos - info['old_tool_pos']
		new_traj = base_env.tool_pos - info['old_tool_pos']
		info['cos_error'] = np.dot(old_traj,new_traj)/(norm(old_traj)*norm(new_traj))
		criterion = info['cos_error'] < threshold
		info['distance_to_target'] = norm(base_env.tool_pos-target_pos)
		return criterion, target_pos
	def reset(self):
		self.bad_contacts = deque(np.zeros(10),10)
		self.ineff_contacts = deque(np.zeros(10),10)

class LightSwitchTrainOracle(Agent):
	def __init__(self,env,qf1, qf2, epsilon=.8):
		super().__init__()
		self.env = env
		self.epsilon = epsilon
		self.policy = ArgmaxDiscretePolicy(
			qf1=qf1,
			qf2=qf2,
		)

	def get_action(self, obs):
		base_env = self.env.base_env
		rng = self.env.rng

		switches = np.array(base_env.switches)[np.nonzero(np.not_equal(base_env.target_string,base_env.current_string))]
		target_poses = np.array(base_env.target_pos)[np.nonzero(np.not_equal(base_env.target_string,base_env.current_string))]
		if len(target_poses) == 0:
			return np.array([1,0,0,0,0,0])

		if norm(target_poses[0]-base_env.tool_pos) < .12:
			action,_info = self.policy.get_action(obs)
		else:
			action = np.zeros(6)
			current_traj = [
				np.array((-1,0,0)),
				np.array((1,0,0)),
				np.array((0,-1,0)),
				np.array((0,1,0)),
				np.array((0,0,-1)),
				np.array((0,0,1)),
			][self.current_index]
			correct_traj = target_poses[0] - base_env.tool_pos
			cos = np.dot(current_traj,correct_traj)/(norm(current_traj)*norm(correct_traj))
			if cos < .5:
				traj = target_poses[0]-base_env.tool_pos
				axis = np.argmax(np.abs(traj))
				self.current_index = index = 2*axis+(traj[axis]>0)
				if rng.random() < self.epsilon:
					index = rng.integers(6)
				action[index] = 1
			else:
				action[self.current_index] = 1
		return action,{}
	def reset(self):
		self.current_index = 0

"""Policies"""
class TranslationPolicy(Agent):
	def __init__(self,env,policy,**kwargs):
		super().__init__()	
		self.smooth_alpha = kwargs['smooth_alpha']
		self.policy = policy
		def joint(action,ainfo={}):
			self.action = self.smooth_alpha*action + (1-self.smooth_alpha)*self.action if np.count_nonzero(self.action) else action
			ainfo['joint'] = self.action
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
		self.action = np.zeros(7)
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
	def __init__(self,env,policy):
		self.policy = policy
		self.rng = env.rng
		self.env = env
	def get_action(self,obs):
		recommend = self.env.recommend
		action,info = self.policy.get_action(obs)
		if np.count_nonzero(recommend):
			return recommend,info
		else:
			return action, info
	def reset(self):
		self.policy.reset()

# from utils import RunningMeanStd
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

		# alpha = self.env.rng.normal(self.noop_rms.mean,np.sqrt(self.noop_rms.var))
		# condition = alpha < .6
		# alpha = max(self.alpha,
		# condition = self.env.rng.random() < self.alpha
		# if condition:
		# 	self.noop_rms.update(np.array([not np.count_nonzero(self.env.recommend)]))
		# 	return action.cpu().numpy(), {"alpha": alpha}
		# else:
		# 	action,ainfo = self.follower.get_action(obs)
		# 	ainfo['alpha'] = alpha
		# 	return action,ainfo

	def reset(self):
		# self.noop_rms = RunningMeanStd()
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

class RandomPolicy:
	def __init__(self,env,epsilon=.25):
		self.epsilon = epsilon
		self.get_base_env = env.get_base_env
		self.rng = env.rng
	def get_action(self,obs):
		rng = self.rng
		self.action_index = self.action_index if rng.random() > self.epsilon else rng.choice(6)
		action = np.zeros(6)
		action[self.action_index] = 1
		return action,{}
	def reset(self):
		self.action_index = 0

class DemonstrationPolicy:
	def __init__(self,env,lower_p=.5,upper_p=1):
		self.polcies = [FollowerPolicy(env),RandomPolicy(env,epsilon=1/10)]
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
