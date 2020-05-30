import assistive_gym
import gym

import numpy as np
import numpy.random as random

import os,sys

from tqdm import tqdm
import argparse

dirname = os.path.dirname(os.path.abspath(__file__))
parentname = os.path.dirname(dirname)
sys.path.append(parentname)

from utils import *

env_name = 'ScratchItchJaco-v1'
env_name0 = env_name[:-1]+'0'
env_name1 = env_name[:-1]+'1'
pretrain_path = os.path.join(dirname,'trained_models','ppo',env_name0+'.pt')
predictor_path = os.path.join(parentname,'model_combine1.h5')


parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=1000, help="seed for environment", type=int)
parser.add_argument("--render", default=False, help="will collect data if False", type=bool)
args, _ = parser.parse_known_args()

if __name__ == '__main__':
	env = gym.make(env_name)
	env.seed(args.seed)

	pretrain = PretrainAgent(pretrain_path)
	# oracle = TrajectoryOracle(pretrain,env.oracle2trajectory,env_name1)
	oracle = TargetOracle(pretrain,env,env_name1)
	pretrain.add()
	agent = BufferAgent(pretrain,Predictor('twin',predictor_path),env.target2obs)

	if args.render:
		env.render()
	else:
		target = []
		pred_target = []
		label = []
		pred_label = []
	for i in tqdm(range(25)):
		obs = env.reset()

		oracle.reset()
		obs = env.observation_space.sample()
		obs = oracle.predict(obs)
		# obs['action'] = np.random.random((3,))

		agent.reset()
		action = agent.predict(obs)
		# action = obs['real_action']
		
		if not args.render:
			"""data collection"""
			target.append([])
			pred_target.append([])
			label.append(env.label)
			pred_label.append([])
		for i in range(100):
			if args.render:
				env.render()
				print("real class: ", env.label, "predicted class: ", agent.predictor.label_prediction)
			else:
				"""data collection"""
				target[-1].append(env.target_pos)
				pred_target[-1].append(agent.prediction_buffer[-1])
				pred_label[-1].append(agent.predictor.label_prediction)

			obs,_r,done,_info = env.step(action)
			obs = oracle.predict(obs,done)
			# obs['action'] = np.random.random((3,))

			action = agent.predict(obs,done)
			# action = obs['real_action']
	env.close()

	data = {'target':target,'pred_target':pred_target, 'label':label, 'pred_label':pred_label}
	np.savez_compressed(os.path.join(parentname,'test',f'combine1_target_results_{args.seed}'),**data)