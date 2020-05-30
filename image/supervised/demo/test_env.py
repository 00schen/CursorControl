import assistive_gym
import gym

import numpy as np
import numpy.random as random

import os,sys

dirname = os.path.dirname(os.path.abspath(__file__))
parentname = os.path.dirname(dirname)
sys.path.append(parentname)

import argparse
# from keypoller import KeyPoller

env_name = 'LaptopJaco-v0'
env_name0 = env_name[:-1]+'0'
env_name1 = env_name[:-1]+'1'

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=1000, help="seed for environment")
args, _ = parser.parse_known_args()

class KeyboardAgent:
	key_mappings = {
		'a':'left',
		'd':'right',
		's':'backward',
		'w':'forward',
		'z':'down',
		'x':'up',
	}

	# def __init__(self):
		# self.poller = KeyPoller()

	def predict(self,obs):
		# key = self.poller.poll()
		key = input('direction: ')
		if key in self.key_mappings:
			return self.key_mappings[key]
		else:
			return None

if __name__ == '__main__':
	env = assistive_gym.TestEnv(env_name)
	env.seed(int(args.seed))

	agent = KeyboardAgent()

	env.render()
	while True:
		obs = env.reset()
		for i in range(100):
			action = agent.predict(obs)
			obs,_r,done,_info = env.step(action)

			env.render()
	env.close()