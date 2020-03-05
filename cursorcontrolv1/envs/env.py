from __future__ import division
from copy import deepcopy as copy
from collections import namedtuple

import gym
from gym import spaces
from gym.envs.classic_control import rendering
import numpy as np
from numpy.linalg import norm
from numpy import random

import pygame
from pygame.locals import *

GOAL_THRESH = 2e-3
ORACLE_DIM = 16
MAX_EP_LEN = 50
MAX_VEL = 10
SCREEN = (500,500)


class CursorControl(gym.Env):
  def __init__(self):
    self.screen = pygame.display.set_mode(SCREEN) 

    self.observation_space = spaces.Box(np.array([0]*3+[-np.inf]*ORACLE_DIM+[0]),np.array([1]*3+[np.inf]*ORACLE_DIM+[1]))
    self.action_space = spaces.Box(np.zeros(3),np.array([2*np.pi,MAX_VEL,1]))

    self.max_ep_len = MAX_EP_LEN

    self.init = .5*np.ones(2)
    self._set_goal()

    self.pos = copy(self.init) # position
    self.click = False
    self.prev_action = (0,0,0)
    self.curr_step = 0 # timestep in current episode
    
    self.succ = 0 #True - most recent episode ended in success
    self.prev_obs = np.concatenate((self.init,np.zeros(4)))

  def _set_goal(self):
    goal = random.random(2)
    self.goal = goal
    self.optimal_user_policy = make_oracle_policy(goal)

  def step(self, action):
    vel, click = action[:2],np.rint(action[2])
    vel = np.minimum([2*np.pi, MAX_VEL], np.maximum([0,0], vel))
    opt_act = self.optimal_user_policy(self.pos)

    self.pos += vel
    self.pos = np.minimum(np.ones(2), np.maximum(np.zeros(2), self.pos))
    goal_dist = norm(self.pos-self.goal)
    
    obs = np.array((*self.pos, self.click, *opt_act))
    self.click = click
    self.prev_obs = obs
    self.prev_action = action

    self.succ = goal_dist <= GOAL_THRESH
    r = self.succ + 1/goal_dist

    self.curr_step += 1
    done = self.succ or self.curr_step >= self.max_ep_len

    info = {
      'goal': self.goal, 'succ': self.succ,
      'pos': self.curr_step, 'opt_action': opt_act
    }

    return obs, r, done, info

  def reset(self):
    self.pos = copy(self.init) # position
    self.click = False
    self.curr_step = 0 # timestep in current episode
    prev_obs = self.prev_obs
    self.prev_obs = np.concatenate((self.init,np.zeros(4)))
    return prev_obs

  def render(self):
    self.screen.fill(pygame.color.THECOLORS["white"])
    pygame.draw.circle(self.screen, (10,10,10,0), self.pos.astype(int)*500, 20)
    pygame.draw.circle(self.screen, (76,187,23,0), self.goal.astype(int)*500, 5)
    pygame.draw.line(self.screen, (200, 10, 10), self.pos.astype(int)*500,self.pos.astype(int)*500+\
      (np.array([np.cos(self.prev_action[0]),np.sin(self.prev_action[0])])*self.prev_action[1]*500).astype(int),2)
    pygame.draw.line(self.screen, (10, 10, 200), self.pos*500,self.pos.astype(int)*500+\
      (np.array([np.cos(self.prev_obs[3]),np.sin(self.prev_obs[3])])*self.prev_obs[4]*500).astype(int),2)

# simulate user with optimal intended actions that go directly to the goal
def make_oracle_policy(goal,noise_sd=1):
  def add_noise(action):
    noise = random.normal(np.vstack((np.identity(2),np.zeros((ORACLE_DIM-2,2)))),noise_sd)
    return np.array((*(noise@action[:2]),action[2] != (random.random() < .1))) # flip click with p = .1

  def policy(pos):
    dist = norm(goal-pos)
    return add_noise((*((goal-pos)/dist*MAX_VEL),dist<=GOAL_THRESH))
  return policy

if __name__ == '__main__':
  pygame.init()
  env = CursorControl()
  env.render()
  action = np.array([2*np.pi,MAX_VEL,1])*random.random(3)
  for i in range(int(1e8)):
    obs, r, done, debug = env.step(action)
    action = (*obs[3:5],obs[-1])
    env.render()

