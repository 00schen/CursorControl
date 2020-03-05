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
import time

GOAL_THRESH = .01
ORACLE_DIM = 16
MAX_EP_LEN = 50
MAX_VEL = .1
SCREEN_SIZE = 500


class CursorControl(gym.Env):
  def __init__(self,max_ep_len=MAX_EP_LEN):
    self.observation_space = spaces.Box(np.array([0]*3+[-np.inf]*ORACLE_DIM+[0]),np.array([1]*3+[np.inf]*ORACLE_DIM+[1]))
    self.action_space = spaces.Box(np.zeros(3),np.array([2*np.pi,MAX_VEL,1]))

    self.max_ep_len = max_ep_len

    self.init = .5*np.ones(2)
    
    self.reset()

    self.do_render = False # will automatically set to true the first time .render is called

  def _set_goal(self):
    goal = random.random(2)
    self.goal = goal
    self.optimal_user_policy = make_oracle_policy(goal)

  def step(self, action):
    vel, click = action[:2],np.rint(action[2])
    vel = np.minimum([2*np.pi, MAX_VEL], np.maximum([0,0], vel))
    opt_act = self.optimal_user_policy(self.pos)

    self.pos += vel[1]*np.array([np.cos(vel[0]),np.sin(vel[0])])
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
      'pos': self.pos, 'vel': vel,
      'opt_action': opt_act, 'step': self.curr_step
    }

    return obs, r, done, info

  def reset(self):
    self._set_goal()

    self.pos = copy(self.init) # position
    self.click = False
    self.prev_action = (0,0,0)
    self.curr_step = 0 # timestep in current episode
    
    self.succ = 0 #True - most recent episode ended in success
    self.prev_obs = np.concatenate((self.init,np.zeros(ORACLE_DIM+2)))
    return self.prev_obs

  def render(self):
    if not self.do_render:
      self._setup_render()
    self.screen.fill(pygame.color.THECOLORS["white"])
    pygame.draw.circle(self.screen, (10,10,10,0), (self.pos*SCREEN_SIZE).astype(int), 8)
    pygame.draw.circle(self.screen, (76,187,23,0), (self.goal*SCREEN_SIZE).astype(int), 5)
    pygame.draw.line(self.screen, (200, 10, 10), (self.pos*SCREEN_SIZE).astype(int), (self.pos*SCREEN_SIZE).astype(int)+\
      (np.array([np.cos(self.prev_action[0]),np.sin(self.prev_action[0])])*self.prev_action[1]*SCREEN_SIZE).astype(int),2)
    pygame.draw.line(self.screen, (10, 10, 200), (self.pos*SCREEN_SIZE).astype(int), (self.pos*SCREEN_SIZE).astype(int)+\
      (np.array([np.cos(self.prev_obs[3]),np.sin(self.prev_obs[3])])*self.prev_obs[4]*SCREEN_SIZE).astype(int),2)

    pygame.display.flip()
    self.clock.tick(2)

  def _setup_render(self):
    pygame.init()
    self.screen = pygame.display.set_mode((SCREEN_SIZE,SCREEN_SIZE)) 
    self.clock = pygame.time.Clock()
    self.do_render = True

# simulate user with optimal intended actions that go directly to the goal
def make_oracle_policy(goal,noise_sd=.01):
  def add_noise(action):
    noise = random.normal(np.vstack((np.identity(2),np.zeros((ORACLE_DIM-2,2)))),noise_sd)
    return np.array((*(noise@action[:2]),action[2] != (random.random() < .1))) # flip click with p = .1

  def policy(pos):
    comp = goal-pos
    vel = ((np.arctan2(comp[1],comp[0])+(2*np.pi))%(2*np.pi), min(MAX_VEL,norm(comp)))
    return add_noise((*vel,norm(comp)<=GOAL_THRESH))
  return policy

class naiveAgent():
  def predict(self,obs=None,r=None):
    if r == None:
      return np.array([2*np.pi,MAX_VEL,1])*random.random(3)
    return (*obs[3:5],obs[-1])

if __name__ == '__main__':
  pygame.init()
  env = CursorControl()
  env.render()
  agent = naiveAgent()

  action = agent.predict()
  for i in range(100):
    obs, r, done, debug = env.step(action)
    action = agent.predict(obs,r)
    env.render()
    if done:
      break

  env.reset()

  action = agent.predict()
  for i in range(100):
    obs, r, done, debug = env.step(action)
    action = agent.predict(obs,r)
    env.render()
    if done:
      break



