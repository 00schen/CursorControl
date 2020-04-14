from copy import deepcopy as copy
from collections import namedtuple

import gym
from gym import spaces
import numpy as np
from numpy.linalg import norm
from numpy import random

import pygame
from pygame.locals import *
import time

SCREEN_SIZE = 500


class VelocityControl(gym.Env):
  """ The agent predicts next action as well as whether to "click", not knowing the target"""
  GOAL_THRESH = .05
  MAX_VEL = .1

  NUM_OBSTACLES = 5
  OBSTACLE_WIDTH = np.array((.2,.3))

  def __init__(self, oracle = None, max_ep_len=30, oracle_noise=.05, gamma=.9, rollout=3, penalty=10, oracle_dim=2):
    self.MAX_EP_LEN = max_ep_len
    self.GAMMA = gamma
    self.penalty = penalty
    self.N = rollout

    self.ORACLE_NOISE = oracle_noise
    self.ORACLE_DIM = oracle_dim
    
    if oracle:
      self.oracle = oracle
      self.observation_space = spaces.Box(np.array(([0]*3+[-np.inf]*self.ORACLE_DIM+[0])*self.N),\
        np.array(([1]*3+[np.inf]*self.ORACLE_DIM+[1])*self.N))
    else: # Used for training the Oracle
      self.oracle = NaiveAgent(self)
      self.observation_space = spaces.Box(np.array([0]*3*self.N+[0]*2),np.array([1]*3*self.N+[1]*2))
    
    self.action_space = spaces.Box(np.zeros(3),np.array([2*np.pi,self.MAX_VEL,1]))

    self.init = .5*np.ones(2)

    self.reset()

    self.do_render = False # will automatically set to true the first time .render is called

  def step(self, action):
    vel, click = action[:2],np.rint(action[2])
    vel = np.minimum([2*np.pi, self.MAX_VEL], np.maximum([0,0], vel))

    self.pos += vel[1]*np.array([np.cos(vel[0]),np.sin(vel[0])])
    self.pos = np.minimum(np.ones(2), np.maximum(np.zeros(2), self.pos))
    self.click = click

    oracle_rollout_obs = np.concatenate(((*self.pos, self.click),self.obs_buffer[:-1,:3].flatten(),self.goal))
    self.opt_act = self.noise.get_noise(self.oracle.predict(oracle_rollout_obs)[0])    
  
    obs = np.array((*self.pos, self.click, *self.opt_act)) if self.oracle else np.array((*self.pos, self.click))
        
    self.obs_buffer = np.concatenate(([obs],self.obs_buffer),axis=0)
    self.obs_buffer = np.delete(self.obs_buffer,-1,0)
    rollout_obs = self.obs_buffer.flatten() if self.oracle else np.concatenate((self.obs_buffer.flatten(),self.goal))
    self.prev_action = copy(action)

    goal_dist = norm(self.pos-self.goal)
    self.success = goal_dist <= self.GOAL_THRESH and self.click

    # r = 100*self.success/(1-self.GAMMA)\
    r = 100*self.success\
      - self.penalty*(self.click and not self.success)
    if self.oracle: r += (np.sqrt(self.GOAL_THRESH/goal_dist) if goal_dist > self.GOAL_THRESH else 1)

    self.curr_step += 1
    done = self.curr_step >= self.MAX_EP_LEN
    # done = self.curr_step >= self.MAX_EP_LEN or self.success

    info = {
      'goal': self.goal, 'is_success': self.success,
      'pos': self.pos, 'vel': vel,
      'opt_action': self.opt_act, 'step': self.curr_step
    }

    return rollout_obs, r, done, info

  def reset(self):
    self._set_goal()

    self.pos = copy(self.init) # position
    self.click = False
    self.obs_buffer = np.array([[*self.init,self.click]+[0]*(self.ORACLE_DIM+1)]*self.N) if self.oracle \
      else np.array([[*self.init,self.click]]*self.N)
    self.prev_action = np.zeros(3)
    self.curr_step = 0 # timestep in current episode
    
    self.noise = Noise(self, self.ORACLE_NOISE, self.ORACLE_DIM)
    self.opt_act = np.zeros(3)
    self.success = False

    return self.obs_buffer.flatten() if self.oracle else np.concatenate((self.obs_buffer.flatten(),self.goal))

  def render(self, label=None):
    if not self.do_render:
      self._setup_render()
    self.screen.fill(pygame.color.THECOLORS["white"])
    pygame.draw.circle(self.screen, (10,10,10,0) if not self.click else (150,150,150,0), (self.pos*SCREEN_SIZE).astype(int), 8)
    pygame.draw.circle(self.screen, (76,187,23,0), (self.goal*SCREEN_SIZE).astype(int), 5)
    pygame.draw.line(self.screen, (200, 10, 10), (self.pos*SCREEN_SIZE).astype(int), (self.pos*SCREEN_SIZE).astype(int)+\
      (np.array([np.cos(self.prev_action[0]),np.sin(self.prev_action[0])])*self.prev_action[1]*SCREEN_SIZE).astype(int),2)
    pygame.draw.line(self.screen, (10, 10, 200), (self.pos*SCREEN_SIZE).astype(int), (self.pos*SCREEN_SIZE).astype(int)+\
      (np.array([np.cos(self.opt_act[0]),np.sin(self.opt_act[0])])*self.opt_act[1]*SCREEN_SIZE).astype(int),2)

    if label:
      font = pygame.font.Font(None, 24)
      text = label
      text = font.render(text, 1, pygame.color.THECOLORS["black"])
      self.screen.blit(text, (5,5))

    pygame.display.flip()
    self.clock.tick(2)

  def _setup_render(self):
    pygame.init()
    self.screen = pygame.display.set_mode((SCREEN_SIZE,SCREEN_SIZE)) 
    self.clock = pygame.time.Clock()
    self.do_render = True

  def _set_goal(self):
    goal = random.random(2)
    self.goal = goal



class Noise():
  def __init__(self, env, sd, dim):
    self.SD = sd
    self.DIM = dim

    self.env = env

    self.projection = np.vstack((np.identity(2),np.zeros((self.DIM-2,2))))
    self.noise = random.normal(np.identity(self.DIM),self.SD)
    self.lag_buffer = []

  def _add_grp(self, action):
    return np.array((*(self.noise@action[:-1]),action[-1] != (random.random() < .1))) # flip click with p = .1

  def _add_dropout(self, action):
    return action if random.random() > .1\
      else np.concatenate((self.env.action_space.sample(),random.random(self.DIM-2)))

  def _add_lag(self, action):
    self.lag_buffer.append(action)
    return self.lag_buffer.pop(0) if random.random() > .1 else self.lag_buffer[0]

  def get_noise(self,action):
    return self._add_lag(self._add_dropout(self._add_grp(action)))
  

class NaiveAgent():
  def __init__(self,env):
    self.env = env
  def predict(self,obs=[]):
    if len(obs) == 0:
      return self.env.action_space.sample()
    comp = self.env.goal - self.env.pos
    return [((np.arctan2(comp[1],comp[0])+(2*np.pi))%(2*np.pi),\
      min(self.env.MAX_VEL,norm(comp)),norm(comp) < self.env.GOAL_THRESH)]

if __name__ == '__main__':
  env = VelocityControl()
  env.render()
  agent = NaiveAgent(env)

  action = agent.predict()
  for i in range(30):
    obs, r, done, debug = env.step(action)
    action = agent.predict(obs)
    env.render("test")
    if done:
      break

  env.reset()
  action = agent.predict()
  for i in range(30):
    obs, r, done, debug = env.step(action)
    action = agent.predict(obs)
    env.render("test")
    if done:
      break



