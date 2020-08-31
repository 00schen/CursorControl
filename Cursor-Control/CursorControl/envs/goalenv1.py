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


class GoalControl(gym.Env):
  """ The agent only predicts the goal location, and all action is hard coded """
  MAX_EP_LEN = 30
  GOAL_THRESH = .05
  MAX_VEL = .1

  ORACLE_NOISE = 0
  ORACLE_DIM = 16

  GAMMA = .9

  N = 3

  penalty = 10

  def __init__(self, max_ep_len=30, oracle_noise=.1, gamma=.9, rollout=3, penalty=10, oracle_dim=16):
    self.observation_space = spaces.Box(np.array([-np.inf]*self.ORACLE_DIM*self.N),\
      np.array([np.inf]*self.ORACLE_DIM*self.N))
    self.action_space = spaces.Box(np.zeros(2),np.ones(2))

    self.init = .5*np.ones(2)

    self.MAX_EP_LEN = max_ep_len
    self.ORACLE_NOISE = oracle_noise
    self.ORACLE_DIM = oracle_dim
    self.GAMMA = gamma
    self.penalty = penalty

    self.reset()

    self.do_render = False # will automatically set to true the first time .render is called

  def step(self, action):
    pred_goal, click = action,norm(self.pos-self.goal)<=self.GOAL_THRESH
    comp = pred_goal - self.pos
    vel = ((np.arctan2(comp[1],comp[0])+(2*np.pi))%(2*np.pi), norm(comp))
    # vel = ((np.arctan2(comp[1],comp[0])+(2*np.pi))%(2*np.pi), min(self.MAX_VEL,norm(comp)))

    self.pos += vel[1]*np.array([np.cos(vel[0]),np.sin(vel[0])])
    self.pos = np.minimum(np.ones(2), np.maximum(np.zeros(2), self.pos))
    self.click = click

    self.opt_act = self.optimal_user_policy(self.pos)    

    goal_dist = norm(self.pos-self.goal)
    self.succ = goal_dist <= self.GOAL_THRESH and self.click
      
    obs = np.array(self.opt_act)
    rollout_obs = np.concatenate((obs,self.obs_buffer.flatten()))
    
    self.obs_buffer = np.concatenate(([obs],self.obs_buffer),axis=0)
    self.obs_buffer = np.delete(self.obs_buffer,-1,0)
    self.prev_action = action

    r = 100*self.succ/(1-self.GAMMA)\
      + (1/goal_dist/50 if goal_dist > self.GOAL_THRESH else 1)\
      - self.penalty*(self.click and not self.succ)

    self.curr_step += 1
    done = self.curr_step >= self.MAX_EP_LEN or self.succ

    info = {
      'goal': self.goal, 'is_success': self.succ,
      'pos': self.pos, 'vel': vel,
      'opt_action': self.opt_act, 'step': self.curr_step
    }

    return rollout_obs, r, done, info

  def reset(self):
    # print("reset called")
    self._set_goal()

    self.pos = copy(self.init) # position
    self.click = False
    self.prev_action = np.zeros(2)
    self.curr_step = 0 # timestep in current episode
    
    self.succ = 0 #True - most recent episode ended in success
    self.obs_buffer = np.array([np.zeros(self.ORACLE_DIM)]*(self.N-1))
    return np.concatenate((np.zeros(self.ORACLE_DIM),self.obs_buffer.flatten()))

  def render(self, label=None):
    if not self.do_render:
      self._setup_render()
    self.screen.fill(pygame.color.THECOLORS["white"])
    pygame.draw.circle(self.screen, (10,10,10,0) if not self.click else (150,150,150,0), (self.pos*SCREEN_SIZE).astype(int), 8)
    pygame.draw.circle(self.screen, (76,187,23,0), (self.goal*SCREEN_SIZE).astype(int), 5)
    pygame.draw.circle(self.screen, (240,94,35,0), (self.prev_action*SCREEN_SIZE).astype(int), 5)
    
    comp = self.prev_action - self.pos
    vel = ((np.arctan2(comp[1],comp[0])+(2*np.pi))%(2*np.pi), min(self.MAX_VEL,norm(comp)))
    pygame.draw.line(self.screen, (200, 10, 10), (self.pos*SCREEN_SIZE).astype(int), (self.pos*SCREEN_SIZE).astype(int)+\
      (np.array([np.cos(vel[0]),np.sin(vel[0])])*vel[1]*SCREEN_SIZE).astype(int),2)
    ocomp = self.opt_act[:2] - self.pos
    ovel = comp
    ovel = ((np.arctan2(ocomp[1],ocomp[0])+(2*np.pi))%(2*np.pi), norm(ocomp))
    # ovel = ((np.arctan2(ocomp[1],ocomp[0])+(2*np.pi))%(2*np.pi), min(self.MAX_VEL,norm(ocomp)))
    pygame.draw.line(self.screen, (10, 10, 200), (self.pos*SCREEN_SIZE).astype(int), (self.pos*SCREEN_SIZE).astype(int)+\
      (np.array([np.cos(ovel[0]),np.sin(ovel[0])])*ovel[1]*SCREEN_SIZE).astype(int),2)

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
    self.optimal_user_policy = self.make_oracle_policy(goal)
    self.opt_act = np.zeros(self.action_space.shape[0])

  # simulate user with optimal intended actions that go directly to the goal
  def make_oracle_policy(self, goal):
    noise = random.normal(np.vstack((np.identity(2),np.zeros((self.ORACLE_DIM-2,2)))),self.ORACLE_NOISE)
    lag_buffer = []

    def add_noise(action):
      return noise@action[:2] # flip click with p = .1

    def add_dropout(action):
      return action if random.random() > .1\
        else np.concatenate((self.action_space.sample(),np.random.random(self.ORACLE_DIM-2)))

    def add_lag(action):
      lag_buffer.append(action)
      return lag_buffer.pop(0) if random.random() > .1 else lag_buffer[0]

    def policy(pos):
      comp = goal-pos
      # vel = ((np.arctan2(comp[1],comp[0])+(2*np.pi))%(2*np.pi), min(self.MAX_VEL,norm(comp)))
      return add_dropout(add_noise(goal))

    return policy

class naiveAgent():
  def __init__(self, goal):
    self.goal = goal
  def set_goal(self, goal):
    self.goal = goal
  def predict(self,obs=None,r=None):
    if r == None:
      return self.goal
    return self.goal


if __name__ == '__main__':
  env = GoalControl(oracle_noise=0)
  env.render()
  agent = naiveAgent(env.goal)

  action = agent.predict()
  for i in range(100):
    obs, r, done, debug = env.step(action)
    action = agent.predict(obs,r)
    env.render("test")
    if done:
      break

  env.reset()
  agent.set_goal(env.goal)
  action = agent.predict()
  for i in range(100):
    obs, r, done, debug = env.step(action)
    action = agent.predict(obs,r)
    env.render()
    if done:
      break


