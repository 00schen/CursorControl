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
  """ The agent predicts goal position as well as whether to "click", and velocity is not limited """
  MAX_EP_LEN = 30
  GOAL_THRESH = .05

  ORACLE_NOISE = 0
  ORACLE_DIM = 16

  GAMMA = .9

  N = 3

  penalty = 10

  def __init__(self, max_ep_len=30, oracle_noise=.1, gamma=.9, rollout=3, penalty=10, oracle_dim=16):
    self.observation_space = spaces.Box(np.array(([0]*3+[-np.inf]*self.ORACLE_DIM+[0])*self.N),\
      np.array(([1]*3+[np.inf]*self.ORACLE_DIM+[1])*self.N))
    self.action_space = spaces.Box(np.zeros(3),np.ones(3))

    self.init = .5*np.ones(2)

    self.MAX_EP_LEN = max_ep_len
    self.ORACLE_NOISE = oracle_noise
    self.ORACLE_DIM = oracle_dim
    self.GAMMA = gamma
    self.penalty = penalty

    self.reset()

    self.do_render = False # will automatically set to true the first time .render is called

  def step(self, action):
    pred_goal, click = action[:2],np.rint(action[2])
    comp = pred_goal - self.pos
    vel = ((np.arctan2(comp[1],comp[0])+(2*np.pi))%(2*np.pi), norm(comp))

    self.pos += vel[1]*np.array([np.cos(vel[0]),np.sin(vel[0])])
    self.pos = np.minimum(np.ones(2), np.maximum(np.zeros(2), self.pos))
    self.click = click

    self.opt_act = self.optimal_user_policy(self.pos)    

    goal_dist = norm(self.pos-self.goal)
    self.succ = goal_dist <= self.GOAL_THRESH and self.click
      
    

    r = 100*self.succ/(1-self.GAMMA)\
      + (1/goal_dist/50 if goal_dist > self.GOAL_THRESH else 1)\
      - self.penalty*(self.click and not self.succ)\
      - 10*norm(self.prev_action[:2]-action[:2])

    obs = np.array((*self.pos, self.click, *self.opt_act))
    rollout_obs = np.concatenate((obs,self.obs_buffer.flatten()))
    
    self.obs_buffer = np.concatenate(([obs],self.obs_buffer),axis=0)
    self.obs_buffer = np.delete(self.obs_buffer,-1,0)
    self.prev_action = action
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
    self.prev_action = np.zeros(3)
    self.curr_step = 0 # timestep in current episode
    
    self.succ = 0 #True - most recent episode ended in success
    self.obs_buffer = np.array([np.zeros(3+self.ORACLE_DIM+1)]*(self.N-1))
    return np.concatenate((self.init,np.zeros(self.ORACLE_DIM+2),self.obs_buffer.flatten()))

  def render(self, label=None):
    if not self.do_render:
      self._setup_render()
    self.screen.fill(pygame.color.THECOLORS["white"])
    pygame.draw.circle(self.screen, (10,10,10,0) if not self.click else (150,150,150,0), (self.pos*SCREEN_SIZE).astype(int), 8)
    pygame.draw.circle(self.screen, (76,187,23,0), (self.goal*SCREEN_SIZE).astype(int), 5)
    pygame.draw.circle(self.screen, (240,94,35,0), (self.prev_action[0:2]*SCREEN_SIZE).astype(int), 5)
    
    ocomp = self.opt_act[:2] - self.pos
    ovel = ((np.arctan2(ocomp[1],ocomp[0])+(2*np.pi))%(2*np.pi), norm(ocomp))
    pygame.draw.line(self.screen, (10, 10, 200), (self.pos*SCREEN_SIZE).astype(int), (self.pos*SCREEN_SIZE).astype(int)+\
      (np.array([np.cos(ovel[0]),np.sin(ovel[0])])*ovel[1]*SCREEN_SIZE).astype(int),2)
    comp = self.prev_action[0:2] - self.pos
    vel = ((np.arctan2(comp[1],comp[0])+(2*np.pi))%(2*np.pi), norm(comp))
    pygame.draw.line(self.screen, (200, 10, 10), (self.pos*SCREEN_SIZE).astype(int), (self.pos*SCREEN_SIZE).astype(int)+\
      (np.array([np.cos(vel[0]),np.sin(vel[0])])*vel[1]*SCREEN_SIZE).astype(int),2)


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
    projection = np.vstack((np.identity(2),np.zeros((self.ORACLE_DIM-2,2))))
    lag_buffer = []

    def add_noise(action):
      noise = random.normal(np.identity(self.ORACLE_DIM),self.ORACLE_NOISE)
      return np.array((*(noise@action[:-1]),action[-1] != (random.random() < .1))) # flip click with p = .1

    def add_dropout(action):
      return action if random.random() > .1\
        else np.concatenate((self.action_space.sample(),np.random.random(self.ORACLE_DIM-2)))

    def add_lag(action):
      lag_buffer.append(action)
      return lag_buffer.pop(0) if random.random() > .1 else lag_buffer[0]

    def policy(pos):
      comp = goal-pos
      return add_dropout(add_noise((*(projection@goal),norm(comp)<= self.GOAL_THRESH)))

    return policy

class naiveAgent():
  def __init__(self, goal):
    self.goal = goal
  def set_goal(self,goal):
    self.goal = goal
  def predict(self,obs=None,r=None):
    if r == None:
      return np.array((*self.goal,False))
    return np.array((*self.goal,obs[-1]))


if __name__ == '__main__':
  env = GoalControl(oracle_noise=.05)
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



