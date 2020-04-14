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


class VelocityControl(gym.GoalEnv):
  """ Version 1 copy only for pretraining 2D Oracle using HER """
  GOAL_THRESH = .05
  MAX_VEL = .1

  NUM_OBSTACLES = 5
  OBSTACLE_WIDTH = np.array((.2,.3))

  def __init__(self, max_ep_len=30, oracle_noise=.05, gamma=.9, rollout=3, penalty=10, oracle_dim=2):
    self.MAX_EP_LEN = max_ep_len
    self.GAMMA = gamma
    self.penalty = penalty
    self.N = rollout

    self.ORACLE_NOISE = oracle_noise
    self.ORACLE_DIM = oracle_dim
    
    self.observation_space = spaces.Dict({
      'observation': spaces.Box(np.array([0]*(3*self.N+self.NUM_OBSTACLES*4)),np.array([1]*(3*self.N+self.NUM_OBSTACLES*4))),
      'desired_goal': spaces.Box(np.zeros(3), np.ones(3)),
      'achieved_goal': spaces.Box(np.zeros(3), np.ones(3)) })\
    
    self.action_space = spaces.Box(np.zeros(3),np.array([2*np.pi,self.MAX_VEL,1]))

    self.init = .5*np.ones(2)

    self.reset()

    self.do_render = False # will automatically set to true the first time .render is called

  def step(self, action):
    vel, click = action[:2],np.rint(action[2])
    vel = np.minimum([2*np.pi, self.MAX_VEL], np.maximum([0,0], vel))

    trajectory = vel[1]*np.array([np.cos(vel[0]),np.sin(vel[0])])
    blocked = self._is_collision(self.pos+trajectory)
    if not blocked:
      self.pos += trajectory
      self.pos = np.minimum(np.ones(2), np.maximum(np.zeros(2), self.pos))
    self.click = click

    obs = np.array((*self.pos, self.click))
        
    self.obs_buffer = np.concatenate(([obs],self.obs_buffer),axis=0)
    self.obs_buffer = np.delete(self.obs_buffer,-1,0)
    rollout_obs = self.obs_buffer.flatten()
    self.prev_action = copy(action)

    goal_dist = norm(self.pos-self.goal)
    self.success = goal_dist <= self.GOAL_THRESH and self.click

    r = 100*self.success\
      - self.penalty*(self.click and not self.success)
      # - self.penalty*blocked

    self.curr_step += 1
    done = self.curr_step >= self.MAX_EP_LEN

    info = {
      'goal': self.goal, 'is_success': self.success,
      'pos': self.pos, 'vel': vel, 'step': self.curr_step
    }

    return {'observation': np.concatenate((rollout_obs,self.obstacles.flatten())),
     'desired_goal': np.array((*self.goal,1)),
     'achieved_goal': np.array((*self.pos,self.click))}, r, done, info
  
  def compute_reward(self, acheived, desired, info):
    """ must be equal to reward if returning desired and achieved """
    click = np.rint(acheived[2])
    goal_dist = norm(acheived[:2]-desired[:2])
    success = goal_dist <= self.GOAL_THRESH and click
    r = 100*success\
      - self.penalty*(click and not success)
      # - self.penalty*blocked
    return r

  def reset(self):
    self._set_goal()
    self._set_obstacles()

    self.pos = copy(self.init) # position
    self.click = False
    self.obs_buffer = np.array([[*self.init,self.click]]*self.N)
    self.prev_action = np.zeros(3)
    self.curr_step = 0 # timestep in current episode
    
    self.noise = Noise(self, self.ORACLE_NOISE, self.ORACLE_DIM)
    self.success = False

    return {'observation': np.concatenate((self.obs_buffer.flatten(),self.obstacles.flatten())),
     'desired_goal': np.array((*self.goal,1)),
     'achieved_goal': np.array((*self.pos,self.click))}

  def render(self, label=None):
    if not self.do_render:
      self._setup_render()
    self.screen.fill(pygame.color.THECOLORS["white"])
    pygame.draw.circle(self.screen, (10,10,10,0) if not self.click else (150,150,150,0), (self.pos*SCREEN_SIZE).astype(int), 8)
    pygame.draw.circle(self.screen, (76,187,23,0), (self.goal*SCREEN_SIZE).astype(int), 5)
    pygame.draw.line(self.screen, (200, 10, 10), (self.pos*SCREEN_SIZE).astype(int), (self.pos*SCREEN_SIZE).astype(int)+\
      (np.array([np.cos(self.prev_action[0]),np.sin(self.prev_action[0])])*self.prev_action[1]*SCREEN_SIZE).astype(int),2)
    for p,q in self.obstacles:
      pygame.draw.line(self.screen, (10,10,10,0), (p*SCREEN_SIZE).astype(int), (q*SCREEN_SIZE).astype(int), 2)

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

  def _set_obstacles(self):
    obstacles = []
    for _i in range(self.NUM_OBSTACLES):
      approach_angle = np.arctan2(*((self.init - self.goal)[[1,0]]))
      qrad,qangle,prad,pangle = random.uniform(*self.OBSTACLE_WIDTH),2*np.pi*random.random(),\
         .5*random.random(), random.uniform(approach_angle-np.pi/3,approach_angle+np.pi/3)
      p = self.goal + prad*np.array([np.cos(pangle),np.sin(pangle)])
      q = p + qrad*np.array([np.cos(qangle),np.sin(qangle)])
      p = np.maximum(np.zeros(2),np.minimum(np.ones(2),p))
      q = np.maximum(np.zeros(2),np.minimum(np.ones(2),q))
      obstacles.append((p,q))
    self.obstacles = np.array(obstacles)

  def _is_collision(self,next_pos):
    p,q = self.obstacles[:,0,:],self.obstacles[:,1,:]
    a,b = next_pos, self.pos
    d1 = (p-a)*(q-a)[:,[1,0]]
    d2 = (p-b)*(q-b)[:,[1,0]]
    d3 = (a-p)*(b-p)[:,[1,0]]
    d4 = (a-q)*(b-q)[:,[1,0]]
    if np.any(np.bitwise_and(np.sign(d1[:,0]-d1[:,1]) != np.sign(d2[:,0]-d2[:,1]),
            np.sign(d3[:,0]-d3[:,1]) != np.sign(d4[:,0]-d4[:,1]) )):
        return True
    return False

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
      return [self.env.action_space.sample()]
    comp = self.env.goal - self.env.pos
    return [((np.arctan2(comp[1],comp[0])+(2*np.pi))%(2*np.pi),\
      min(self.env.MAX_VEL,norm(comp)),False)]


if __name__ == '__main__':
  env = VelocityControl()
  env.render()
  agent = NaiveAgent(env)

  for j in range(5):
    action = agent.predict()[0]
    for i in range(10):
      obs, r, done, debug = env.step(action)
      action = agent.predict(obs)[0]
      env.render("test")
      if done:
        break

    env.reset()
  