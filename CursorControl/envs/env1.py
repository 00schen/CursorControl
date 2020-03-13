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


class CursorControl(gym.Env):
  MAX_EP_LEN = 30
  GOAL_THRESH = .05
  MAX_VEL = .1

  ORACLE_NOISE = .1
  ORACLE_DIM = 2

  GAMMA = .9

  def __init__(self):
    self.observation_space = spaces.Box(np.array([0]*7+[-np.inf]*self.ORACLE_DIM+[0]),np.array([1]*7+[np.inf]*self.ORACLE_DIM+[1]))
    self.action_space = spaces.Box(np.zeros(3),np.array([2*np.pi,self.MAX_VEL,1]))

    self.init = .5*np.ones(2)
    
    self.reset()

    self.do_render = False # will automatically set to true the first time .render is called

  def set_max_ep_len(self, max_ep_len):
    self.MAX_EP_LEN = max_ep_len
  def set_oracle_noise(self, oracle_noise):
    self.ORACLE_NOISE = oracle_noise
  def set_gamma(self, gamma):
    self.GAMMA = gamma

  def step(self, action):
    vel, click = action[:2],np.rint(action[2])
    vel = np.minimum([2*np.pi, self.MAX_VEL], np.maximum([0,0], vel))

    self.pos += vel[1]*np.array([np.cos(vel[0]),np.sin(vel[0])])
    self.pos = np.minimum(np.ones(2), np.maximum(np.zeros(2), self.pos))
    self.click = click

    self.opt_act = self.optimal_user_policy(self.pos)    

    goal_dist = norm(self.pos-self.goal)
    self.succ = goal_dist <= self.GOAL_THRESH and self.click
      
    obs = np.array((*self.pos, *self.pos_buffer[0], *self.pos_buffer[1], self.click, *self.opt_act))
    
    self.pos_buffer.append(self.pos)
    self.pos_buffer.pop(0)
    self.prev_action = action

    r = 30*self.succ/(1-self.GAMMA) + (1/goal_dist/50 if goal_dist > self.GOAL_THRESH else 1) - 10*(self.click and not self.succ)

    self.curr_step += 1
    done = self.curr_step >= self.MAX_EP_LEN or self.succ

    info = {
      'goal': self.goal, 'is_success': self.succ,
      'pos': self.pos, 'vel': vel,
      'opt_action': self.opt_act, 'step': self.curr_step
    }

    return obs, r, done, info

  def reset(self):
    # print("reset called")
    self._set_goal()

    self.pos = copy(self.init) # position
    self.click = False
    self.prev_action = (0,0,0)
    self.curr_step = 0 # timestep in current episode
    
    self.succ = 0 #True - most recent episode ended in success
    self.pos_buffer = [np.zeros(2),np.zeros(2)]
    return np.concatenate((self.init,np.zeros(4+self.ORACLE_DIM+2)))

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
    self.optimal_user_policy = self.make_oracle_policy(goal)
    self.opt_act = np.zeros(3)

  # simulate user with optimal intended actions that go directly to the goal
  def make_oracle_policy(self, goal):
    noise = random.normal(np.vstack((np.identity(2),np.zeros((self.ORACLE_DIM-2,2)))),self.ORACLE_NOISE)
    def add_noise(action):
      return np.array((*(noise@action[:2]),action[2] != (random.random() < .1))) # flip click with p = .1

    def policy(pos):
      comp = goal-pos
      vel = ((np.arctan2(comp[1],comp[0])+(2*np.pi))%(2*np.pi), min(self.MAX_VEL,norm(comp)))
      return add_noise((*vel,norm(comp)<= self.GOAL_THRESH))
      # return (*vel,norm(comp)<=self.GOAL_THRESH)

    return policy

class naiveAgent(CursorControl):
  def predict(self,obs=None,r=None):
    if r == None:
      return np.array([2*np.pi,self.MAX_VEL,1])*random.random(3)
    return (*obs[7:9],obs[-1])

if __name__ == '__main__':
  pygame.init()
  env = CursorControl()
  env.render()
  agent = naiveAgent()
  env.set_oracle_noise(0)

  action = agent.predict()
  for i in range(100):
    obs, r, done, debug = env.step(action)
    action = agent.predict(obs,r)
    env.render("test")
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



