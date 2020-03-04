from __future__ import division
from copy import deepcopy as copy
from collections import namedtuple

import gym
from gym import spaces
import numpy as np
from numpy.linalg import norm
from numpy import random

from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

WIDTH = 1000
GOAL_THRESH = 1
ORACLE_DIM = 16
MAX_EP_LEN = 50


class CursorControl(gym.Env):
  def __init__(
      self,
      max_ep_len=MAX_EP_LEN # max number of timesteps
      ):

    self.observation_space = spaces.Box(np.zeros(ORACLE_DIM+4),np.array([WIDTH]*2+[1]+[]))
    self.action_space = spaces.Box(np.zeros(3),np.array([2*np.pi,MAX_VEL,1]))

    self.max_ep_len = max_ep_len

    self.init = WIDTH/2*np.ones(2)
    self._set_goal()

    self.pos = copy(self.init) # position
    self.click = False
    self.curr_step = 0 # timestep in current episode
    self.viewer = None
    self.succ = 0 #True - most recent episode ended in success

  def _set_goal(self):
    goal = random.uniform(0,WIDTH,2)
    self.goal = goal
    self.optimal_user_policy = make_oracle_policy(goal)

  def step(self, action):
    vel, click = action[:2],np.rint(action[2])
    opt_act = self.optimal_user_policy(self.pos)

    self.pos += vel
    self.pos = np.minimum(np.ones(2), np.maximum(np.zeros(2), self.pos))
    goal_dist = norm(self.pos-self.goal)
    
    obs = (*goal_dist, self.click, *opt_act)
    self.click = click

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

  def render(self):
    pass
    # if self.viewer is None:
    #   self.viewer = rendering.SimpleImageViewer()

    # fig = plt.figure()
    # canvas = FigureCanvas(fig)

    # size = 100

    # plt.scatter(
    #   [self.goal[0]], [self.goal[1]],
    #   color='green',
    #   linewidth=0, alpha=0.75, marker='o', s=size*5
    # )

    # plt.scatter(
    #   [self.pos[0]], [self.pos[1]],
    #   color='orange',
    #   linewidth=0, alpha=0.75, s=size
    # )

    # plt.xlim([-0.05, 1.05])
    # plt.ylim([-0.05, 1.05])
    # plt.axis('off')

    # agg = canvas.switch_backends(FigureCanvas)
    # agg.draw()
    # width, height = fig.get_size_inches() * fig.get_dpi()
    # self.viewer.imshow(
    #   np.fromstring(agg.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3))
    # plt.close()


# simulate user with optimal intended actions that go directly to the goal
def make_oracle_policy(goal,noise_sd=1):
  def add_noise(action):
    noise = random.normal(np.vstack((np.identity(2),np.zeros((ORACLE_DIM-2,2)))),noise_sd)
    return (*(noise@action[:2]),action[2] != (random.random() < .1))

  def policy(pos):
    dist = norm(goal-pos)
    return add_noise(*((goal-pos)/dist*MAX_VEL),dist<=GOAL_THRESH)
  return policy

