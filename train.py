import numpy as np
import gym

import tensorflow as tf

from spinup import sac_tf1, td3_tf1
from spinup.utils.run_utils import ExperimentGrid
import gym_pull

gym_pull.pull('github.com/00schen/CursorControl')

ENV_NAME = '00schen/CursorControl'

eg = ExperimentGrid(name='sac-tf1-bench')
eg.add('env_name', ENV_NAME, '',True)
eg.add('epochs', 10)
eg.add('steps_per_epoch', 4000)
eg.add('ac_kwargs:hidden_sizes', [(32,), (64,64)], 'hid')
eg.add('ac_kwargs:activation', [tf.tanh, tf.nn.relu], '')
eg.run(sac_tf1, num_cpu=2)