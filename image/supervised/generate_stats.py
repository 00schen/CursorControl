import numpy as np
import numpy.random as random

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--env',help='environment directory name in data folder')
args, _ = parser.parse_known_args()

data = np.load(os.path.join('data',args.env,'t.noised_target.npz'))
X,Y,C = data.values()

obs_mean,obs_std = np.mean(X,axis=(0,1)),np.std(X,axis=(0,1))
obs_stats = np.vstack((obs_mean,obs_std))

target_mean,target_std = np.mean(Y,axis=0),np.std(Y,axis=0)
target_stats = np.vstack((target_mean,target_std))

np.savez_compressed(os.path.join('data',args.env,'stats1'),obs_stats=obs_stats,target_stats=target_stats)