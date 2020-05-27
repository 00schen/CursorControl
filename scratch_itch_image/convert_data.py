import numpy as np
import numpy.random as random

import os

from gym import spaces

from tqdm import tqdm

from utils import Noise

dirname = os.path.dirname(__file__)
file_path = os.path.join(dirname,'samples','ScratchItchJaco-v0')
filenames = [f'{1000+i}_{j}.npz' for j in range(1,5) for i in range(1,26)]


data = {'obs':[],'act':[],'target':[]}
save_count = 0
for file in tqdm(filenames):
	try:
		new_data = np.load(os.path.join(file_path,file))
	except:
		continue
	data['obs'].append(new_data['obs_data'])
	data['act'].append(new_data['act_data'])
	data['target'].append(new_data['targets'])
data['obs'] = np.concatenate(data['obs'])
data['act'] = np.concatenate(data['act'])
data['target'] = np.concatenate(data['target'])

# generating action
act = data['obs'][:,list(range(1,200))+[-1],:3] - data['obs'][:,:,:3]

print('Adding noise to actions')
act = act.transpose((1,0,2))

noise = Noise(spaces.Box(low=-1*np.ones(3),high=np.ones(3)),3,batch=act.shape[1])
for i in tqdm(range(len(act))):
	act[i] = noise(act[i])

act = np.array(act).transpose((1,0,2))

X = np.concatenate((data['obs'][...,:7],data['obs'][...,13:],act),axis=2)
np.savez_compressed(os.path.join(file_path,'t.noised_trajectory'),X=X,Y=data['target'])
