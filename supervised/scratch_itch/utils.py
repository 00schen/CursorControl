from scipy.io import loadmat,savemat,whosmat
import argparse
import torch
import numpy as np
import gym
import os
from CursorControl import velNaiveAgent
from stable_baselines import SAC

parser = argparse.ArgumentParser(description='Goal Estimation')
parser.add_argument('--data', type=str, default='velocitycontrol-v0',
                    help='environment with no obstacles')
args = parser.parse_args()


def data_generator(dataset):
    os.makedirs("data", exist_ok=True)

    if dataset == 'velocitycontrol-v0':
        env = gym.make('velocitycontrol-v0',**{'oracle':None})
        agent = velNaiveAgent(env)
    elif dataset == 'velocitycontrol-v1':
        agent = SAC.load('../../logs/sac_pretrain_2d_obstacle/rl_model_1000000_steps')
        env = gym.make('velocitycontrol-v1',**{'oracle': agent})

    data = []
    for i in range(100000):
        action = agent.predict()
        done = False
        episode = [[*env.goal,*([0]*(env.observation_space.shape[0]+env.action_space.shape[0]-4))]]
        while not done:
            obs, r, done, debug = env.step(action)
            action = agent.predict(obs)[0]
            stored_action = env.noise.get_noise(action)

            episode.append([*obs[:3],*stored_action])
        data.append(episode)
        env.reset()

        print("{} episodes done.".format(i))

    data = np.array(data)
    idx = np.random.choice(range(len(data)),int(.2*len(data)),replace=False)
    X_train = np.delete(data, idx, axis=0)
    X_valid = data[idx[:len(idx)//2]]
    X_test = data[idx[len(idx)//2:]]
    f = "data/"+dataset+"_data.mat"
    savemat(f, {'train':X_train,'validation':X_valid,'test':X_test})
    data = []
            
    
def data_loader(dataset):
    dataset = loadmat("data/"+dataset+"_data.mat")
    X_train = dataset['train']
    X_valid = dataset['validation']
    X_test = dataset['test']
    print(X_test[0][10])
    for data in [X_train, X_valid, X_test]:
        for i in range(len(data)):
            data[i] = torch.Tensor(data[i].astype(np.float64))

    return X_train, X_valid, X_test

if __name__ == '__main__':
    data_generator(args.data)