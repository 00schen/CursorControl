import argparse
import numpy as np
import gym
import os
from CursorControl import VelNaiveAgent
from stable_baselines import SAC

parser = argparse.ArgumentParser(description='Goal Estimation')
parser.add_argument('--data', type=str, default='velocitycontrol-v0',
                    help='environment with no obstacles')
args = parser.parse_args()


def data_generator(dataset):
    os.makedirs("data", exist_ok=True)

    if dataset == 'velocitycontrol-v0':
        env = gym.make('velocitycontrol-v0',**{'rollout': 1})
        agent = VelNaiveAgent(env)
    elif dataset == 'velocitycontrol-v1':
        agent = SAC.load('../logs/sac_pretrain_2d_obstacle/rl_model_1000000_steps')
        env = gym.make('velocitycontrol-v1', **{'rollout':1})

    data = []
    for i in range(int(1e5)):
        obs = env.reset()
        action = agent.predict(obs)[0]
        done = False
        episode = []
        while not done:
            obs, r, done, debug = env.step(action)
            action = agent.predict(obs)[0]
            stored_action = env.noise.get_noise(action)

            episode.append([*obs[:3],*stored_action])
        episode.append([*env.goal,*([0]*(env.observation_space.shape[0]+env.action_space.shape[0]-4))])
        data.append(episode)
        print("{} episodes done.".format(i + 1))

    np.save("data/"+dataset+"_data", data)
            
    
def data_loader(dataset):
    dataset = np.load("data/"+dataset+"_data.npy",allow_pickle=True)
    idx = np.random.choice(range(len(dataset)),int(.2*len(dataset)),replace=False)
    X_train = np.delete(dataset, idx, axis=0)
    X_valid = dataset[idx]

    return X_train, X_valid

if __name__ == '__main__':
    data_generator(args.data)