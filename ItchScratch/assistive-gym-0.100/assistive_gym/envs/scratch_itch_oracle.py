from .scratch_itch_robots import ScratchItchJacoEnv
import numpy as np
import pybullet as p
from gym import spaces

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class ScratchItchJacoOracleEnv(ScratchItchJacoEnv):
    def __init__(self,oracle):
        super().__init__()
        observation_size = self.observation_space.shape[0] -2 + self.action_space.shape[0]
        self.observation_space = spaces.Box(low=np.array([-1.0]*observation_size,high=np.array([1.0]*observation_size)))
        self.oracle = oracle

    def step(self,action):
        obs,r,done,info = super().step(action)
        opt_act = oracle.predict(action)
        oracled_obs = np.concatenate((obs[:2],obs[4:],opt_act))
        return oracled_obs,r,done,info

    def reset(self):
        return None

class PretrainOracle():
    def __init__(self,model_path):
        self.model = torch.load(model_path)
        self.model.eval()

    def predict(self,obs):
        return self.model(obs)

class PredictingAgent():
    def __init__(self,env,model_path):
        self.pretrain = PretrainOracle(model_path)
        self.env = env(self.pretrain)
        self.predictor = MLP(len(self.env.observation_space),2*len(self.env.target_pos))
    
    def get_action(self,obs):
        if obs == None:
            return self.env.action_space.sample()
        predicted_target_rel = self.predictor.forward(obs)
        predicted_obs = np.concatenate((obs[:2],predicted_target_rel,obs[2:-len(self.env.action_space)]))
        action = self.pretrain.predict(predicted_obs)
        return action, predicted_target_rel

class OnPolicyTrainer():
    def __init__(self,model,env,agent,gamma,batch_size=500):
        self.model = model
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model, lr=gamma)

    def train(self,timesteps):
        action = self.agent.get_action()
        for step in range(timesteps):               
            optimizer.zero_grad()
            obs,_ = self.env.step(action)
            action, outputs = self.agent.get_action(obs)
            loss = self.criterion(outputs, obs[2:4])
            loss.backward()
            self.optimizer.step()
            
            if (step+1) % 100 == 0:
                print ('Timestep [%d/%d]'%(step+1, timesteps))

class OffPolicyTrainer():
    def __init__(self,model,gamma,batch_size=500):
        self.model = model
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model, lr=gamma)

        self.batch_size = batch_size
        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

    def train(self,timesteps):
        for step in range(timesteps,step=self.batch_size):
            for i, (obs, labels) in enumerate(self.train_loader):                  
                optimizer.zero_grad()
                outputs = self.model(obs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                if (i+1) % 100 == 0:
                    print ('Timestep [%d/%d]'%(step+1, timesteps))

class MLP(nn.Module):
    hidden_size = 32
    def __init__(self, obs_size, goal_size):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(obs_size,self.hidden_size)
        self.l2 = nn.Linear(self.hidden_size,self.hidden_size)
        self.l3 = nn.Linear(self.hidden_size,self.hidden_size)
        self.l4 = nn.Linear(self.hidden_size,self.hidden_size)
        self.l5 = nn.Linear(self.hidden_size,goal_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return F.relu(self.l5(x))