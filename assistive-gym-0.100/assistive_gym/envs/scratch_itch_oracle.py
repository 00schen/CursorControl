from .scratch_itch_robots import ScratchItchJacoEnv
import numpy as np
import pybullet as p
from gym import spaces
import numpy.random as random

import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F

from numpy.linalg import norm
model_path = "trained_models/ppo/ScratchItchJaco-v0.pt"

class ScratchItchJacoOracleEnv(ScratchItchJacoEnv):
    N=3
    ORACLE_DIM = 16
    ORACLE_NOISE = 0.0

    def step(self,action):
        obs,r,done,info = super(ScratchItchJacoOracleEnv,self).step(action)

        opt_act = self.oracle.predict(obs,done)
        tool_pos = self._get_tool_pos()
        obs_2d = [*(tool_pos[:2]-self.org_tool_pos[:2]),norm(tool_pos-self.target_pos) < 0.025,\
            *self._noiser(self._real_action_to_2D(opt_act)),norm(tool_pos-self.target_pos) < 0.025]
        self.buffer_2d.pop()
        self.buffer_2d.insert(0,obs_2d)

        return obs,r,done,{"obs_2d":np.array(self.buffer_2d).flatten(),"target_func":self.vel_to_target_rel}

    def reset(self):
        obs = super().reset()
        actor_critic, _ob_rms = torch.load(model_path)
        self.oracle = PretrainOracle1(actor_critic)

        tool_pos = self._get_tool_pos()
        obs_2d = [*tool_pos[:2],norm(tool_pos-self.target_pos) < 0.025]\
            +[0]*self.ORACLE_DIM+[norm(tool_pos-self.target_pos) < 0.025]
        self.buffer_2d = [obs_2d]*self.N

        self._noiser = self._make_noising()

        self.unnoised_opt_act = np.zeros(3)
        self.org_tool_pos = self._get_tool_pos()

        return obs

    def vel_to_target_rel(self, vel):
        torso_pos = np.array(p.getLinkState(self.robot, 15 if self.robot_type == 'pr2' else 0, computeForwardKinematics=True, physicsClientId=self.id)[0])
        tool_pos = self._get_tool_pos()
        vel_coord = [np.cos(vel[0])*vel[1],np.sin(vel[0])*vel[1]]
        pred_target = tool_pos + np.array(vel_coord+[self.unnoised_opt_act[2]])
        return tool_pos-pred_target, pred_target-torso_pos

    def _real_action_to_2D(self,real_action):
        """take a real action by oracle and convert it into a 2d action"""
        org_tool_pos = self._get_tool_pos()
        realID = p.saveState()
        super().step(real_action)
        new_tool_pos = self._get_tool_pos()
        p.restoreState(realID)
        self.unnoised_opt_act = new_tool_pos - org_tool_pos
        return (new_tool_pos - org_tool_pos)[:2]

    def _make_noising(self):
        # simulate user with optimal intended actions that go directly to the goal
        projection = np.vstack((np.identity(2),np.zeros((self.ORACLE_DIM-2,2))))
        noise = random.normal(np.identity(self.ORACLE_DIM),self.ORACLE_NOISE)
        lag_buffer = []

        def add_noise(action):
            return np.array((*(noise@action[:2]),action[2] != (random.random() < .1))) # flip click with p = .1

        def add_dropout(action):
            return action if random.random() > .1\
                else np.concatenate((self.action_space.sample(),np.random.random(self.ORACLE_DIM-2)))

        def add_lag(action):
            lag_buffer.append(action)
            return lag_buffer.pop(0) if random.random() > .1 else lag_buffer[0]
        
        def noiser(action):
            return projection@action

        return noiser

    def _get_tool_pos(self):
        state = p.getLinkState(self.tool, 1, computeForwardKinematics=True, physicsClientId=self.id)
        tool_pos = np.array(state[0])
        return tool_pos

class PretrainOracle():
    def __init__(self,model):
        self.model = model

        self.recurrent_hidden_states = torch.zeros(1, self.model.recurrent_hidden_state_size)
        self.masks = torch.zeros(1, 1)

    def predict(self,obs,done):
        self.masks.fill_(0.0 if done else 1.0)
        # obs = torch.tensor(obs,dtype=torch.float)
        with torch.no_grad():
            value, action, _, self.recurrent_hidden_states = self.model.act(
                obs, self.recurrent_hidden_states, self.masks, deterministic=True)
        return action

class PretrainOracle1():
    def __init__(self,model):
        self.model = model

        self.recurrent_hidden_states = torch.zeros(1, self.model.recurrent_hidden_state_size)
        self.masks = torch.zeros(1, 1)

    def predict(self,obs,done):
        self.masks.fill_(0.0 if done else 1.0)
        obs = torch.tensor(obs,dtype=torch.float)
        with torch.no_grad():
            value, action, _, self.recurrent_hidden_states = self.model.act(
                obs, self.recurrent_hidden_states, self.masks, deterministic=True)
        return action.numpy()[0,0,0,:]

class TwoDAgent():
    def __init__(self,env,pretrain,predictor):
        self.env = env
        self.pretrain = pretrain
        self.predictor = predictor       

    def predict(self,obs,obs_2d=None,target_func=None,done=False):
        if target_func == None:
            action = self.pretrain.predict(obs,done)
            return action

        vel,_states = self.predictor.predict(obs_2d)
        predicted_obs = np.concatenate((obs[0,:7],*target_func(vel),obs[0,13:]))
        action = self.pretrain.predict(torch.tensor(predicted_obs,dtype=torch.float),done)        
        return action[0,0,:]

# class PredictingAgent():
#     def __init__(self,env,model_path):
#         self.pretrain = PretrainOracle(model_path)
#         self.env = env(self.pretrain)
#         self.predictor = MLP(len(self.env.observation_space),2*len(self.env.target_pos))
    
#     def get_action(self,obs):
#         if obs == None:
#             return self.env.action_space.sample()
#         predicted_target_rel = self.predictor.forward(obs)
#         predicted_obs = np.concatenate((obs[:2],predicted_target_rel,obs[2:-len(self.env.action_space)]))
#         action = self.pretrain.predict(predicted_obs)
#         return action, predicted_target_rel

# class OnPolicyTrainer():
#     def __init__(self,model,env,agent,gamma,batch_size=500):
#         self.model = model
#         self.criterion = nn.MSELoss()
#         self.optimizer = torch.optim.Adam(model, lr=gamma)

#     def train(self,timesteps):
#         action = self.agent.get_action()
#         for step in range(timesteps):               
#             optimizer.zero_grad()
#             obs,_ = self.env.step(action)
#             action, outputs = self.agent.get_action(obs)
#             loss = self.criterion(outputs, obs[2:4])
#             loss.backward()
#             self.optimizer.step()
            
#             if (step+1) % 100 == 0:
#                 print ('Timestep [%d/%d]'%(step+1, timesteps))

# class OffPolicyTrainer():
#     def __init__(self,model,gamma,batch_size=500):
#         self.model = model
#         self.criterion = nn.MSELoss()
#         self.optimizer = torch.optim.Adam(model, lr=gamma)

#         self.batch_size = batch_size
#         self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
#                                            batch_size=batch_size, 
#                                            shuffle=True)

#     def train(self,timesteps):
#         for step in range(timesteps,step=self.batch_size):
#             for i, (obs, labels) in enumerate(self.train_loader):                  
#                 optimizer.zero_grad()
#                 outputs = self.model(obs)
#                 loss = self.criterion(outputs, labels)
#                 loss.backward()
#                 self.optimizer.step()
                
#                 if (i+1) % 100 == 0:
#                     print ('Timestep [%d/%d]'%(step+1, timesteps))

# class MLP(nn.Module):
#     hidden_size = 32
#     def __init__(self, obs_size, goal_size):
#         super(MLP, self).__init__()
#         self.l1 = nn.Linear(obs_size,self.hidden_size)
#         self.l2 = nn.Linear(self.hidden_size,self.hidden_size)
#         self.l3 = nn.Linear(self.hidden_size,self.hidden_size)
#         self.l4 = nn.Linear(self.hidden_size,self.hidden_size)
#         self.l5 = nn.Linear(self.hidden_size,goal_size)

#     def forward(self, x):
#         x = F.relu(self.l1(x))
#         x = F.relu(self.l2(x))
#         x = F.relu(self.l3(x))
#         x = F.relu(self.l4(x))
#         return F.relu(self.l5(x))