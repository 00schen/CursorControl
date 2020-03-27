from .scratch_itch_robots import ScratchItchJacoEnv
import numpy as np
import pybullet as p
from gym import spaces,make
import numpy.random as random
from numpy.linalg import norm
from copy import deepcopy

import torch

from a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs
from a2c_ppo_acktr.utils import get_render_func, get_vec_normalize


model_path = "trained_models/ppo/ScratchItchJaco-v0.pt"

class ScratchItchJacoOracleEnv(ScratchItchJacoEnv):
    N=3
    ORACLE_DIM = 16
    ORACLE_NOISE = 0.0
    def __init__(self):
        actor_critic, ob_rms = torch.load(model_path)

        dummy_env = make('ScratchItchJaco-v0')
        self.action_space = dummy_env.action_space
        self.observation_space = dummy_env.observation_space

        env = make_vec_envs('ScratchItchJaco-v0', 1001, 1, None, None,
                    False, device='cpu', allow_early_resets=False)
        vec_norm = get_vec_normalize(env)
        if vec_norm is not None:
            vec_norm.eval()
            vec_norm.ob_rms = ob_rms
        self.env = env

        self.oracle = PretrainOracle(actor_critic)

        render_func = get_render_func(env)
        self.render = lambda: render_func('human') if render_func is not None else None

    def step(self,action):
        obs,r,done,info = self.env.step(action)
        done,info = done[0],info[0]
        self.tool_pos,self.torso_pos,self.real_step = info['tool_pos'],info['torso_pos'],info['real_step']

        opt_act = self.oracle.predict(obs,done,False)
        click = norm(obs[7:10]) < 0.025
        obs_2d = [*(self.tool_pos[:2]-self.org_tool_pos[:2]),self.click,\
            *self._noiser(self._real_action_to_2D(opt_act)),click]
        self.click = click
        self.buffer_2d.pop()
        self.buffer_2d.insert(0,obs_2d)
        info.update({'opt_act':opt_act})
        return (obs,np.array(self.buffer_2d).flatten()),r,done,info

    def reset(self):
        obs = self.env.reset()
        actor_critic, _ob_rms = torch.load(model_path)
        self.oracle = PretrainOracle(actor_critic)

        self.buffer_2d = [[0]*(self.ORACLE_DIM+4)]*self.N
        self.click = False

        self._noiser = self._make_noising()
        self.unnoised_opt_act = np.zeros(3)
        obs,_reward,_done,info = self.env.step(self.oracle.predict(obs,False,True))
        info = info[0]
        self.tool_pos,self.torso_pos,self.real_step = info['tool_pos'],info['torso_pos'],info['real_step']
        self.id = info['id']
        self.org_tool_pos = deepcopy(self.tool_pos)

        return obs

    def vel_to_target_rel(self, vel):
        vel_coord = [np.cos(vel[0])*vel[1],np.sin(vel[0])*vel[1]]
        pred_target = self.tool_pos + np.array(vel_coord+[self.unnoised_opt_act[2]])
        return self.tool_pos-pred_target, pred_target-self.torso_pos

    def normalize(self,obs):
        """ Use env.step as a proxy to normalize obs """
        self.real_step[0] = False
        obs,_r,_done,_info = self.env.step(obs)
        self.real_step[0] = True
        return obs

    def _real_action_to_2D(self,sim_action):
        """take a real action by oracle and convert it into a 2d action"""
        realID = p.saveState()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=self.id)
        _obs,_r,_done,info = self.env.step(sim_action)
        info = info[0]
        new_tool_pos = info['tool_pos']
        p.restoreState(realID)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)
        self.unnoised_opt_act = new_tool_pos - self.tool_pos
        return self.unnoised_opt_act[:2]

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

class PretrainAgent():
    def __init__(self,model):
        self.model = model

        self.recurrent_hidden_states = torch.zeros(1, self.model.recurrent_hidden_state_size)
        self.masks = torch.zeros(1, 1)

    def predict(self,obs,done):
        self.masks.fill_(0.0 if done else 1.0)
        with torch.no_grad():
            value, action, _, self.recurrent_hidden_states = self.model.act(
                obs, self.recurrent_hidden_states, self.masks, deterministic=True)
        
        return action

class PretrainOracle():
    def __init__(self,model):
        self.model = model

        self.recurrent_hidden_states = torch.zeros(1, self.model.recurrent_hidden_state_size)
        self.masks = torch.zeros(1, 1)

    def predict(self,obs,done,real):
        self.masks.fill_(0.0 if done else 1.0)
        # obs = torch.tensor([obs],dtype=torch.float)
        with torch.no_grad():
            value, action, _, recurrent_hidden_states = self.model.act(
                obs, self.recurrent_hidden_states, self.masks, deterministic=True)
        if real:
            self.recurrent_hidden_states = recurrent_hidden_states
        return action

class TwoDAgent():
    def __init__(self,env,pretrain,predictor):
        self.env = env
        self.pretrain = pretrain
        self.predictor = predictor       

    def predict(self,obs,real_step=None,done=False):
        if len(obs) == 1:
            return self.pretrain.predict(obs,done)
        if real_step is not None:
            self.env.real_step = real_step
        obs,obs_2d = obs

        ### Directly using pretrained action ###
        # return self.pretrain.predict(obs,done)
        # return real_step

        ### Using in-the-loop set up ###
        vel,_states = self.predictor.predict(obs_2d)
        # obs_unnorm = np.concatenate((obs[0,:7],*self.env.vel_to_target_rel(vel),obs[0,13:]))
        # obs_norm = self.env.normalize(torch.tensor(obs_unnorm.reshape((1,-1)),dtype=torch.float))
        # predicted_obs = np.concatenate((obs[0,:7],obs_norm[0,7:13],obs[0,13:]))
        predicted_obs = np.concatenate((obs[0,:7],*self.env.vel_to_target_rel(vel),obs[0,13:]))
        action = self.pretrain.predict(torch.tensor([predicted_obs],dtype=torch.float),done)        
        return action

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