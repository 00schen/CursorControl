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

# class ScratchItchJacoOracleEnv(ScratchItchJacoEnv):
#     """
#     List of obs processing:
#         rebase tool position against original
#         normalize predictions
#     """

#     N=3
#     ORACLE_DIM = 16
#     ORACLE_NOISE = 0.0
#     def __init__(self):
#         dummy_env = make('ScratchItchJaco-v0')
#         self.action_space = dummy_env.action_space
#         self.observation_space = dummy_env.observation_space

#         actor_critic, ob_rms = torch.load(model_path)

#         env = make_vec_envs('ScratchItchJaco-v0', random.randint(100), 1, None, None,
#                     False, device='cpu', allow_early_resets=False)
#         vec_norm = get_vec_normalize(env)
#         if vec_norm is not None:
#             vec_norm.eval()
#             vec_norm.ob_rms = ob_rms
#         self.env = env

#         render_func = get_render_func(env)
#         self.render = lambda: render_func('human') if render_func is not None else None

#         self.oracle = PretrainOracle(actor_critic)


#     def step(self,action):
#         obs,r,done,info = self.env.step(action)
#         done,info = done[0],info[0]
#         self.tool_pos,self.torso_pos,self.real_step = info['tool_pos'],info['torso_pos'],info['real_step']

#         opt_act = self.oracle.predict(obs,done,False)
#         click = norm(obs[7:10]) < 0.025
#         obs_2d = [*(self.tool_pos[:2]-self.org_tool_pos[:2]),self.click,\
#             *self._noiser(self._real_action_to_2D(opt_act)),click]
#         self.click = click
#         self.buffer_2d.pop()
#         self.buffer_2d.insert(0,obs_2d)
#         info.update({'opt_act':opt_act})
#         return (obs,np.array(self.buffer_2d).flatten()),r,done,info

#     def reset(self):
#         obs = self.env.reset()

#         self.buffer_2d = [[0]*(self.ORACLE_DIM+4)]*self.N
#         self.click = False

#         self._noiser = self._make_noising()
#         self.unnoised_opt_act = np.zeros(3)
#         obs,_reward,_done,info = self.env.step(self.oracle.predict(obs,False,True))
#         info = info[0]
#         self.tool_pos,self.torso_pos,self.real_step = info['tool_pos'],info['torso_pos'],info['real_step']
#         self.id = info['id']
#         self.org_tool_pos = deepcopy(self.tool_pos)

#         return obs

#     def vel_to_target_rel(self, vel):
#         vel_coord = [np.cos(vel[0])*vel[1],np.sin(vel[0])*vel[1]]
#         pred_target = self.tool_pos + np.array(vel_coord+[self.unnoised_opt_act[2]])
#         return self.tool_pos-pred_target, pred_target-self.torso_pos

#     def normalize(self,obs):
#         """ Use env.step as a proxy to normalize obs """
#         self.real_step[0] = False
#         obs,_r,_done,_info = self.env.step(obs)
#         self.real_step[0] = True
#         return obs

#     def _real_action_to_2D(self,sim_action):
#         """take a real action by oracle and convert it into a 2d action"""
#         realID = p.saveState()
#         p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=self.id)
#         _obs,_r,_done,info = self.env.step(sim_action)
#         info = info[0]
#         new_tool_pos = info['tool_pos']
#         p.restoreState(realID)
#         p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)
#         self.unnoised_opt_act = new_tool_pos - self.tool_pos
#         return self.unnoised_opt_act[:2]

#     def _make_noising(self):
#         # simulate user with optimal intended actions that go directly to the goal
#         projection = np.vstack((np.identity(2),np.zeros((self.ORACLE_DIM-2,2))))
#         noise = random.normal(np.identity(self.ORACLE_DIM),self.ORACLE_NOISE)
#         lag_buffer = []

#         def add_noise(action):
#             return np.array(noise@action) # flip click with p = .1

#         def add_dropout(action):
#             return action if random.random() > .1\
#                 else np.concatenate((self.action_space.sample(),np.random.random(self.ORACLE_DIM-2)))

#         def add_lag(action):
#             lag_buffer.append(action)
#             return lag_buffer.pop(0) if random.random() > .1 else lag_buffer[0]
        
#         def noiser(action):
#             return add_noise(projection@action)

#         return noiser

# class PretrainAgent():
#     def __init__(self,model):
#         self.model = model

#         self.recurrent_hidden_states = torch.zeros(1, self.model.recurrent_hidden_state_size)
#         self.masks = torch.zeros(1, 1)

#     def predict(self,obs,done):
#         self.masks.fill_(0.0 if done else 1.0)
#         with torch.no_grad():
#             value, action, _, self.recurrent_hidden_states = self.model.act(
#                 obs, self.recurrent_hidden_states, self.masks, deterministic=True)
        
#         return action

# class PretrainOracle():
#     def __init__(self,model):
#         self.model = model

#         self.recurrent_hidden_states = torch.zeros(1, self.model.recurrent_hidden_state_size)
#         self.masks = torch.zeros(1, 1)

#     def predict(self,obs,done,real):
#         self.masks.fill_(0.0 if done else 1.0)
#         # obs = torch.tensor([obs],dtype=torch.float)
#         with torch.no_grad():
#             value, action, _, recurrent_hidden_states = self.model.act(
#                 obs, self.recurrent_hidden_states, self.masks, deterministic=True)
#         if real:
#             self.recurrent_hidden_states = recurrent_hidden_states
#         return action

# class TwoDAgent():
#     def __init__(self,env,pretrain,predictor):
#         self.env = env
#         self.pretrain = pretrain
#         self.predictor = predictor       

#     def predict(self,obs,opt_act=None,done=False):
#         if len(obs) == 1:
#             return self.pretrain.predict(obs,done)
        
#         obs,obs_2d = obs

#         ### Directly using pretrained action ###
#         # return self.pretrain.predict(obs,done)
#         # return opt_act

#         ### Using in-the-loop set up ###
#         vel,_states = self.predictor.predict(obs_2d)
#         # obs_unnorm = np.concatenate((obs[0,:7],*self.env.vel_to_target_rel(vel),obs[0,13:]))
#         # obs_norm = self.env.normalize(torch.tensor(obs_unnorm.reshape((1,-1)),dtype=torch.float))
#         # predicted_obs = np.concatenate((obs[0,:7],obs_norm[0,7:13],obs[0,13:]))
#         predicted_obs = np.concatenate((obs[0,:7],*self.env.vel_to_target_rel(vel),obs[0,13:]))
#         action = self.pretrain.predict(torch.tensor([predicted_obs],dtype=torch.float),done)        
#         return action

class ScratchItchJacoOracleEnv(ScratchItchJacoEnv):
    """
    List of obs processing:
        rebase tool position against original
        normalize predictions
    """

    N=3
    ORACLE_DIM = 16
    ORACLE_NOISE = 0.0
    
    scale = .1

    def step(self, action):
        """ Use step as a proxy to format and normalize predicted observation """
        if self.real_step[0]:
            return self._step(action)
        self.real_step = [True]
        return np.concatenate((action[:7],*self.goal_to_target_rel(action[-2:]),action[13:-2])),0,False,{}


    def _step(self,action):
        obs,r,done,info = super().step(action)
        self._update_pos()

        click = norm(obs[7:10]) < 0.025
        obs_2d = [*((self.tool_pos[:2]-self.org_tool_pos[:2])/self.scale+np.array([.5,.5])),self.click,\
            *self._noiser((self.target_pos[:2]-self.org_tool_pos[:2])/self.scale+np.array([.5,.5])),click]
        self.click = click
        self.buffer_2d.pop()
        self.buffer_2d.insert(0,obs_2d)
        info.update({"obs_2d":np.array(self.buffer_2d).flatten(), "unnorm_obs":obs})
        self.real_step = [False]

        return obs,r,done,info

    def reset(self):
        obs = super().reset()

        self.buffer_2d = [[0]*(self.ORACLE_DIM+4)]*self.N
        self.click = False

        self._noiser = self._make_noising()

        self.real_step = [True]

        self._update_pos()
        self.org_tool_pos = deepcopy(self.tool_pos)

        self.pred_visual = [-1]*10

        sphere_collision = -1
        sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.01, rgbaColor=[10, 255, 10, 1], physicsClientId=self.id)
        p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=sphere_collision, baseVisualShapeIndex=sphere_visual,\
             basePosition=self.org_tool_pos, useMaximalCoordinates=False, physicsClientId=self.id)

        return obs

    def goal_to_target_rel(self, pred_target_2d):
        pred_target = np.array([*((pred_target_2d-np.array([.5,.5]))*self.scale+self.org_tool_pos[:2]),self.target_pos[2]])

        sphere_collision = -1
        sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.01, rgbaColor=[250, 110, 0, 1], physicsClientId=self.id)
        pred_visual = self.pred_visual.pop()
        if pred_visual != -1:
            p.removeBody(pred_visual)
        new_pred_visual = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=sphere_collision, baseVisualShapeIndex=sphere_visual,\
             basePosition=pred_target, useMaximalCoordinates=False, physicsClientId=self.id)
        self.pred_visual.insert(0,new_pred_visual)

        return self.tool_pos-pred_target, pred_target-self.torso_pos

    def _make_noising(self):
        # simulate user with optimal intended actions that go directly to the goal
        projection = np.vstack((np.identity(2),np.zeros((self.ORACLE_DIM-2,2))))
        noise = random.normal(np.identity(self.ORACLE_DIM),self.ORACLE_NOISE)
        lag_buffer = []

        def add_noise(action):
            return np.array(noise@action) # flip click with p = .1

        def add_dropout(action):
            return action if random.random() > .1\
                else np.concatenate((self.action_space.sample(),np.random.random(self.ORACLE_DIM-2)))

        def add_lag(action):
            lag_buffer.append(action)
            return lag_buffer.pop(0) if random.random() > .1 else lag_buffer[0]
        
        def noiser(action):
            return projection@action

        return noiser

    def _update_pos(self):
        self.torso_pos = np.array(p.getLinkState(self.robot, 0, computeForwardKinematics=True, physicsClientId=self.id)[0])
        self.tool_pos = np.array(p.getLinkState(self.tool, 1, computeForwardKinematics=True, physicsClientId=self.id)[0])


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

class TwoDAgent():
    buffer_length = 50
    success_length = 5
    def __init__(self,env,pretrain,predictor):
        self.env = env
        self.pretrain = pretrain
        self.predictor = predictor
        self.prediction_buffer = [] 
        self.curr_prediction = []  
        self.succeeded = []   

    def predict(self,obs,info=None,opt_act=None,done=False):
        if info == None:
            return self.pretrain.predict(obs,done)
        
        ### Directly using pretrained action ###
        # return self.pretrain.predict(obs,done)

        ### Using in-the-loop set up ###
        obs_2d, obs = info['obs_2d'], info['unnorm_obs']
        pred_goal = self.predictor.predict(obs_2d)[0]

        if len(self.prediction_buffer) == 0:
            self.prediction_buffer = np.array([pred_goal[:2]]*10)
        else:
            self.prediction_buffer = np.concatenate(([pred_goal[:2]],self.prediction_buffer),axis=0)
            self.prediction_buffer = np.delete(self.prediction_buffer,-1,0)
        mean_pred = np.mean(self.prediction_buffer,axis=0)

        # if info['task_success'] > len(self.succeeded) and len(self.succeeded) < self.success_length:
        #     self.succeeded.append(self.curr_prediction)
        #     self.curr_prediction = np.mean(self.succeeded,axis=0)
        # if len(self.curr_prediction) == 0 or len(self.succeeded) < self.success_length:
        #     self.curr_prediction = np.mean(self.succeeded+[mean_pred],axis=0)     

        self.curr_prediction = mean_pred

        pred_obs_rel = np.concatenate((obs,self.curr_prediction))
        norm_obs,_r,_done,_info = self.env.step(torch.tensor([pred_obs_rel],dtype=torch.float))
        action = self.pretrain.predict(norm_obs,done)    
        # print(obs.dot(norm_obs.flatten())/norm(norm_obs)/norm(obs)) 
        return action
