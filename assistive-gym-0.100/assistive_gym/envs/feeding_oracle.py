from .feeding_robots import FeedingJacoEnv
import numpy as np
import pybullet as p
from gym import spaces,make
import numpy.random as random
from numpy.linalg import norm
from copy import deepcopy

import torch


model_path = "trained_models/ppo/FeedingJaco-v0.pt"

class FeedingJacoOracleEnv():
    """
    List of obs processing:
        rebase spoon position against original
        normalize predictions
    """

    N=3
    ORACLE_DIM = 16
    ORACLE_NOISE = 0.0
    def __init__(self):
        dummy_env = make('FeedingJaco-v0')
        self.action_space = dummy_env.action_space
        self.observation_space = dummy_env.observation_space

        actor_critic, ob_rms = torch.load(model_path)

        env = make_vec_envs('FeedingJaco-v0', random.randint(100), 1, None, None,
                    False, device='cpu', allow_early_resets=False)
        vec_norm = get_vec_normalize(env)
        if vec_norm is not None:
            vec_norm.eval()
            vec_norm.ob_rms = ob_rms
        self.env = env

        render_func = get_render_func(env)
        self.render = lambda: render_func('human') if render_func is not None else None

        self.oracle = PretrainOracle(actor_critic)


    def step(self,action):
        obs,r,done,info = self.env.step(action)
        done,info = done[0],info[0]
        self.spoon_pos,self.torso_pos = info['spoon_pos'],info['torso_pos']

        opt_act = self.oracle.predict(obs,done,False)
        click = norm(obs[7:10]) < 0.025
        obs_2d = [*(self.spoon_pos[:2]-self.org_spoon_pos[:2]),self.click,\
            *self._noiser(self._real_action_to_2D(opt_act)),click]
        self.click = click
        self.buffer_2d.pop()
        self.buffer_2d.insert(0,obs_2d)
        info.update({'opt_act':opt_act})
        return (obs,np.array(self.buffer_2d).flatten()),r,done,info

    def reset(self):
        obs = self.env.reset()

        self.buffer_2d = [[0]*(self.ORACLE_DIM+4)]*self.N
        self.click = False

        self._noiser = self._make_noising()
        self.unnoised_opt_act = np.zeros(3)
        obs,_reward,_done,info = self.env.step(self.oracle.predict(obs,False,True))
        info = info[0]
        self.spoon_pos,self.torso_pos = info['spoon_pos'],info['torso_pos']
        self.id = info['id']
        self.org_spoon_pos = deepcopy(self.spoon_pos)

        return obs

    def vel_to_target_rel(self, vel):
        vel_coord = [np.cos(vel[0])*vel[1],np.sin(vel[0])*vel[1]]
        pred_target = self.spoon_pos + np.array(vel_coord+[self.unnoised_opt_act[2]])
        return self.spoon_pos-pred_target

    def _real_action_to_2D(self,sim_action):
        """take a real action by oracle and convert it into a 2d action"""
        realID = p.saveState()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=self.id)
        _obs,_r,_done,info = self.env.step(sim_action)
        info = info[0]
        new_spoon_pos = info['spoon_pos']
        p.restoreState(realID)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)
        self.unnoised_opt_act = new_spoon_pos - self.spoon_pos
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

class FeedingPretrainAgent():
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

class FeedingTwoDAgent():
    def __init__(self,env,pretrain,predictor):
        self.env = env
        self.pretrain = pretrain
        self.predictor = predictor       

    def predict(self,obs,opt_act=None,done=False):
        if len(obs) == 1:
            return self.pretrain.predict(obs,done)
        
        obs,obs_2d = obs

        ### Directly using pretrained action ###
        # return self.pretrain.predict(obs,done)
        # return opt_act

        ### Using in-the-loop set up ###
        vel,_states = self.predictor.predict(obs_2d)
        predicted_obs = np.concatenate((obs[0,:7],self.env.vel_to_target_rel(vel),obs[0,10:]))
        action = self.pretrain.predict(torch.tensor([predicted_obs],dtype=torch.float),done)        
        return action
