from gym.envs.registration import register
import gym
from CursorControl.envs import VelNaiveAgent

register(id='velocitycontrol-v0',
        entry_point='CursorControl.envs:VelocityControl',
        )
register(id='velocitycontrol-v1',
        entry_point='CursorControl.envs:VelocityControl1',
        )
register(id='velocitycontrol-v2',
        entry_point='CursorControl.envs:VelocityControl2',
        )
register(id='goalcontrol-v0',
        entry_point='CursorControl.envs:GoalControl',
        )
register(id='goalcontrol-v1',
        entry_point='CursorControl.envs:GoalControl1',
        )
register(id='goalcontrol-v2',
        entry_point='CursorControl.envs:GoalControl2',
        )