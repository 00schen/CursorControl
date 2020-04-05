from gym.envs.registration import register
import gym

register(id='velocitycontrol-v0',
        entry_point='CursorControl.envs:VelocityControl',
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