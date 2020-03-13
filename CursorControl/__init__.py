from gym.envs.registration import register
import gym

register(id='cursorcontrol-v0',
        entry_point='CursorControl.envs:CursorControl',
        )
register(id='cursorcontrol-v1',
        entry_point='CursorControl.envs:CursorControl1',
        )
register(id='cursorcontrol-v2',
        entry_point='CursorControl.envs:CursorControl2',
        )