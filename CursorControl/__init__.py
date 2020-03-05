from gym.envs.registration import register
import gym

register(id='cursorcontrol-v1',
        entry_point='CursorControl.envs:CursorControl',
        )