from gym.envs.registration import register
import gym

register(id='cursorcontrol-v1',
        entry_point='cursorcontrolv1.envs:CursorControl',
        )