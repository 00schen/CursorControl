from gym.envs.registration import register
import gym

if 'cursorcontrol-v1' in gym.registry.env_specs:
        del gym.registry.env_specs['cursorcontrol-v1']

register(id='cursorcontrol-v1',
        entry_point='cursorcontrolv1.envs:CursorControl',
        )