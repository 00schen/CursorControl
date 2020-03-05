from gym.envs.registration import register
import gym

if 'cursorcontrol-v1' in gym.envs.registry.env_specs:
        del gym.envs.registry.env_specs['cursorcontrol-v1']

register(id='cursorcontrol-v1',
        entry_point='cursorcontrolv1.envs:CursorControl',
        )