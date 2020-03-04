from gym.envs.registration import register

register(id='cursorcontrol-v1',
        entry_point='cursorcontrolv1.envs:env',
        )