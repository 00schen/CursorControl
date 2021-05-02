import copy
import random
import numpy as np
from .modded_buffer import ModdedReplayBuffer


class HERReplayBuffer(ModdedReplayBuffer):
    def __init__(
            self,
            max_replay_buffer_size,
            env,
            # env_info_sizes={'episode_success':1, 'target1_reached': 1},
            env_info_sizes=None,
            sample_base=5000 * 200,
            k=4,
            reward_fn=lambda state, goal: -int(np.linalg.norm(state - goal) > 0.01),
            terminal_fn=lambda state, goal: np.linalg.norm(state - goal) <= 0.01,
    ):
        # env_info_sizes.update({'episode_success':1, 'target1_reached': 1})
        # env_info_sizes.update({'episode_success': 1, })
        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size * (1 + k),
            env=env,
            env_info_sizes=env_info_sizes,
            sample_base=sample_base * (1 + k)
        )
        self.k = k
        self.reward_fn = reward_fn
        self.terminal_fn = terminal_fn

    def add_path(self, path):
        for i, (
                obs,
                action,
                reward,
                next_obs,
                terminal,
                agent_info,
                env_info
        ) in enumerate(zip(
            path["observations"],
            path["actions"],
            path["rewards"],
            path["next_observations"],
            path["terminals"],
            path["agent_infos"],
            path["env_infos"],
        )):
            self.add_sample(
                observation=obs,
                action=action,
                reward=reward,
                next_observation=next_obs,
                terminal=terminal,
                agent_info=agent_info,
                env_info=env_info,
            )

            for j in range(self.k):
                future_index = np.random.randint(low=i, high=len(path['rewards']))
                future_goal = path['next_observations'][future_index]['hindsight_goal']

                mod_obs = copy.deepcopy(obs)
                mod_obs['goal'] = copy.deepcopy(future_goal)

                mod_next_obs = copy.deepcopy(next_obs)
                mod_next_obs['goal'] = copy.deepcopy(future_goal)

                hindsight_goal = next_obs['hindsight_goal']
                mod_reward = self.reward_fn(hindsight_goal, future_goal)
                mod_terminal = self.terminal_fn(hindsight_goal, future_goal)

                self.add_sample(observation=mod_obs,
                                action=action,
                                reward=mod_reward,
                                next_observation=mod_next_obs,
                                terminal=mod_terminal,
                                agent_info=agent_info,
                                env_info=env_info)

        self.terminate_episode()
