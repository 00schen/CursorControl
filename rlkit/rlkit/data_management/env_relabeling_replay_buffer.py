from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
import random
import numpy as np
import copy


class EnvRelabelingReplayBuffer(EnvReplayBuffer):
    def __init__(
            self,
            max_replay_buffer_size,
            env,
            env_info_sizes=None,
            k=4,
            reward_fn=lambda state, goal: -int(np.linalg.norm(state - goal) > 0.01),
            terminal_fn=lambda state, goal: np.linalg.norm(state - goal) <= 0.01,
            state_key='tool_pos'
    ):
        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size * (1 + k),
            env=env,
            env_info_sizes=env_info_sizes,
        )
        self.k = k
        self.input_dim = self.env.oracle.size
        self.reward_fn = reward_fn
        self.terminal_fn = terminal_fn
        self.state_key = state_key

    def add_path(self, path):
        """
        Add a path to the replay buffer.

        This default implementation naively goes through every step, but you
        may want to optimize this.

        NOTE: You should NOT call "terminate_episode" after calling add_path.
        It's assumed that this function handles the episode termination.

        :param path: Dict like one outputted by rlkit.samplers.util.rollout
        """
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
                future_info = random.choice(path["env_infos"][i:])  # infos associated with next observations
                future_goal = future_info[self.state_key]
                mod_obs = copy.deepcopy(obs)
                mod_obs[-self.input_dim:] = copy.deepcopy(future_goal)
                mod_next_obs = copy.deepcopy(next_obs)
                next_state = env_info[self.state_key]
                mod_reward = self.reward_fn(next_state, future_goal)
                mod_next_obs[-self.input_dim:] = copy.deepcopy(future_goal)
                mod_terminal = self.terminal_fn(next_state, future_goal)
                self.add_sample(observation=mod_obs,
                                action=action,
                                reward=mod_reward,
                                next_observation=mod_next_obs,
                                terminal=mod_terminal,
                                agent_info=agent_info,
                                env_info=env_info)

        self.terminate_episode()

