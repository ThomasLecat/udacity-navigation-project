from typing import Tuple

import numpy as np
from unityagents import UnityEnvironment

from navigation.preprocessors import PreprocessorInterface


class SingleAgentEnvWrapper:
    """Wrap the Unity environment into a format similar to single agents gym
    environments.

    Call the preprocessor on observations before sending them back to the agent.
    """

    def __init__(self, env: UnityEnvironment, preprocessor: PreprocessorInterface):
        self.env = env
        self.preprocessor = preprocessor
        self.brain_name = env.brain_names[0]
        self.brain = env.brains[self.brain_name]

    def reset(self) -> np.ndarray:
        env_info = self.env.reset()[self.brain_name]
        return self.preprocessor.transform(env_info.vector_observations)

    @property
    def num_actions(self) -> int:
        return self.brain.vector_action_space_size

    @property
    def obs_size(self) -> int:
        return self.preprocessor.observation_size(
            raw_obs_size=self.brain.vector_observation_space_size
        )

    def step(self, action: int) -> Tuple:
        env_info = self.env.step(action)[self.brain_name]
        assert (
            len(env_info.rewards)
            == len(env_info.local_done)
            == env_info.vector_observations.shape[0]
            == 1
        ), "More than one agent found for this environment."
        return (
            env_info.vector_observations[0],
            env_info.rewards[0],
            env_info.local_done[0],
            None,
        )
