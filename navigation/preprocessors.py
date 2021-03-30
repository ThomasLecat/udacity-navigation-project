import abc
import numpy as np


class PreprocessorInterface:
    @abc.abstractmethod
    def transform(self, raw_observation: np.ndarray) -> np.ndarray:
        """Preprocess the raw observation into an array to feed to the NN"""

    @abc.abstractmethod
    def observation_size(self, raw_obs_size) -> int:
        """Return the size of the observation after preprocessing"""


class IdentityPreprocessor(PreprocessorInterface):
    """Identity preprocessor"""
    def transform(self, observation):
        return observation

    def observation_size(self, raw_obs_size) -> int:
        return raw_obs_size
