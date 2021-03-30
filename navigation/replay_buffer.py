import abc
from collections import deque, namedtuple

import numpy as np


class ReplayBufferInterface:
    def __init__(self, buffer_size: int):
        self.buffer = deque([], maxlen=buffer_size)
        self.buffer_indexes = np.arange(buffer_size)
        self.sample_batch = namedtuple(
            "SampleBatch",
            ["observations", "actions", "rewards", "next_observations", "dones"],
        )

    def add(self, observation, action, reward, next_obs, done):
        """Add one transition to the replay buffer"""

    def sample(self, num_samples: int):
        sampled_indexes = np.random.choice(
            self.buffer_indexes, size=num_samples, p=self.probabilities()
        )
        samples = [self.buffer[idx] for idx in sampled_indexes]
        observations = np.stack([s["observation"] for s in samples], axis=0)
        actions = np.array([s["action"] for s in samples], dtype=np.int)
        rewards = np.array([s["reward"] for s in samples], dtype=np.float)
        dones = np.array([s["dones"] for s in samples], dtype=np.bool)
        next_observations = np.stack([s["next_observation"] for s in samples], axis=0)

        return self.sample_batch(
            observations, actions, rewards, next_observations, dones
        )

    @property
    @abc.abstractmethod
    def transition(self) -> namedtuple:
        """Define what data is recorded for each step"""

    @abc.abstractmethod
    def probabilities(self) -> np.ndarray:
        """Return the probability distribution over the samples in the buffer."""


class UniformReplayBuffer(ReplayBufferInterface):
    def __init__(self, buffer_size: int):
        super().__init__(buffer_size)
        self.transition_ = namedtuple(
            "Transition",
            ["observation", "action", "reward", "next_observation", "done"],
        )
        self._uniform_probabilities = np.full(fill_value=1 / buffer_size, shape=buffer_size)

    def add(self, observation, action, reward, next_obs, done):
        self.buffer.append(self.transition(observation, action, reward, next_obs, done))

    @property
    def transition(self) -> namedtuple:
        return self.transition_

    def probabilities(self):
        return self._uniform_probabilities
