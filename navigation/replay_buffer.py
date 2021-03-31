import abc
from collections import deque, namedtuple

import numpy as np


SampleBatch = namedtuple(
    "SampleBatch", ["observations", "actions", "rewards", "dones", "next_observations"]
)
TorchSampleBatch = namedtuple(
    "TorchSampleBatch", ["observations", "actions", "rewards", "dones", "next_observations"]
)


class ReplayBufferInterface:
    def __init__(self, buffer_size: int):
        self.buffer = deque([], maxlen=buffer_size)
        self.buffer_indexes = np.arange(buffer_size)

    def add(self, observation, action, reward, next_obs, done):
        """Add one transition to the replay buffer"""

    def sample(self, num_samples: int):
        sampled_indexes = np.random.choice(
            self.buffer_indexes[: len(self.buffer)],
            size=num_samples,
            p=self.probabilities(),
        )
        samples = [self.buffer[idx] for idx in sampled_indexes]
        observations = np.stack([s.observation for s in samples], axis=0)
        actions = np.array([s.action for s in samples], dtype=np.int)
        rewards = np.array([s.reward for s in samples], dtype=np.float32)
        dones = np.array([s.done for s in samples], dtype=np.int)
        next_observations = np.stack([s.next_observation for s in samples], axis=0)

        return SampleBatch(observations, actions, rewards, dones, next_observations)

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
        self.Transition = namedtuple(
            "Transition",
            ["observation", "action", "reward", "done", "next_observation"],
        )

    def add(self, observation, action, reward, done, next_obs):
        self.buffer.append(self.transition(observation, action, reward, done, next_obs))

    @property
    def transition(self) -> namedtuple:
        return self.Transition

    def probabilities(self):
        return np.full(fill_value=1 / len(self.buffer), shape=len(self.buffer))
