from collections import namedtuple
from typing import ClassVar, List

import numpy as np
import torch

from navigation.config import DQNConfig
from navigation.environment import SingleAgentEnvWrapper
from navigation.model import MultilayerPerceptron
from navigation.replay_buffer import (
    ReplayBufferInterface,
    SampleBatch,
    TorchSampleBatch,
)
from navigation.scheduler import SchedulerInterface
from navigation.utils import OneHot, convert_to_torch


class ExtendedDQN:
    def __init__(
        self,
        env: SingleAgentEnvWrapper,
        replay_buffer: ReplayBufferInterface,
        epsilon_scheduler: SchedulerInterface,
        config: ClassVar[DQNConfig],
    ):
        self.env: SingleAgentEnvWrapper = env
        self.replay_buffer: ReplayBufferInterface = replay_buffer
        self.epsilon_scheduler: SchedulerInterface = epsilon_scheduler
        self.config: ClassVar[DQNConfig] = config

        # Use GPU if available
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # Create networks
        obs_size: int = env.obs_size
        self.num_actions: int = env.num_actions
        self.q_network = MultilayerPerceptron(
            input_size=obs_size, hidden_layers=[64, 64], output_size=self.num_actions
        ).to(self.device)
        self.target_q_network = MultilayerPerceptron(
            input_size=obs_size, hidden_layers=[64, 64], output_size=self.num_actions
        ).to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())

        # Optimizer
        self.optimizer = torch.optim.Adam(
            params=self.q_network.parameters(), lr=config.LEARNING_RATE
        )

        self.one_hot = OneHot(
            batch_size=config.BATCH_SIZE,
            num_digits=self.num_actions,
            device=self.device,
        )

    def compute_action(self, observation: np.ndarray, epsilon: float) -> int:
        if np.random.uniform(0, 1) > epsilon:
            with torch.no_grad():
                observation = torch.Tensor(observation).to(self.device)
                # Create fake batch dimension of one
                # (obs_size) -> (1, obs_size)
                observation = torch.unsqueeze(observation, dim=0)
                # (1, num_actions)
                q_values = self.q_network(observation)
                # (1, num_actions) -> (num_actions)
                q_values = q_values.squeeze()
            return q_values.argmax().item()
        return np.random.choice(self.num_actions)

    def train(self, num_episodes: int) -> List:
        """Train the agent for 'num_episodes' and return the list of rewards
        per episode (summed over all steps of the episode).
        """
        reward_per_episode: List[float] = []
        num_steps_sampled: int = 0

        for episode_idx in range(1, num_episodes + 1):
            # Log progress
            if episode_idx % self.config.LOG_EVERY == 0:
                window_rewards = reward_per_episode[self.config.LOG_EVERY :]
                print(
                    f"episode {episode_idx}/{num_episodes}, "
                    f"avg. episode reward: {sum(window_rewards) / len(window_rewards)}, "
                    f"num steps sampled: {num_steps_sampled}"
                )

            # Sample one episode
            observation = self.env.reset()
            episode_length: int = 0
            episode_reward: float = 0.0
            while True:
                epsilon = self.epsilon_scheduler.get_value(num_steps_sampled)
                action = self.compute_action(observation, epsilon)
                next_obs, reward, done, _ = self.env.step(action)
                self.replay_buffer.add(observation, action, reward, done, next_obs)
                observation = next_obs
                episode_length += 1
                episode_reward += reward
                if (
                    episode_length % self.config.UPDATE_EVERY == 0
                    and num_steps_sampled > self.config.LEARNING_STARTS
                ):
                    self.update_once()
                if done is True:
                    break
            reward_per_episode.append(episode_reward)
            num_steps_sampled += episode_length
        return reward_per_episode

    def update_once(self) -> None:
        """Perform one SGD update."""
        self.optimizer.zero_grad()
        sample_batch: namedtuple = self.replay_buffer.sample(self.config.BATCH_SIZE)
        loss = self.compute_loss(sample_batch)
        loss.backward()
        self.optimizer.step()
        # Soft update of target Q network
        self.soft_update_target_network()

    def compute_loss(self, sample_batch: SampleBatch) -> torch.Tensor:
        # Compute TD targets
        sample_batch: TorchSampleBatch = convert_to_torch(sample_batch, self.device)
        # (batch_size, num_actions)
        q_target_tp1 = self.target_q_network(sample_batch.next_observations)
        # (batch_size)
        td_targets = sample_batch.rewards + self.config.DISCOUNT * torch.max(
            q_target_tp1, dim=1
        )[0] * (1 - sample_batch.dones)
        td_targets.detach()
        # Compute TD errors
        # (batch_size, num_actions)
        q_values = self.q_network(sample_batch.observations)
        one_hot_actions = self.one_hot(sample_batch.actions)
        # (batch_size)
        selected_q_values = torch.sum(q_values * one_hot_actions, dim=1)
        td_errors = selected_q_values - td_targets
        if self.config.CLIP_TD_ERROR:
            td_errors = torch.clamp(td_errors, -1, 1)
        return torch.sum(td_errors ** 2)

    def soft_update_target_network(self):
        target_state_dict = self.target_q_network.state_dict()
        for param_name, param_tensor in self.q_network.state_dict().items():
            target_state_dict[param_name] = (
                (1 - self.config.TARGET_UPDATE_COEFF) * target_state_dict[param_name]
                + self.config.TARGET_UPDATE_COEFF * param_tensor
            )
        self.target_q_network.load_state_dict(target_state_dict)
