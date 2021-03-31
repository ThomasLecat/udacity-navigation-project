from collections import namedtuple

import numpy as np
import torch
from navigation.environment import SingleAgentEnvWrapper
from navigation.model import MultilayerPerceptron
from navigation.replay_buffer import ReplayBufferInterface
from navigation.scheduler import SchedulerInterface


NumberOfSteps = int


class ExtendedDQN:
    def __init__(
        self,
        env: SingleAgentEnvWrapper,
        replay_buffer: ReplayBufferInterface,
        batch_size: int,
        epsilon_scheduler: SchedulerInterface,
        discount_factor: float,
        learning_rate: float,
        learning_start: NumberOfSteps,
        target_network_update_coefficient: float,
        clip_td_errors: bool,
        update_frequency: NumberOfSteps = 1,
    ):
        # TODO: make config for hyperparameters.
        self.env: SingleAgentEnvWrapper = env
        self.replay_buffer: ReplayBufferInterface = replay_buffer
        self.batch_size: int = batch_size
        self.epsilon_scheduler: SchedulerInterface = epsilon_scheduler
        self.target_network_update_coefficient = target_network_update_coefficient
        self.discount_factor: float = discount_factor
        self.learning_start: NumberOfSteps = learning_start
        self.clip_td_errors: bool = clip_td_errors
        self.update_frequency: NumberOfSteps = update_frequency
        self.num_actions: int = env.num_actions

        # Use GPU if available
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # Create networks
        obs_size: int = env.obs_size
        self.q_network = MultilayerPerceptron(
            input_size=obs_size, hidden_layers=[64, 64], output_size=self.num_actions
        ).to(self.device)
        self.target_q_network = MultilayerPerceptron(
            input_size=obs_size, hidden_layers=[64, 64], output_size=self.num_actions
        ).to(self.device)
        self.target_q_network.parameters = self.q_network.parameters

        # Logging parameters
        self.log_frequency = 100  # in number of episodes

        # Optimizer
        self.optimizer = torch.optim.Adam(
            params=self.q_network.parameters(), lr=learning_rate
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

    def train(self, num_episodes):
        cumulative_reward: float = 0.0
        num_steps_sampled: int = 0
        for episode_idx in range(1, num_episodes + 1):
            if episode_idx % self.log_frequency == 0:
                print(
                    f"episode {episode_idx}/{num_episodes}, "
                    f"average episode reward: {cumulative_reward/self.log_frequency}, "
                    f"num steps sampled: {num_steps_sampled}"
                )
                cumulative_reward = 0.0
            observation = self.env.reset()
            episode_length: int = 0
            while True:
                epsilon = self.epsilon_scheduler.get_value(num_steps_sampled)
                action = self.compute_action(observation, epsilon)
                next_obs, reward, done, _ = self.env.step(action)
                self.replay_buffer.add(observation, action, reward, done, next_obs)
                cumulative_reward += reward
                observation = next_obs
                episode_length += 1
                if episode_length % self.update_frequency == 0 and num_steps_sampled > self.learning_start:
                    self.update_once()
                if done is True:
                    break
            num_steps_sampled += episode_length

    def update_once(self):
        """Perform one SGD update."""
        self.optimizer.zero_grad()
        sample_batch: namedtuple = self.replay_buffer.sample(self.batch_size)
        loss = self.compute_loss(sample_batch)
        loss.backward()
        self.optimizer.step()
        # Soft update of target Q network
        self.soft_update_target_network()

    def compute_loss(self, sample_batch):
        # Compute TD targets
        # (batch_size, obs_size)
        next_obs = torch.Tensor(sample_batch.next_observations).to(self.device)
        # (batch_size, num_actions)
        q_target_tp1 = self.target_q_network(next_obs)
        # (batch_size)
        td_targets = sample_batch.rewards + self.discount_factor * q_target_tp1.max(
            dim=1
        )[0]
        # TODO: Set TD targets to 0 when done
        # Compute TD errors
        # (batch_size, obs_size)
        observations = torch.Tensor(sample_batch.observations).to(self.device)
        # (batch_size)
        q_values = self.q_network(observations)
        td_errors = q_values - td_targets
        if self.clip_td_errors:
            td_errors = torch.clip(td_errors, -1, 1)
        return torch.sum(td_errors ** 2)

    def soft_update_target_network(self):
        target_state_dict = self.target_q_network.state_dict()
        for param_name, param_tensor in self.q_network.state_dict():
            target_state_dict[param_name] = (
                (1 - self.target_network_update_coefficient)
                * target_state_dict[param_name]
                + self.target_network_update_coefficient * param_tensor
            )
        self.target_q_network.load_state_dict(target_state_dict)
