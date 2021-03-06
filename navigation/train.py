import argparse

import torch
from unityagents import UnityEnvironment

from navigation.agent import ExtendedDQN
from navigation.config import DQNConfig
from navigation.environment import SingleAgentEnvWrapper
from navigation.preprocessors import IdentityPreprocessor
from navigation.replay_buffer import UniformReplayBuffer
from navigation.scheduler import LinearScheduler, Milestone
from navigation.utils import write_list_to_csv


def train(environment_path: str, num_episodes: int):
    """Train the agent for 'num_episodes', save the score for each training episode
    and the checkpoint of the trained agent.
    """
    config = DQNConfig
    preprocessor = IdentityPreprocessor()
    env = UnityEnvironment(environment_path, no_graphics=True)
    env = SingleAgentEnvWrapper(env, preprocessor, skip_frames=config.SKIP_FRAMES)
    replay_buffer = UniformReplayBuffer(config.BUFFER_SIZE)
    epsilon_scheduler = LinearScheduler(
        [
            Milestone(config.EPS_DECAY_START, config.EPSILON_START),
            Milestone(config.EPS_DECAY_END, config.EPSILON_END),
        ]
    )
    agent = ExtendedDQN(
        env=env,
        config=config,
        replay_buffer=replay_buffer,
        epsilon_scheduler=epsilon_scheduler,
    )
    reward_per_episode = agent.train(num_episodes=num_episodes)
    with open("reward_per_episode.csv", "w") as f:
        write_list_to_csv(f, reward_per_episode)
    with open("dqn_checkpoint.pt", "wb") as f:
        torch.save(agent.q_network, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--environment_path",
        "-p",
        type=str,
        help="Path to your single agent Unity environment file.",
    )
    parser.add_argument(
        "--num_episodes",
        "-n",
        type=int,
        default=500,
        help="Number of episodes on which to train the agent",
    )
    args = parser.parse_args()
    train(args.environment_path, args.num_episodes)
