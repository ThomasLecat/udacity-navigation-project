import argparse
import time

import torch
from unityagents import UnityEnvironment

from navigation.agent import ExtendedDQN
from navigation.config import DQNConfig
from navigation.environment import SingleAgentEnvWrapper
from navigation.preprocessors import IdentityPreprocessor, PreprocessorInterface


def evaluate(environment_path: str, checkpoint_path: str, show_graphics: bool) -> None:
    """Play one episode with the specified checkpoint of the trained DQN agent."""
    preprocessor: PreprocessorInterface = IdentityPreprocessor()
    env = UnityEnvironment(environment_path, no_graphics=not show_graphics)
    env = SingleAgentEnvWrapper(env, preprocessor, skip_frames=DQNConfig.SKIP_FRAMES)
    agent = ExtendedDQN(
        env=env, config=DQNConfig, replay_buffer=None, epsilon_scheduler=None
    )
    # Load saved model
    agent.q_network = torch.load(checkpoint_path)

    # Play episode
    episode_reward: float = 0.0
    episode_length: int = 0
    observation = env.reset()
    while True:
        if show_graphics:
            time.sleep(0.05)
        action = agent.compute_action(observation, epsilon=0.0)
        observation, reward, done, _ = agent.env.step(action)
        episode_reward += reward
        episode_length += 1
        if done is True:
            break
    print(
        f"Episode finished in {episode_length} steps, episode reward: {episode_reward}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--environment_path",
        "-p",
        type=str,
        help="Path to your single agent Unity environment file.",
    )
    parser.add_argument(
        "--checkpoint_path", "-c", type=str, help="Path to the PyTorch checkpoint"
    )
    parser.add_argument(
        "--show_graphics",
        "-s",
        type=bool,
        default=True,
        help="Visualize the agent playing on the environment",
    )
    args = parser.parse_args()
    evaluate(args.environment_path, args.checkpoint_path, args.show_graphics)
