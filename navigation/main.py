import argparse

import torch
from unityagents import UnityEnvironment

from navigation.agent import ExtendedDQN
from navigation.config import DQNConfig
from navigation.environment import SingleAgentEnvWrapper
from navigation.preprocessors import IdentityPreprocessor
from navigation.replay_buffer import UniformReplayBuffer
from navigation.scheduler import LinearScheduler, Milestone


def main(environment_path: str):
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
        replay_buffer=replay_buffer,
        epsilon_scheduler=epsilon_scheduler,
        config=config,
    )
    agent.train(num_episodes=500)
    with open("dqn_checkpoint.pt", "w") as f:
        torch.save(agent.q_network, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--environment_path",
        "-p",
        type=str,
        help="Path to your single agent Unity environment file.",
    )
    args = parser.parse_args()
    main(args.environment_path)
