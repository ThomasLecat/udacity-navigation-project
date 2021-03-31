from unityagents import UnityEnvironment

from navigation.agent import ExtendedDQN
from navigation.environment import SingleAgentEnvWrapper
from navigation.preprocessors import IdentityPreprocessor
from navigation.replay_buffer import UniformReplayBuffer
from navigation.scheduler import LinearScheduler, Milestone


def main():
    preprocessor = IdentityPreprocessor()
    env = UnityEnvironment("Banana.app", no_graphics=True)
    env = SingleAgentEnvWrapper(env, preprocessor, skip_frames=4)
    replay_buffer = UniformReplayBuffer(buffer_size=1_000_000)
    epsilon_scheduler = LinearScheduler(
        [
            Milestone(step=0, value=1.0),
            Milestone(step=1_000_000, value=0.1),
        ]
    )
    agent = ExtendedDQN(
        env=env,
        replay_buffer=replay_buffer,
        batch_size=32,
        epsilon_scheduler=epsilon_scheduler,
        discount_factor=0.99,
        learning_rate=0.0001,
        learning_start=50_000,
        target_network_update_coefficient=0.001,
        clip_td_errors=True,
        update_frequency=1,
    )
    agent.train(num_episodes=10000)


if __name__ == "__main__":
    main()
