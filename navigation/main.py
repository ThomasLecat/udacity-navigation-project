from navigation.agent import ExtendedDQN
from navigation.environment import EnvWrapper
from navigation.preprocessors import IdentityPreprocessor
from navigation.replay_buffer import UniformReplayBuffer
from unityagents import UnityEnvironment


def main():
    preprocessor = IdentityPreprocessor()
    env = UnityEnvironment("Banana.app")
    env = EnvWrapper(env, preprocessor)
    replay_buffer = UniformReplayBuffer(buffer_size=1_000_000)
    agent = ExtendedDQN(
        env=env,
        replay_buffer=replay_buffer,
        batch_size=32,
        epsilon=0.1,
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
