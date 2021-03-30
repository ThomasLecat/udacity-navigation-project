from navigation.agent import ExtendedDQN
from navigation.environment import EnvWrapper
from navigation.preprocessors import IdentityPreprocessor
from navigation.replay_buffer import UniformReplayBuffer
from navigation.scheduler import LinearScheduler, milestone
from unityagents import UnityEnvironment


def main():
    preprocessor = IdentityPreprocessor()
    env = UnityEnvironment("Banana.app")
    env = EnvWrapper(env, preprocessor)
    replay_buffer = UniformReplayBuffer(buffer_size=1_000_000)
    epsilon_scheduler = LinearScheduler(
        [
            milestone(step=0, value=1.0),
            milestone(step=1_000_000, value=0.1),
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
