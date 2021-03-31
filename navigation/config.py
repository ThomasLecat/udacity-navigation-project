from navigation.types import NumberOfSteps


class DQNConfig:
    # Sampling
    EPSILON_START: float = 1.0
    EPSILON_END: float = 0.1
    DECAY_START: NumberOfSteps = 0
    DECAY_END: NumberOfSteps = 50000
    SKIP_FRAMES: int = 4

    # Optimisation
    BUFFER_SIZE: int = 1_000_000
    BATCH_SIZE: int = 32
    DISCOUNT: float = 0.99
    LEARNING_RATE: float = 0.0005
    LEARNING_STARTS: NumberOfSteps = 10000
    TARGET_UPDATE_COEFF: float = 0.001
    CLIP_TD_ERROR: bool = True
    UPDATE_EVERY: NumberOfSteps = 1

    # DQN Extensions
    DOUBLE_Q: bool = False

    # Logging
    LOG_EVERY: NumberOfSteps = 100

    def __setattr__(self, key, value):
        raise AttributeError("Config objets are immutable")
