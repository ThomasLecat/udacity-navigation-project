from navigation.types import NumberOfSteps


class DQNConfig:
    # Sampling
    EPSILON_START: float = 1.0
    EPSILON_END: float = 0.1
    EPS_DECAY_START: NumberOfSteps = 0
    EPS_DECAY_END: NumberOfSteps = 30000
    SKIP_FRAMES: int = 1

    # Optimisation
    BUFFER_SIZE: int = 100_000
    BATCH_SIZE: int = 64
    DISCOUNT: float = 0.99
    LEARNING_RATE: float = 0.0005
    LEARNING_STARTS: NumberOfSteps = 1000
    TARGET_UPDATE_COEFF: float = 0.001
    CLIP_TD_ERROR: bool = False
    UPDATE_EVERY: NumberOfSteps = 4

    # DQN Extensions
    DOUBLE_Q: bool = False

    # Logging
    LOG_EVERY: NumberOfSteps = 100

    def __setattr__(self, key, value):
        raise AttributeError("Config objets are immutable")
