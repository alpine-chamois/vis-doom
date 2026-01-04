from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """
    Class to encapsulate configuration and hyperparameters for agents.
    """

    # Scenario-specific
    training_steps: int
    learning_rate: float
    n_steps: int
    reward_scale: float
    ent_coef: float
    eval_freq: int
    num_demo_episodes: int

    # Common
    num_envs: int = 4
    screen_width: int = 80
    screen_height: int = 60
    frame_skip: int = 4
    frame_stack: int = 4
    verbose: int = 1
    num_eval_episodes: int = 10
