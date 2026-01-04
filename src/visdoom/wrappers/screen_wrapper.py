from gymnasium import Env, ObservationWrapper
import numpy as np
from visdoom.constants import SCREEN_KEY


class ScreenWrapper(ObservationWrapper):
    """
    Wrapper to extract the screen from the observation dictionary.
    """

    def __init__(self, env: Env) -> None:
        super().__init__(env)
        # Update observation space
        self.observation_space = env.observation_space[SCREEN_KEY]  # type: ignore

    def observation(self, observation: dict) -> np.ndarray:
        """
        Extract the screen from the observation dictionary.
        """
        return observation[SCREEN_KEY]
