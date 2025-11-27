from gymnasium import Env, ObservationWrapper
import numpy as np

OBS_KEY = "screen"


class ScreenWrapper(ObservationWrapper):
    """
    Wrapper to extract the screen from the observation dictionary.
    """

    def __init__(self, env: Env) -> None:
        super().__init__(env)
        # Update onservation space
        self.observation_space = env.observation_space[OBS_KEY]  # type: ignore

    def observation(self, observation: dict) -> np.ndarray:
        """
        Extract the screen from the observation dictionary.
        """
        return observation[OBS_KEY]
