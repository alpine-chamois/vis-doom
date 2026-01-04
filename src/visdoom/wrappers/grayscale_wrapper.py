from gymnasium import Env, ObservationWrapper
from gymnasium.spaces import Box
import cv2
import numpy as np


class GrayscaleWrapper(ObservationWrapper):
    """
    Wrapper to convert RGB observations to grayscale.
    """

    def __init__(self, env: Env) -> None:
        super().__init__(env)

        height, width, _ = env.observation_space.shape  # type: ignore

        # Update observation space
        self.observation_space = Box(
            low=0,
            high=255,
            shape=(height, width, 1),
            dtype=np.uint8,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """
        Convert RGB observations to grayscale.
        """
        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        observation = np.expand_dims(observation, axis=-1)  # Add a channel dimension
        return observation
