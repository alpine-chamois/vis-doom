from gymnasium import Env, ObservationWrapper
from gymnasium.spaces import Box
import cv2
import numpy as np


class ResizeWrapper(ObservationWrapper):
    """
    Wrapper to resize the observation to the specified width and height.
    """

    def __init__(self, env: Env, width: int, height: int) -> None:
        super().__init__(env)

        self.width = width
        self.height = height
        _, _, channels = env.observation_space.shape  # type: ignore

        # Update observation space
        self.observation_space = Box(
            low=0,
            high=255,
            shape=(self.height, self.width, channels),  # type: ignore
            dtype=np.uint8,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """
        Resize the observation to the specified width and height.
        """
        observation = cv2.resize(observation, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return observation
