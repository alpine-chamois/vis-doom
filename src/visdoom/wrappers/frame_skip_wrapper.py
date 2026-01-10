from gymnasium import Env, Wrapper
import numpy as np


class FrameSkipWrapper(Wrapper):
    """
    Wrapper to implement frame skipping without affecting rendering.
    """

    def __init__(self, env: Env, skip: int) -> None:
        super().__init__(env)
        self.skip = skip

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Repeat action, sum reward, and render all intermediate frames.
        """
        total_reward: float = 0.0

        for _ in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward  # type: ignore

            # Always render intermediate frames
            self.env.render()

            if terminated or truncated:
                break

        return obs, total_reward, terminated, truncated, info
