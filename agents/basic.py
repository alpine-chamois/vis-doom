import argparse
from pathlib import Path
import cv2
import numpy as np
import gymnasium
from gymnasium.envs.registration import register
from stable_baselines3 import ppo
from stable_baselines3.common import callbacks
from stable_baselines3.common import policies
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    DummyVecEnv,
    VecTransposeImage,
    VecFrameStack,
)
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TransformReward

# Config
ENV = "MyVizdoomBasic-v0"
SCENARIO_DIR = Path(__file__).parent.parent / "scenarios"
CFG_FILE = SCENARIO_DIR / "basic.cfg"
LOG_NAME = "basic"
LOG_DIR = "logs/basic/"
MODEL_DIR = "models/basic/"
OBS_KEY = "screen"
MODEL_NAME = "best_model.zip"
RGB = "rgb_array"
HUMAN = "human"
TRAINING_STEPS = 25000
FRAME_SKIP = 4
FRAME_STACK = 4
NUM_ENVS = 4
NUM_EVAL_EPISODES = 10
EVAL_FREQ = 500
LEARNING_RATE = 1e-3
N_STEPS = 128
VERBOSE = 1
REWARD_SCALE = 0.01
SCREEN_WIDTH = 80
SCREEN_HEIGHT = 60
NUM_DEMO_EPISODES = 10

# Register custom VizDoom environment with local scenario file
register(
    id=ENV,
    entry_point="vizdoom.gymnasium_wrapper.gymnasium_env_defns:VizdoomScenarioEnv",
    kwargs={
        "scenario_file": CFG_FILE,
    },
)


# Environment wrapper to extract the screen from the observation
class ObservationWrapper(gymnasium.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Update onservation space
        self.observation_space = env.observation_space[OBS_KEY]  # type: ignore

    def observation(self, observation):
        return observation[OBS_KEY]


# Environment wrapper to resize the screen
class ResizeWrapper(gymnasium.ObservationWrapper):
    def __init__(self, env, width, height):
        super().__init__(env)
        self.width = width
        self.height = height
        # Update observation space
        self.observation_space = gymnasium.spaces.Box(
            low=0,
            high=255,
            shape=(self.height, self.width, env.observation_space.shape[2]),
            dtype=np.uint8,
        )

    def observation(self, observation):
        observation = cv2.resize(
            observation, (self.width, self.height), interpolation=cv2.INTER_AREA
        )
        return observation


# Environment wrapper to implement frame skipping without affecting rendering
class FrameSkipWrapper(gymnasium.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward = 0.0
        obs = None
        terminated = False
        truncated = False

        for _ in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward  # type: ignore

            # Always render intermediate frames
            self.env.render()

            if terminated or truncated:
                break

        return obs, total_reward, terminated, truncated, info


# Environment factory to wrap the environment creation
def make_env(render_mode):
    def _env_factory():
        env = gymnasium.make(ENV, render_mode=render_mode)
        # Custom frame skip wrapper that still renders all frames
        env = FrameSkipWrapper(env, skip=FRAME_SKIP)
        env = ObservationWrapper(env)
        env = ResizeWrapper(env, width=SCREEN_WIDTH, height=SCREEN_HEIGHT)
        env = TransformReward(env, lambda r: float(r) * REWARD_SCALE)  # Scale reward
        env = Monitor(env)
        return env

    return _env_factory


# Environment wrapper to wrap the vectorised environment
def wrap_vec_env(vec_env):
    vec_env = VecTransposeImage(vec_env)
    vec_env = VecFrameStack(vec_env, n_stack=FRAME_STACK)
    return vec_env


# Train the agent
def train():
    # Create the training and evaluation environments
    training_env = wrap_vec_env(SubprocVecEnv([make_env(RGB) for _ in range(NUM_ENVS)]))
    evaluation_env = wrap_vec_env(DummyVecEnv([make_env(RGB)]))

    # Create the agent
    agent = ppo.PPO(
        policy=policies.ActorCriticCnnPolicy,
        env=training_env,
        learning_rate=LEARNING_RATE,
        tensorboard_log=LOG_DIR,
        n_steps=N_STEPS,
        verbose=VERBOSE,
    )

    # Add an evaluation callback that will save the best model
    evaluation_callback = callbacks.EvalCallback(
        evaluation_env,
        n_eval_episodes=NUM_EVAL_EPISODES,
        eval_freq=EVAL_FREQ,
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
    )

    # Train the agent
    agent.learn(
        total_timesteps=TRAINING_STEPS,
        tb_log_name=LOG_NAME,
        callback=evaluation_callback,
    )

    training_env.close()
    evaluation_env.close()


# Demo the agent
def demo():
    # Load moedel
    model_path = Path(MODEL_DIR) / MODEL_NAME
    model = ppo.PPO.load(model_path)

    # Create the demo environment
    demo_env = wrap_vec_env(DummyVecEnv([make_env(HUMAN)]))

    # Initialise demo
    observation = demo_env.reset()

    # Run demo
    for _ in range(NUM_DEMO_EPISODES):
        done = [False]
        while not done[0]:
            action, _ = model.predict(observation, deterministic=True)  # type: ignore
            observation, _, done, _ = demo_env.step(action)
        observation = demo_env.reset()

    demo_env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()

    if args.train:
        train()
    else:
        demo()


if __name__ == "__main__":
    main()
