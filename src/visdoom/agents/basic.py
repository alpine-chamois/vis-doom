import argparse
from pathlib import Path
from typing import Callable
import gymnasium
from gymnasium.envs.registration import register
from gymnasium import Env
from gymnasium.wrappers import TransformReward
from stable_baselines3.ppo import PPO
from stable_baselines3.common import callbacks, policies
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    DummyVecEnv,
    VecTransposeImage,
    VecFrameStack,
    VecEnv,
)
from stable_baselines3.common.monitor import Monitor
from visdoom.wrappers import FrameSkipWrapper, ResizeWrapper, ScreenWrapper

# Config
ENV = "MyVizdoomBasic-v0"
SCENARIO_DIR = Path.cwd() / "scenarios"
CFG_FILE = SCENARIO_DIR / "basic.cfg"
LOG_NAME = "basic"
MODEL_DIR = "models/basic/"
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


def make_env(render_mode: str) -> Callable[[], Env]:
    """
    Factory function to create the environment with the necessary wrappers.
    """

    def _env_factory() -> Env:
        """
        Create the environment with the necessary wrappers.
        """
        env = gymnasium.make(ENV, render_mode=render_mode)
        # Custom frame skip wrapper that still renders all frames
        env = FrameSkipWrapper(env, skip=FRAME_SKIP)
        env = ScreenWrapper(env)
        env = ResizeWrapper(env, width=SCREEN_WIDTH, height=SCREEN_HEIGHT)
        env = TransformReward(env, lambda r: float(r) * REWARD_SCALE)  # Scale reward
        env = Monitor(env)
        return env

    return _env_factory


def wrap_vec_env(vec_env: VecEnv) -> VecEnv:
    """
    Apply wrappers to the vectorized environment.
    """
    vec_env = VecTransposeImage(vec_env)
    vec_env = VecFrameStack(vec_env, n_stack=FRAME_STACK)
    return vec_env


# Train the agent
def train() -> None:
    """
    Train the PPO agent on the VizDoom scenario.
    """
    # Create the training and evaluation environments
    training_env = wrap_vec_env(SubprocVecEnv([make_env(RGB) for _ in range(NUM_ENVS)]))
    evaluation_env = wrap_vec_env(DummyVecEnv([make_env(RGB)]))

    # Create the agent
    agent = PPO(
        policy=policies.ActorCriticCnnPolicy,
        env=training_env,
        learning_rate=LEARNING_RATE,
        tensorboard_log=MODEL_DIR,
        n_steps=N_STEPS,
        verbose=VERBOSE,
    )

    # Add an evaluation callback that will save the best model
    evaluation_callback = callbacks.EvalCallback(
        evaluation_env,
        n_eval_episodes=NUM_EVAL_EPISODES,
        eval_freq=EVAL_FREQ,
        best_model_save_path=MODEL_DIR,
        log_path=MODEL_DIR,
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
def demo() -> None:
    """
    Run a demo of the trained PPO agent.
    """
    # Load moedel
    model_path = Path(MODEL_DIR) / MODEL_NAME
    model = PPO.load(model_path)

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


def main() -> None:
    """
    Main function to parse arguments and run training or demo.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()

    if args.train:
        train()
    else:
        demo()


if __name__ == "__main__":
    main()
