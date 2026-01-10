from typing import Callable
from pathlib import Path
import gymnasium
from gymnasium import Env, register, registry
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
from visdoom.config import Config
from visdoom.constants import ENV_ID, RGB, HUMAN, MODEL_FOLDER
import torch


class Agent:
    """
    Class to train and demo agents.
    """

    def __init__(self, scenario: str, config: Config) -> None:
        self.scenario = scenario
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        super().__init__()

    def _make_env(self, render_mode: str) -> Callable[[], Env]:
        """
        Returns a factory function to create the environment with the necessary wrappers.
        """

        def _env_factory() -> Env:
            """
            Create the environment with the necessary wrappers.
            """
            # Register the environment
            if ENV_ID not in registry:
                register(
                    id=ENV_ID,
                    entry_point="vizdoom.gymnasium_wrapper.gymnasium_env_defns:VizdoomScenarioEnv",
                    kwargs={
                        "scenario_file": Path.cwd() / MODEL_FOLDER / self.scenario / (self.scenario + ".cfg"),
                    },
                )

            # Create the environment
            env = gymnasium.make(ENV_ID, render_mode=render_mode)
            # Custom frame skip wrapper that still renders all frames
            env = FrameSkipWrapper(env, skip=self.config.frame_skip)
            env = ScreenWrapper(env)
            env = ResizeWrapper(env, width=self.config.screen_width, height=self.config.screen_height)
            # env = GrayscaleWrapper(env)
            env = TransformReward(env, lambda r: float(r) * self.config.reward_scale)  # Scale reward
            env = Monitor(env)
            return env

        return _env_factory

    def _wrap_vec_env(self, vec_env: VecEnv) -> VecEnv:
        """
        Apply wrappers to the vectorized environment.
        """
        vec_env = VecTransposeImage(vec_env)
        vec_env = VecFrameStack(vec_env, n_stack=self.config.frame_stack)
        return vec_env

    # Train the agent
    def train(self) -> None:
        """
        Train the PPO agent on the VizDoom scenario.
        """
        # Create the training and evaluation environments
        training_env = self._wrap_vec_env(SubprocVecEnv([self._make_env(RGB) for _ in range(self.config.num_envs)]))
        evaluation_env = self._wrap_vec_env(DummyVecEnv([self._make_env(RGB)]))

        # Create the agent
        model = PPO(
            policy=policies.ActorCriticCnnPolicy,
            env=training_env,
            learning_rate=self.config.learning_rate,
            tensorboard_log=MODEL_FOLDER + "/" + self.scenario,
            n_steps=self.config.n_steps,
            verbose=self.config.verbose,
            ent_coef=self.config.ent_coef,
            device=self.device,
        )

        # Add an evaluation callback that will save the best model
        evaluation_callback = callbacks.EvalCallback(
            evaluation_env,
            n_eval_episodes=self.config.num_eval_episodes,
            eval_freq=self.config.eval_freq,
            best_model_save_path=MODEL_FOLDER + "/" + self.scenario,
        )

        # Train the agent
        model.learn(
            total_timesteps=self.config.training_steps,
            tb_log_name=self.scenario,
            callback=evaluation_callback,
        )

        training_env.close()
        evaluation_env.close()

    # Demo the agent
    def demo(self) -> None:
        """
        Run a demo of the trained PPO agent.
        """
        # Load model
        model_path = MODEL_FOLDER + "/" + self.scenario + "/" + "best_model.zip"
        model = PPO.load(model_path, device=self.device)

        # Create the demo environment
        demo_env = self._wrap_vec_env(DummyVecEnv([self._make_env(HUMAN)]))

        # Initialise demo
        observations = demo_env.reset()

        # Run demo
        for _ in range(self.config.num_demo_episodes):
            dones = [False]
            while not dones[0]:
                actions, _ = model.predict(observations, deterministic=True)  # type: ignore
                observations, _, dones, _ = demo_env.step(actions)
            observations = demo_env.reset()

        demo_env.close()
