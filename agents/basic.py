from pathlib import Path
import gymnasium
from gymnasium.envs.registration import register
from stable_baselines3 import ppo
from stable_baselines3.common import callbacks
from stable_baselines3.common import policies
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TransformReward

ENV = "MyVizdoomBasic-v0"
SCENARIO_DIR = Path(__file__).parent.parent / "scenarios"
CFG_FILE = SCENARIO_DIR / "basic.cfg"
LOG_NAME = "basic"
LOG_DIR = "logs/basic/"
MODEL_DIR = "models/basic/"
TRAINING_STEPS = 25000
FRAME_SKIP = 4
NUM_ENVS = 1  # 4
NUM_EVAL_EPISODES = 10
EVAL_FREQ = 500  # 2500
LEARNING_RATE = 1e-4
OBS_KEY = "screen"
N_STEPS = 128
VERBOSE = 1
REWARD_SCALE = 0.01

# Register custom VizDoom environment with local scenario file
register(
    id=ENV,
    entry_point="vizdoom.gymnasium_wrapper.gymnasium_env_defns:VizdoomScenarioEnv",
    kwargs={
        "scenario_file": CFG_FILE,
    },
)


# Environment wrapper to extract only the screen observation
class ObservationWrapper(gymnasium.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Replace the dict space with only the screen box
        self.observation_space = env.observation_space[OBS_KEY]  # type: ignore

    def observation(self, observation):
        return observation[OBS_KEY]


# Environment factory to wrap the environment creation
def make_env(render_mode="none"):
    def _env_factory():
        env = gymnasium.make(ENV, render_mode=render_mode, frame_skip=FRAME_SKIP)
        env = ObservationWrapper(env)
        env = TransformReward(env, lambda r: float(r) * REWARD_SCALE)
        env = Monitor(env)
        # TODO: Consider frame stacking
        return env

    return _env_factory


def main():
    # Create the training and evaluation environments
    training_env = SubprocVecEnv([make_env("none") for _ in range(NUM_ENVS)])
    evaluation_env = DummyVecEnv([make_env("human")])

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


if __name__ == "__main__":
    main()

# TODO: Parameterise for either training or evaluation (with rendering) of a trained model

