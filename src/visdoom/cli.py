import argparse
from pathlib import Path
import tomllib
from visdoom.config import Config
from visdoom import Agent
from visdoom.constants import MODEL_FOLDER


def _get_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="ViZDoom Agent Runner")
    parser.add_argument(
        "scenario",
        help="The scenario to run.",
    )
    parser.add_argument("--train", action="store_true", help="Train the agent.")
    return parser.parse_args()


def _get_config(scenario: str) -> Config:
    """
    Retrieve the configuration for the given scenario.
    """
    with open(Path.cwd() / MODEL_FOLDER / scenario / (scenario + ".toml"), "rb") as f:
        data = tomllib.load(f)
    return Config(**data)


def main() -> None:
    """
    Main function to parse arguments and run the agent.
    """
    # Parse command line arguments
    args = _get_args()

    # Get configuration for the selected scenario
    config = _get_config(args.scenario)

    # Run the agent
    agent = Agent(args.scenario, config)
    if args.train:
        agent.train()
    else:
        agent.demo()


if __name__ == "__main__":
    main()
