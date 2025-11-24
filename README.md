# vis-doom
Training a visual RL agent to play Doom like it's 1993!

## Installation
[Install UV](https://docs.astral.sh/uv/getting-started/installation/), then run:

```uv run examples/gymnasium_wrapper.py```

## Basic Agent
Run the following commands to train, and monitor the agent:

```uv run agents/basic.py --train```

```tensorboard --logdir logs```

Here's how it trains:

![Training Plots](/images/basic-training.png)

And here it is playing the game:

![Basic Scenario](/images/basic-render.png)