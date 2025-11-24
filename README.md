# vis-doom
Training a visual RL agent to play Doom like it's 1993!

## Installation
[Install UV](https://docs.astral.sh/uv/getting-started/installation/), then run:

```uv run examples/gymnasium_wrapper.py```

## Basic Agent
Run the following commands to train, and monitor the agent:

```uv run agents/basic.py --train```

```tensorboard --logdir logs```

![Training Plots](/images/basic-training.png)

Run the following command to demo the trained agent:

```uv run agents/basic.py```

![Basic Scenario](/images/basic-render.png)
