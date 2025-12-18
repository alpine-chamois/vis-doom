# vis-doom
Training a visual RL agent to play Doom like it's 1993!

## Setup
[Install UV](https://docs.astral.sh/uv/getting-started/installation/), then run the following command:

```
uv sync
```

## Basic Agent
Run the following commands to train, and monitor the agent:

```
uv run -m visdoom.agents.basic --train
tensorboard --logdir models/basic
```

![Training Plots](/images/basic-training.png)

Run the following command to demo the trained agent:

```uv run -m visdoom.agents.basic```

![Basic Scenario](/images/basic-render.png)


## Defend the Centre Agent
Run the following commands to train, and monitor the agent:

```
uv run -m visdoom.agents.defend_the_centre --train
tensorboard --logdir models/defend_the_centre
```

Run the following command to demo the trained agent:

```uv run -m visdoom.agents.defend_the_centre```