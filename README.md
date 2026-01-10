# vis-doom
Training visual deep reinforcement learning agents to play Doom like it's 1993!

## Details
This code uses [ViZDoom](https://vizdoom.farama.org/) - a library for developing AI bots that play Doom using visual information. For each scenario, Doom is rendered at a screen resolution of 640x480 (because that's how we rolled back in the '90s). The agents are [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) Proximal Policy Optimisation (PPO) models with an actor-critic Convolutional Neural Network (CNN) policy. This is the "Nature CNN" made famous in the DeepMind papers on "Playing Atari with Deep Reinforcement Learning". For the ViZDoom "Deadly Corridor" scenario, this results in a model with ~886k learnable parameters. All agents "see" in full-colour and incorporate frame skipping, frame stacking, downsampling and reward scaling.

## Setup
[Install UV](https://docs.astral.sh/uv/getting-started/installation/), then run the following command:

```
uv venv
uv pip install -e .
```

## Pre-Trained Agents
### Basic Agent
Run the following commands to train, and monitor the agent:

```
uv run visdoom basic --train
uv run tensorboard --logdir models/basic
```

![Training Plots](/images/basic-training.png)

Run the following command to demo the trained agent:

```uv run visdoom basic```

![Basic Scenario](/images/basic-render.png)

### Defend the Centre Agent
Run the following commands to train, and monitor the agent:

```
uv run visdoom defend_the_centre --train
uv run tensorboard --logdir models/defend_the_centre
```

![Training Plots](/images/defend-the-centre-training.png)

Run the following command to demo the trained agent:

```uv run visdoom defend_the_centre```

![Defend the Centre Scenario](/images/defend-the-centre-render.png)

### Deadly Corridor Agent
Run the following commands to train, and monitor the agent:

```
uv run visdoom deadly_corridor --train
uv run tensorboard --logdir models/deadly_corridor
```

![Training Plots](/images/deadly-corridor-training.png)

Run the following command to demo the trained agent:

```uv run visdoom deadly_corridor```

![Deadly Corridor Scenario](/images/deadly-corridor-render.png)

## Training Agents for New Scenarios
To train an agent for a new scenario, add a new folder under `/models`. This needs to contain 3 files - Doom CFG and WAD files and an agent TOML file. See pre-trained agent scenario folders for examples and naming conventions.


## References
* [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) (DeepMind, NIPS Deep Learning Workshop, 2013)
* [ViZDoom: A Doom-based AI Research Platform for Visual Reinforcement Learning](https://arxiv.org/abs/1605.02097) (IEEE Conference on Computational Intelligence and Games, 2013)
* [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) (OpenAI, 2017)
* [Stable-Baselines3: Reliable Reinforcement Learning Implementations](https://jmlr.org/papers/v22/20-1364.html) (Journal of Machine Learning Research, 2021)
