# udacity-navigation-project

This repository implements an RL agent that solves the Unity banana environment.

The agent is a DQN with the following extensions:
*  [x] Double DQN
*  [ ] Dueling DQN *---- coming soon -----*
*  [ ] Prioritized Experience Replay    *---- coming soon ----*

## Installation

This project uses the drlnd conda environment from the Udacity Deep Reinforcement
Learning program.

1. Follow the instructions from Udacity's [README](https://github.com/udacity/deep-reinforcement-learning#dependencies) 
to create the environment and install the dependencies.
1. Install the project's package: `$ source activate drlnd && pip install -e .`
1. Download the RL environment for your OS, place the file in the navigation/ folder 
and unzip (or decompress) it. 

*  Linux: click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
*  Mac OSX: click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
*  Windows (32-bit): click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
*  Windows (64-bit): click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

*(Optional)* To contribute, install the pre-commits:

```bash
$ pre-commit install
```

## Usage

Before training or evaluating an agent, make sure you conda environment is activated:
```
$ source activate drlnd
```

### Training

1. Tune DQN's learning parameters in `navigation/config.py`
2. run `python navigation/train.py --environment_path /path/to/Banana.app`. You can 
also specify the number of training episodes with the `--num_episodes` argument.

At the end of training, two files are saved on disk:
*  `dqn_checkpoint.pt`: PyTorch checkpoint containing the trained model's weights.
*  `reward_per_episode.csv`: score of all training episodes.

### Evaluation

Using the same config parameters as in training, run:
```
python navigation/evaluate.py --environment_path /path/to/Banana.app --checkpoint_path dqn_checkpoint.pt --show_graphics True
```

## Description of the environment

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is 
provided for collecting a blue banana. The goal is to collect as many yellow bananas 
as possible while avoiding blue bananas.

The state space is a vector of 37 dimensions which contains the agent's velocity, along
with ray-based perception of objects around the agent's forward direction.

Four discrete actions are available, corresponding to:

    0 - move forward.
    1 - move backward.
    2 - turn left.
    3 - turn right.

The environment is considered solved when the agent receives an average reward of at 
least +13 over 100 consecutive episodes.

![Agent playing on Banana environment](doc/banana.gif)
