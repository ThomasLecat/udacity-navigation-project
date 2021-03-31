# udacity-navigation-project

This repository implements an RL agent that solves the Unity banana environment.

The agent is a DQN with the following extensions:
*  Double DQN
*  Dueling DQN
*  Prioritized Experience Replay

## Installation

This project uses the drlnd conda environment from the Udacity Deep Reinforcement
Learning program.

1. Follow the instructions from Udacity's [README](https://github.com/udacity/deep-reinforcement-learning#dependencies) 
to create the environment and install the dependencies.
2. Download the RL environment for your OS, place the file in the 
navigation/ folder and unzip (or decompress) it. 

*  Linux: click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
*  Mac OSX: click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
*  Windows (32-bit): click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
*  Windows (64-bit): click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

(Optional) To contribute, install the additional dependencies and pre-commits:

```bash
$ source activate drlnd
$ pip install -r requirements.txt
# pre-commit install
```

## Description of the environment

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is 
provided for collecting a blue banana. The goal is to collect as many yellow bananas 
as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with 
ray-based perception of objects around the agent's forward direction.

Four discrete actions are available, corresponding to:

    0 - move forward.
    1 - move backward.
    2 - turn left.
    3 - turn right.
