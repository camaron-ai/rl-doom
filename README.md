# The DOOM of AI
This project focuses on training an AI agent to play DOOM. It is a work in progress, with the ultimate goal of developing an agent able to play in a **multiplayer deathmatch environment**. We will start with a simple setup and gradually increase the complexity over time.

The first agent will be tested in a basic environment to evaluate the effectiveness of our approaches. It features:

- Convolutional Deep Q-Network (DQN).
- Double DQN.
- Dueling DQN.
- Prioritized Experience Replay.

## Setup

A Python package is included to help with training. To install it, run:

```sh
pip install .
```
## Training

Before training, you need a configuration file with the training settings. A sample file is provided in `config.yml`:


```yaml
name: basic_rl_doom
seed: 2025
enviroment_file:  simpler_basic.cfg
frame_repeat: 12
batch_size: 512
discount_factor: 0.99
lr: 0.00025
init_epsilon: 1.
epsilon_decay: 0.9996
min_epsilon: 0.1
image_resolution: [84, 84]
n_channels: 2
log_every_k_epochs: 1
steps_per_epoch: 512
num_epochs: 100
capacity: 50000
```

To start training, run:

```sh
python run.py train --confpath config.yml 
```

You can monitor your agent using the test command as follows
```sh
python run.py test --confpath config.yml 
```


## Contributing
Contributions are welcome! If you’d like to help improve this project, feel free to:

- Report bugs or suggest improvements by opening an issue
- Submit a pull request with new features or optimizations
- Share ideas or feedback in discussions
