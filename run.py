from vizdoom.vizdoom import DoomGame
import os
import pprint
from typing import Tuple
from pydantic import BaseModel
from rldoom.agent import DuelQNet, V1StateProcessor, Agent
from rldoom.train import Trainer
from rldoom.test import VisualizeAgent
from rldoom.memory import PerMemory
from rldoom.util import load_yml, get_torch_device
from rldoom.envs import setup_env 
import typer
import torch
import glob

cli = typer.Typer()
LOGDIR = os.path.join("logs")


class ExperimentConfig(BaseModel):
    name: str
    seed: int
    enviroment_file: str
    frame_repeat: int
    batch_size: int
    discount_factor: float
    lr: float
    init_epsilon: float
    epsilon_decay: float 
    min_epsilon: float
    image_resolution: Tuple[int, int]
    n_channels: int
    log_every_k_epochs: int
    steps_per_epoch: int
    num_epochs: int
    capacity: int


@cli.command()
def train(confpath: str = "config.yml"):

    # load configuration file
    config_as_dict = load_yml(confpath)
    # get device
    device = get_torch_device()
    config = ExperimentConfig.model_validate(config_as_dict)
    pprint.pprint(config_as_dict)
    logdir = os.path.join(LOGDIR, config.name)
    game = setup_env(config.enviroment_file)
    
    # SETUP AGENT
    state_processor = V1StateProcessor(size=config.image_resolution)
    n_actions = 2 ** game.get_available_buttons_size()
    policy_net = DuelQNet(n_actions, n_channles=config.n_channels)
    target_net = DuelQNet(n_actions, n_channles=config.n_channels)


    print(policy_net)

    agent = Agent(
        n_actions=n_actions,
        state_processor=state_processor,
        policy_net=policy_net,
        target_net=target_net,
        batch_size=config.batch_size,
        discount_factor=config.discount_factor,
        lr=config.lr,
        device=device
    )

    # SETUP MEMORY
    memory = PerMemory(
        state_dim=(config.n_channels, *config.image_resolution),
        capacity=config.capacity,
    )

    # SETUP TRAINER
    trainer = Trainer(
        game=game,
        logdir=logdir,
        memory=memory,
        num_epochs=config.num_epochs,
        steps_per_epoch=config.steps_per_epoch,
        log_every_k_epochs=config.log_every_k_epochs,
        seed=config.seed 
    )
    trainer.train(agent)



@cli.command()
def test(
    confpath: str = "config.yml",
    epoch: int = -1):
    # load configuration file
    config_as_dict = load_yml(confpath)
    # setup device
    device = get_torch_device()
    config = ExperimentConfig.model_validate(config_as_dict)
    pprint.pprint(config_as_dict)
    logdir = os.path.join(LOGDIR, config.name)
    game = setup_env(config.enviroment_file)
    
    # SETUP AGENT
    state_processor = V1StateProcessor(size=config.image_resolution)
    n_actions = 2 ** game.get_available_buttons_size()
    policy_net = DuelQNet(n_actions, n_channles=config.n_channels)

    # find the correct checkpoint based on the epoch
    cpkt_pattern = os.path.join(logdir, "*.ckpt")
    cpkt_paths = sorted(glob.glob(cpkt_pattern))
    epoch = min(epoch, len(cpkt_paths) - 1)
    cpkt_path = cpkt_paths[epoch]
    print(f"using checkpoint: {cpkt_path}")
    # load checkpoint
    state_dict = torch.load(cpkt_path)
    policy_net.load_state_dict(state_dict, strict=True)

    print(policy_net)

    agent = Agent(
        n_actions=n_actions,
        state_processor=state_processor,
        policy_net=policy_net,
        target_net=policy_net,
        batch_size=config.batch_size,
        discount_factor=config.discount_factor,
        lr=config.lr,
        device=device
    )


    tester = VisualizeAgent(game, episodes_to_watch=10, frame_repeat=config.frame_repeat)
    tester.run(agent)


if __name__ == '__main__':
    cli()