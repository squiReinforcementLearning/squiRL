import argparse
from typing import List

import gym3
import pytorch_lightning as pl
import torch
import torch.optim as optim
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from squiRL.common import reg_policies
from squiRL.common.agents import GreedyAgent
from squiRL.common.data_stream import ExperienceReplayBuffer, ExperienceReplayDataset


class DQN(pl.LightningModule):
    def __init__(self, hparams: argparse.Namespace) -> None:
        super(DQN, self).__init__()
        self.hparams = hparams
        self.save_hyperparameters()

        self.env = gym3.vectorize_gym(num=self.hparams.num_envs,
                                      env_kwargs={"id": self.hparams.env})
        self.gamma = self.hparams.gamma
        self.eps = self.hparams.eps
        obs_size = self.env.obs_space.shape
        action_size = self.env.ac_space.shape

        self.net = reg_policies[self.hparams.policy](obs_size, action_size)
        self.target_net = reg_policies[self.hparams.policy](obs_size, action_size)
        self.target_net.load_state_dict(self.net.state_dict())
        self.replay_buffer = ExperienceReplayBuffer(self.hparams.buffer_size, self.hparams.batch_size)

        self.agent = GreedyAgent(self.env, self.replay_buffer, self.hparams.eps_start, self.hparams.eps_end,
                                 self.hparams.eps_frames)

    @staticmethod
    def add_model_specific_args(
            parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Adds model specific config args to parser

        Args:
            parent_parser (argparse.ArgumentParser): Argument parser

        Returns:
            argparse.ArgumentParser: Updated argument parser
        """
        parser.add_argument("--policy",
                            type=str,
                            default='MLP',
                            help="NN policy used by agent")
        parser.add_argument("--lr",
                            type=float,
                            default=0.0005,
                            help="learning rate")
        parser.add_argument("--eps_start",
                            type=float,
                            default=1e-3,
                            help="greedy action chance start value")
        parser.add_argument("--eps_end",
                            type=float,
                            default=1e-5,
                            help="greedy action chance end value")
        parser.add_argument("--eps_frames",
                            type=float,
                            default=1e6,
                            help="greedy action chance decay over number of frames")
        parser.add_argument("--gamma",
                            type=float,
                            default=0.99,
                            help="discount factor")
        parser.add_argument("--num_workers",
                            type=int,
                            default=20,
                            help="num of dataloader cpu workers")
        parser.add_argument("--num_envs",
                            type=int,
                            default=1,
                            help="num of parallel envs")
        parser.add_argument("--batches_per_epoch",
                            type=int,
                            default=1000,
                            help="number of sampled batches per epoch")
        return parser

    # TODO: verify!
    def calculate_loss(self, batch):
        states, actions, rewards, dones, next_states = batch

        actions = actions.long().squeeze(-1)

        state_action_values = (
            self.net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        )

        with torch.no_grad():
            next_state_values = self.target_net(next_states).max(1)[0]
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * self.gamma + rewards

        return nn.MSELoss()(state_action_values, expected_state_action_values)

    def training_step(self, batch, nb_batch):
        loss = self.calculate_loss(batch)

        self.log('loss',
                 loss,
                 on_step=True,
                 on_epoch=False,
                 prog_bar=True,
                 logger=True)

        return loss

    def configure_optimizers(self) -> List[Optimizer]:
        """ Initialize Adam optimizer"""
        optimizer = optim.Adam(self.net.parameters(), lr=self.hparams.lr)
        return [optimizer]

    def step_once(self):
        self.agent.play_step(self.target_net)

    def __dataloader(self) -> DataLoader:
        """Initialize the RL dataset used for retrieving experiences

        Returns:
            DataLoader: Handles loading the data for training
        """
        dataset = ExperienceReplayDataset(replay_buffer=self.replay_buffer,
                                          episodes_per_epoch=self.hparams.episodes_per_batch,
                                          step_function=self.step_once)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=1,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train data loader

        Returns:
            DataLoader: Handles loading the data for training
        """
        return self.__dataloader()

    # TODO: implement test dataloader
