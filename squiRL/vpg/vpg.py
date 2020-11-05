"""Script for training workflow of the Vanilla Policy Gradient Algorithm.
"""
import argparse
from copy import copy
from typing import Tuple, List
import gym3
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from collections import OrderedDict

from squiRL.common.utils import collate_episodes
from squiRL.common import reg_policies
from squiRL.common.data_stream import RLDataset, RolloutCollector
from squiRL.common.agents import Agent


class VPG(pl.LightningModule):
    """Basic Vanilla Policy Gradient training acrhitecture

    Attributes:
        agent (Agent): Agent that interacts with env
        env (gym3.Env): OpenAI gym environment
        eps (float): Small offset used in calculating loss
        gamma (float): Discount rate
        hparams (argeparse.Namespace): Stores all passed args
        net (nn.Module): NN used to learn policy
        replay_buffer (RolloutCollector): Stores generated experience
    """
    def __init__(self, hparams: argparse.Namespace) -> None:
        """Initializes VPG class

        Args:
            hparams (argparse.Namespace): Stores all passed args
        """
        super(VPG, self).__init__()
        self.hparams = hparams

        self.env = gym3.vectorize_gym(num=self.hparams.num_envs,
                                      env_kwargs={"id": self.hparams.env})
        self.gamma = self.hparams.gamma
        self.eps = self.hparams.eps
        obs_size = self.env.ob_space.size
        n_actions = self.env.ac_space.eltype.n

        self.net = reg_policies[self.hparams.policy](obs_size, n_actions)
        self.replay_buffer = RolloutCollector(self.hparams.episode_length)

        self.agent = Agent(self.env, self.replay_buffer)

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
        parser.add_argument("--eps",
                            type=float,
                            default=np.finfo(np.float32).eps.item(),
                            help="small offset")
        parser.add_argument("--gamma",
                            type=float,
                            default=0.99,
                            help="discount factor")
        parser.add_argument("--num_workers",
                            type=int,
                            default=20,
                            help="num of dataloader cpu workers")
        parser.add_argument("--episodes_per_batch",
                            type=int,
                            default=1,
                            help="num of episodes per batch")
        parser.add_argument("--num_envs",
                            type=int,
                            default=1,
                            help="num of parallel envs")
        return parser

    def reward_to_go(self, rewards: torch.Tensor) -> torch.tensor:
        """Calculates reward to go over an entire episode

        Args:
            rewards (torch.Tensor): Episode rewards

        Returns:
            torch.tensor: Reward to go for each episode step
        """
        rewards = rewards.detach().cpu().numpy()
        res = []
        sum_r = 0.0
        for r in reversed(rewards):
            sum_r *= self.gamma
            sum_r += r
            res.append(copy(sum_r))
        return list(reversed(res))

    def vpg_loss(self, batch: Tuple[torch.Tensor,
                                    torch.Tensor]) -> torch.Tensor:
        """
        Calculates the loss based on the REINFORCE objective, using the
        discounted
        reward to go per episode step

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): Current mini batch of
            replay data

        Returns:
            torch.Tensor: Calculated loss
        """
        action_logits, actions, rewards = batch

        log_probs = F.log_softmax(action_logits,
                                  dim=-1).squeeze(0)[range(len(actions)),
                                                     actions]

        discounted_rewards = self.reward_to_go(rewards)
        discounted_rewards = torch.tensor(discounted_rewards)
        advantage = (discounted_rewards - discounted_rewards.mean()) / (
            discounted_rewards.std() + self.eps)
        advantage = advantage.type_as(log_probs)

        loss = -advantage * log_probs
        return loss.sum()

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor],
                      nb_batch) -> OrderedDict:
        """
        Carries out an entire episode in env and calculates loss

        Returns:
            OrderedDict: Training step result

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): Current mini batch of
            replay data
            nb_batch (TYPE): Current index of mini batch of replay data
        """
        states, actions, rewards, firsts, _ = batch
        ind = torch.nonzero(firsts).squeeze().numpy().tolist() + [
            firsts.shape[0]
        ]
        ind = np.add(ind, 1).tolist()
        ind = [ind[0]] + [i - j for j, i in zip(ind, ind[1:])]
        ind = ind[1:]

        actions = torch.split(actions, ind)
        rewards = torch.split(rewards, ind)
        firsts = torch.split(firsts, ind)
        action_logits = torch.split(self.net(states.float()), ind)
        episode_rewards = []
        loss = 0
        for ep in range(len(actions)):
            if rewards[ep].shape[0] == 1:
                continue
            episode_rewards.append(rewards[ep].sum().item())
            loss += self.vpg_loss(
                (action_logits[ep], actions[ep], rewards[ep]))
        mean_episode_reward = torch.tensor(np.mean(episode_rewards))

        result = pl.TrainResult(loss)
        result.log('loss',
                   loss,
                   on_step=True,
                   on_epoch=True,
                   prog_bar=False,
                   logger=True)
        result.log('mean_episode_reward',
                   mean_episode_reward,
                   on_step=True,
                   on_epoch=True,
                   prog_bar=True,
                   logger=True)

        return result

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer

        Returns:
            List[Optimizer]: List of used optimizers
        """
        optimizer = optim.Adam(self.net.parameters(), lr=self.hparams.lr)
        return [optimizer]

    def __dataloader(self) -> DataLoader:
        """Initialize the RL dataset used for retrieving experiences

        Returns:
            DataLoader: Handles loading the data for training
        """
        dataset = RLDataset(self.replay_buffer,
                            self.hparams.episodes_per_batch, self.net,
                            self.agent, self.hparams.num_envs)
        dataloader = DataLoader(
            dataset=dataset,
            collate_fn=collate_episodes,
            # batch_size=self.hparams.episodes_per_batch,
            batch_size=1,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train data loader

        Returns:
            DataLoader: Handles loading the data for training
        """
        return self.__dataloader()
