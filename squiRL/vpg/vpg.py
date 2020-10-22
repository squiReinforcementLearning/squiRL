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
from torch.utils.data._utils import collate
from collections import OrderedDict

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
        # self.hparams.batch_size = self.hparams.episodes_per_batch * self.hparams.num_envs

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
        # ind = torch.nonzero(firsts).squeeze().numpy().tolist()
        # ind = [ind] + [self.hparams.batch_size] if type(
        # ind) == int else ind + [self.hparams.batch_size]
        # ind = [ind[0]] + [i - j for j, i in zip(ind, ind[1:])]
        # ind = [
        # self.hparams.episodes_per_batch
        # for i in range(self.hparams.num_envs)
        # ]

        ind = torch.nonzero(firsts).squeeze().numpy().tolist() + [
            firsts.shape[0]
        ]
        ind = np.add(ind, 1).tolist()
        ind = [ind[0]] + [i - j for j, i in zip(ind, ind[1:])]
        ind = ind[1:]

        actions = torch.split(actions, ind)
        rewards = torch.split(rewards, ind)
        firsts = torch.split(firsts, ind)
        # states = torch.cat(states)
        # ind = [len(s) for s in firsts]
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

        # result = pl.TrainResult(loss)
        self.log('loss',
                 loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=False,
                 logger=True)
        self.log('mean_episode_reward',
                 mean_episode_reward,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)

        return loss

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer

        Returns:
            List[Optimizer]: List of used optimizers
        """
        optimizer = optim.Adam(self.net.parameters(), lr=self.hparams.lr)
        return [optimizer]

    def collate_fn(self, batch):
        """Manually processes collected batch of experience

        Args:
            batch (TYPE): Current mini batch of replay data

        Returns:
            TYPE: Processed mini batch of replay data
        """
        # extract from batch
        states = [s[0].squeeze() for s in batch][0]
        actions = [s[1].squeeze() for s in batch][0]
        rewards = [s[2].squeeze() for s in batch][0]
        firsts = [s[3].squeeze() for s in batch][0]
        next_states = [s[4].squeeze() for s in batch][0]
        total_episodes_sampled = batch[0][5]

        # remove streams with no full episodes
        counts = [
            np.count_nonzero(firsts[:, i]) for i in range(firsts.shape[1])
        ]
        no_episode_streams = np.where(np.array(counts) <= 1)[0].tolist()
        for j, i in enumerate(no_episode_streams):
            i = i - j
            states = np.delete(states, i, 1)
            actions = np.delete(actions, i, 1)
            rewards = np.delete(rewards, i, 1)
            firsts = np.delete(firsts, i, 1)
            next_states = np.delete(next_states, i, 1)

        # removes incomplete episodes from remaining streams
        inds = [np.nonzero(firsts[:, i]) for i in range(firsts.shape[1])]
        cleaned_states = []
        cleaned_actions = []
        cleaned_rewards = []
        cleaned_firsts = []
        cleaned_next_states = []
        for j, i in enumerate(inds):
            i = i[0]
            c_firsts = np.delete(firsts[:, j], np.s_[:i[0]], 0)
            c_firsts = np.delete(c_firsts, np.s_[i[-1] - i[0]:], 0)
            cleaned_firsts.append(c_firsts)
            c_states = np.delete(states[:, j], np.s_[:i[0]], 0)
            c_states = np.delete(c_states, np.s_[i[-1] - i[0]:], 0)
            cleaned_states.append(c_states)
            c_actions = np.delete(actions[:, j], np.s_[:i[0]], 0)
            c_actions = np.delete(c_actions, np.s_[i[-1] - i[0]:], 0)
            cleaned_actions.append(c_actions)
            c_rewards = np.delete(rewards[:, j], np.s_[:i[0]], 0)
            c_rewards = np.delete(c_rewards, np.s_[i[-1] - i[0]:], 0)
            cleaned_rewards.append(c_rewards)
            c_next_states = np.delete(next_states[:, j], np.s_[:i[0]], 0)
            c_next_states = np.delete(c_next_states, np.s_[i[-1] - i[0]:], 0)
            cleaned_next_states.append(c_next_states)

        # convert to torch tensors
        states = torch.cat(collate.default_convert(cleaned_states))
        actions = torch.cat(collate.default_convert(cleaned_actions))
        rewards = torch.cat(collate.default_convert(cleaned_rewards))
        firsts = torch.cat(collate.default_convert(cleaned_firsts))
        next_states = torch.cat(collate.default_convert(cleaned_next_states))

        return states, actions, rewards, firsts, next_states

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
            collate_fn=self.collate_fn,
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
