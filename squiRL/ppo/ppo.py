"""Script for training workflow of the Vanilla Policy Gradient Algorithm.
"""
import argparse
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

from squiRL.common.utils import collate_episodes, reward_to_go
from squiRL.common import reg_policies
from squiRL.common.data_stream import RLDataset, RolloutCollector
from squiRL.common.agents import Agent


class PPO(pl.LightningModule):
    """Basic Vanilla Policy Gradient training acrhitecture

    Attributes:
        agent (Agent): Agent that interacts with env
        env (gym3.Env): OpenAI gym environment
        eps (float): Small offset used in calculating loss
        gamma (float): Discount rate
        hparams (argeparse.Namespace): Stores all passed args
        actor (nn.Module): NN used to learn policy
        critic (nn.Module): NN used to evaluate policy
        replay_buffer (RolloutCollector): Stores generated experience
    """
    def __init__(self, hparams: argparse.Namespace) -> None:
        """Initializes PPO class

        Args:
            hparams (argparse.Namespace): Stores all passed args
        """
        super(PPO, self).__init__()
        self.hparams = hparams
        self.automatic_optimization = False

        self.env = gym3.vectorize_gym(num=self.hparams.num_envs,
                                      env_kwargs={"id": self.hparams.env})
        self.gamma = self.hparams.gamma
        self.eps = self.hparams.eps
        obs_size = self.env.ob_space.size
        n_actions = self.env.ac_space.eltype.n

        self.actor = reg_policies[self.hparams.policy](obs_size, n_actions)
        self.new_actor = reg_policies[self.hparams.policy](obs_size, n_actions)
        self.critic = reg_policies[self.hparams.policy](obs_size, 1)
        self.replay_buffer = RolloutCollector(self.hparams.episodes_per_batch)

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
        parser.add_argument("--custom_optimizers",
                            type=bool,
                            default=True,
                            help="this value must not be changed")
        parser.add_argument("--actor_updates_per_iter",
                            type=int,
                            default=10,
                            help="model updates per iteration")
        parser.add_argument("--clip_rt",
                            type=float,
                            default=0.1,
                            help="actor update clipping ratio")
        parser.add_argument("--lr_critic",
                            type=float,
                            default=0.0005,
                            help="learning rate")
        parser.add_argument("--lr_actor",
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

    def ppo_loss(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                 actor_optimizer) -> torch.Tensor:
        """
        Calculates the actor and critic losses based on the REINFORCE
        objective, using the discounted reward to go per episode step

        Args:
            batch (Tuple[torch.Tensor, ,torch.Tensor, torch.Tensor]): Current
            mini batch of replay data

        Returns:
            torch.Tensor: Calculated loss
        """
        action_logits, actions, rewards, states, values = batch

        log_probs = F.log_softmax(action_logits,
                                  dim=-1).squeeze(0)[range(len(actions)),
                                                     actions]
        discounted_rewards = reward_to_go(rewards, self.gamma)
        discounted_rewards = torch.tensor(discounted_rewards).float()
        advantage = discounted_rewards - values
        advantage = advantage.type_as(log_probs)
        criterion = torch.nn.MSELoss()
        critic_loss = criterion(discounted_rewards, values.view(-1).float())

        for _ in range(self.hparams.actor_updates_per_iter):
            actor_optimizer.zero_grad()
            new_action_logits = self.new_actor(states.float())
            new_log_probs = F.log_softmax(
                new_action_logits, dim=-1).squeeze(0)[range(len(actions)),
                                                      actions]
            ratio = torch.exp(new_log_probs - log_probs)
            clip_min = 1 - self.hparams.clip_rt
            clip_max = 1 + self.hparams.clip_rt
            clipped_ratio = torch.clamp(ratio, clip_min, clip_max)
            actor_loss = -torch.min(clipped_ratio * advantage,
                                    ratio * advantage)
            self.manual_backward(actor_loss.sum(),
                                 actor_optimizer,
                                 retain_graph=True)
            actor_optimizer.step()

        return actor_loss.sum(), critic_loss.mean()

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], nb_batch,
                      optimizer_idx) -> OrderedDict:
        """
        Carries out an entire episode in env and calculates loss

        Returns:
            OrderedDict: Training step result

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): Current mini batch of
            replay data
            nb_batch (TYPE): Current index of mini batch of replay data
        """
        (actor_optimizer, critic_optimizer) = self.optimizers()
        states, actions, rewards, firsts, _ = batch
        ind = [len(i) for i in firsts]

        action_logits = torch.split(self.actor(torch.cat(states).float()), ind)
        values = torch.split(self.critic(torch.cat(states).float()), ind)

        actor_loss, critic_loss = 0, 0
        for ep in range(len(actions)):
            if rewards[ep].shape[0] == 1:
                continue
            ac_loss, cr_loss = self.ppo_loss(
                (action_logits[ep], actions[ep], rewards[ep], states[ep],
                 values[ep]), actor_optimizer)
            actor_loss += ac_loss
            critic_loss += cr_loss

        self.manual_backward(cr_loss)
        critic_optimizer.step()
        critic_optimizer.zero_grad()

        mean_episode_reward = torch.tensor(
            np.mean([i.sum().item() for i in rewards]))

        self.actor.load_state_dict(self.new_actor.state_dict())

        self.log('actor_loss',
                 actor_loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=False,
                 logger=True)
        self.log('critic_loss',
                 critic_loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=False,
                 logger=True)
        self.log('mean_episode_reward',
                 mean_episode_reward,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)

        return actor_loss + critic_loss

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimize, statesr

        Returns:
            List[Optimizer]: List of used optimizers
        """
        actor_optimizer = optim.Adam(self.new_actor.parameters(),
                                     lr=self.hparams.lr_actor)
        critic_optimizer = optim.Adam(self.critic.parameters(),
                                      lr=self.hparams.lr_critic)
        return actor_optimizer, critic_optimizer

    def __dataloader(self) -> DataLoader:
        """Initialize the RL dataset used for retrieving experiences

        Returns:
            DataLoader: Handles loading the data for training
        """
        dataset = RLDataset(self.replay_buffer,
                            self.hparams.episodes_per_batch, self.actor,
                            self.agent, self.hparams.num_envs)
        dataloader = DataLoader(
            dataset=dataset,
            collate_fn=collate_episodes,
            batch_size=1,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train data loader

        Returns:
            DataLoader: Handles loading the data for training
        """
        return self.__dataloader()
