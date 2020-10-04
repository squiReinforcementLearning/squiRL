import argparse
from copy import copy
from typing import Tuple, List
import gym
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data._utils import collate
from collections import OrderedDict

from squiRL.common.policies import MLP
from squiRL.common.data_stream import RLDataset, RolloutCollector
from squiRL.common.agents import Agent


class VPGLightning(pl.LightningModule):
    """ Basic VPG Model """
    def __init__(self, hparams: argparse.Namespace) -> None:
        super().__init__()
        self.hparams = hparams

        self.env = gym.make(self.hparams.env)
        self.gamma = self.hparams.gamma
        self.eps = self.hparams.eps
        obs_size = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.n

        self.net = MLP(obs_size, n_actions)  # .to('cuda:0')
        self.replay_buffer = RolloutCollector(self.hparams.episode_length)

        self.agent = Agent(self.env, self.replay_buffer)
        self.episode_reward = 0

    def reward_to_go(self, rewards: torch.Tensor) -> torch.tensor:
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
        Calculates the CE loss using a mini batch of a full episode

        Args:
            batch: current mini batch of replay data

        Returns:
            loss
        """
        states, actions, rewards, dones, next_states = batch

        action_logit = self.net(states.float()).to(self.device)
        log_probs = F.log_softmax(action_logit,
                                  dim=-1).squeeze(0)[range(len(actions)),
                                                     actions]

        discounted_rewards = self.reward_to_go(rewards)
        discounted_rewards = torch.tensor(discounted_rewards).to(self.device)
        advantage = (discounted_rewards - discounted_rewards.mean()) / (
            discounted_rewards.std() + self.eps)

        loss = -advantage * log_probs
        return loss.sum()

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor],
                      nb_batch) -> OrderedDict:
        """
        Carries out an entire episode in env and calculates loss

        Returns:
            Training loss and log metrics
        """
        _, _, rewards, _, _ = batch
        self.episode_reward = rewards.sum().detach()

        # calculates training loss
        loss = self.vpg_loss(batch)

        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)

        result = pl.TrainResult(loss)
        result.log('loss',
                   loss,
                   on_step=True,
                   on_epoch=True,
                   prog_bar=False,
                   logger=True)
        result.log('episode_reward',
                   self.episode_reward,
                   on_step=True,
                   on_epoch=True,
                   prog_bar=True,
                   logger=True)

        return result

    # def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
    #     _, _, rewards, _, _ = batch

    def configure_optimizers(self) -> List[Optimizer]:
        """ Initialize Adam optimizer"""
        optimizer = optim.Adam(self.net.parameters(), lr=self.hparams.lr)
        return [optimizer]

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch"""
        return batch[0].device.index if self.on_gpu else 'cpu'

    def collate_fn(self, batch):
        batch = collate.default_convert(batch)

        states = torch.cat([s[0] for s in batch])
        actions = torch.cat([s[1] for s in batch])
        rewards = torch.cat([s[2] for s in batch])
        dones = torch.cat([s[3] for s in batch])
        next_states = torch.cat([s[4] for s in batch])

        return states, actions, rewards, dones, next_states

    def __dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving
        experiences"""
        dataset = RLDataset(self.replay_buffer, self.env, self.net, self.agent)
        dataloader = DataLoader(
            dataset=dataset,
            collate_fn=self.collate_fn,
            batch_size=1,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader"""
        return self.__dataloader()
