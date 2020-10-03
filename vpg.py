import argparse
from collections import OrderedDict, deque
from collections import namedtuple
from copy import copy
from typing import Tuple, List

import gym
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data._utils import collate
from torch.utils.data.dataset import IterableDataset

Experience = namedtuple('Experience',
                        ('state', 'action', 'reward', 'done', 'last_state'))


class MLP(nn.Module):
    """
    Simple MLP network

    Args:
        obs_size: observation/obs size of the environment
        n_actions: number of discrete actions available in the environment
        layers: size of hidden layers
    """
    def __init__(self, obs_size: int, n_actions: int, layers: int = [64, 32]):
        super(MLP, self).__init__()
        self.fc = nn.ModuleList()
        self.layers_size = len(layers)
        prev = obs_size
        for n in range(0, self.layers_size):
            self.fc.append(nn.Linear(prev, layers[n]))
            prev = layers[n]
        self.fc.append(nn.Linear(prev, n_actions))
        self.layers_n = len(self.fc) - 1

    def forward(self, x):
        for n in range(self.layers_n):
            x = F.relu(self.fc[n](x))
        x = self.fc[self.layers_n](x)
        return x


class RolloutCollector:
    """
    Buffer for collecting rollout experiences allowing the agent to learn from
    them

    Args:
        capacity: size of the buffer
    """
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.replay_buffer = deque(maxlen=self.capacity)

    def __len__(self) -> None:
        return len(self.replay_buffer)

    def append(self, experience: Experience) -> None:
        """
        Add experience to the buffer

        Args:
            experience: tuple (state, action, reward, done, new_state)
        """
        self.replay_buffer.append(experience)

    def sample(self) -> Tuple:
        states, actions, rewards, dones, next_states = zip(
            *[self.replay_buffer[i] for i in range(len(self.replay_buffer)-1)])

        return (np.array(states), np.array(actions),
                np.array(rewards, dtype=np.float32),
                np.array(dones, dtype=np.bool), np.array(next_states))

    def empty_buffer(self):
        self.replay_buffer.clear()


class Agent:
    """
    Base Agent class handling the interaction with the environment

    Args:
        env: training environment
    """
    def __init__(self, env: gym.Env, replay_buffer: RolloutCollector) -> None:
        self.env = env
        self.replay_buffer = replay_buffer
        self.reset()

    def reset(self) -> None:
        """ Resets the environment and updates the obs"""
        self.obs = self.env.reset()

    def get_action(self, net: nn.Module,
                   device: str) -> Tuple[int, torch.Tensor]:
        """
        Using the given network, decide what action to carry out

        Args:(b
            net: policy network
            device: current device

        Returns:
            action
        """
        obs = torch.from_numpy(self.obs).float().unsqueeze(0)
        if device not in ['cpu']:
            obs = obs.cuda(device)

        action_logit = net(obs).to(device)
        probs = F.softmax(action_logit, dim=1).to(device)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample().item()
        action = int(action)
        return action

    @torch.no_grad()
    def play_step(self,
                  net: nn.Module,
                  device: str = 'cuda:0') -> Tuple[float, bool, torch.Tensor]:
        """
        Carries out a single interaction step between the agent and the
        environment

        Args:
            net: policy network
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            reward, done
        """
        action = self.get_action(net, device)

        # do step in the environment
        new_obs, reward, done, _ = self.env.step(action)
        exp = Experience(self.obs, action, reward, done, new_obs)
        self.replay_buffer.append(exp)

        self.obs = new_obs
        if done:
            self.reset()
        return reward, done


class RLDataset(IterableDataset):
    """
    Iterable Dataset containing the ExperienceBuffer
    which will be updated with new experiences during training

    Args:
        replay_buffer: replay buffer
        sample_size: number of experiences to sample at a time
    """
    def __init__(self,
                 replay_buffer: RolloutCollector,
                 env: gym.Env,
                 agent: Agent,
                 net: MLP,
                 sample_size: int = 200) -> None:
        self.replay_buffer = replay_buffer
        self.sample_size = sample_size
        self.env = env
        self.net = net
        self.agent = agent
        self.device = 'cuda:0'  # need a better way

    def populate(self) -> None:
        """
        Samples an entire episode

        """
        self.replay_buffer.empty_buffer()
        done = False
        while not done:
            reward, done = self.agent.play_step(self.net)

    def __iter__(self):
        for i in range(1):
            self.populate()
            states, actions, rewards, dones, new_states = self.replay_buffer.sample()
            yield (states, actions, rewards, dones, new_states)


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
        log_probs = F.log_softmax(action_logit, dim=1).squeeze(0)[actions]

        discounted_rewards = self.reward_to_go(rewards)
        discounted_rewards = torch.tensor(discounted_rewards).to(self.device)
        advantage = (discounted_rewards - discounted_rewards.mean()) / (
                discounted_rewards.std() + self.eps)

        loss = -advantage * log_probs[range(len(actions)), actions]
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
        dataset = RLDataset(self.replay_buffer, self.env, self.agent, self.net,
                            self.hparams.episode_length)
        dataloader = DataLoader(
            dataset=dataset,
            collate_fn=self.collate_fn,
            batch_size=1,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader"""
        return self.__dataloader()


def main(hparams) -> None:
    seed_everything(41)

    wandb_logger = WandbLogger(project='vpg-lightning-test')
    model = VPGLightning(hparams)

    trainer = pl.Trainer(
        gpus=1,
        # distributed_backend='dp',
        max_epochs=2000,
        reload_dataloaders_every_epoch=False,
        logger=wandb_logger,
    )

    trainer.fit(model)

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.005, help="learning rate")
parser.add_argument("--eps",
                    type=float,
                    default=np.finfo(np.float32).eps.item(),
                    help="small offset")
parser.add_argument("--env",
                    type=str,
                    default="CartPole-v0",
                    help="gym environment tag")
parser.add_argument("--gamma",
                    type=float,
                    default=0.99,
                    help="discount factor")
parser.add_argument("--episode_length",
                    type=int,
                    default=200,
                    help="max length of an episode")
parser.add_argument("--max_episode_reward",
                    type=int,
                    default=200,
                    help="max episode reward in the environment")

args, _ = parser.parse_known_args()

main(args)
