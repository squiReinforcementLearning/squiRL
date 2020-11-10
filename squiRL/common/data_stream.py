"""Contains RL experience buffers

Attributes:
    Experience (namedtuple): An environment step experience
"""
import random
import numpy as np
from torch.utils.data.dataset import IterableDataset
from collections import deque
from collections import namedtuple
from squiRL.common.policies import MLP
from typing import Tuple

Experience = namedtuple('Experience',
                        ('state', 'action', 'reward', 'first', 'next_state'))


class RolloutCollector:
    """
    Buffer for collecting rollout experiences allowing the agent to learn from
    them
    Args:
        episodes_per_batch (int): number of episodes per batch

    Attributes:
        replay_buffer (deque): Experience buffer
    """
    def __init__(self, episodes_per_batch: int) -> None:
        """Stores rollout data collected by agents.

        Args:
        """
        self.replay_buffer = deque()
        self.episodes_per_batch = episodes_per_batch

    def __len__(self) -> int:
        """Calculates length of buffer

        Returns:
            int: Length of buffer
        """
        return len(self.replay_buffer)

    def append(self, experience: Experience) -> None:
        """
        Add experience to the buffer

        Args:
            experience (Experience): Tuple (state, action, reward, first,
            new_state)
        """
        self.replay_buffer.append(experience)

    def sample(self) -> Tuple:
        """Sample experience from buffer

        Returns:
            Tuple: Sampled experience
        """
        data = random.sample(self.replay_buffer, self.episodes_per_batch)
        data = {
            k: [s[i] for s in data]
            for i, k in enumerate(Experience._fields)
        }
        data = {k: [np.array(i) for i in v] for k, v in data.items()}

        return data.values()

    def empty_buffer(self) -> None:
        """Empty replay buffer
        """
        self.replay_buffer.clear()


class RLDataset(IterableDataset):
    """
    Iterable Dataset containing the ExperienceBuffer
    which will be updated with new experiences during training

    Args:
        replay_buffer: Replay buffer
        sample_size: Number of experiences to sample at a time

    Attributes:
        agent (Agent): Agent that interacts with env
        episodes_per_batch (int): number of episodes per batch
        net (nn.Module): Policy network
        num_envs (int): Number of vectorized envs
        replay_buffer: Replay buffer
    """
    def __init__(self, replay_buffer: RolloutCollector,
                 episodes_per_batch: int, net: MLP, agent,
                 num_envs: int) -> None:
        """Summary

        Args:
            replay_buffer (RolloutCollector): Description
            episodes_per_batch (int): number of episodes per batch
            net (nn.Module): Policy network
            num_envs (int): Number of vectorized envs
            agent (Agent): Agent that interacts with env
        """
        self.replay_buffer = replay_buffer
        self.episodes_per_batch = episodes_per_batch
        self.net = net
        self.agent = agent
        self.num_envs = num_envs

    def populate(self) -> None:
        """
        Samples n entire episodes using m vectorized envs

        """
        self.total_episodes_sampled = np.zeros([self.num_envs])
        self.agent.reset_all()
        while self.total_episodes_sampled.sum(
        ) < self.episodes_per_batch or np.count_nonzero(
                self.total_episodes_sampled == 0) > 0:
            firsts = self.agent.play_step(self.net)
            self.total_episodes_sampled += firsts

    def __iter__(self):
        """Iterates over sampled batch

        Yields:
            Tuple: Sampled experience
        """
        self.populate()
        states, actions, rewards, firsts, new_states = self.replay_buffer.sample(
        )
        yield (states, actions, rewards, firsts, new_states)
        self.replay_buffer.empty_buffer()
