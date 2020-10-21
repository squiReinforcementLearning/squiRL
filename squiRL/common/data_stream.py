"""Contains RL experience buffers

Attributes:
    Experience (namedtuple): An environment step experience
"""
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
        capacity: Size of the buffer

    Attributes:
        capacity (int): Size of the buffer
        replay_buffer (deque): Experience buffer
    """
    def __init__(self, capacity: int) -> None:
        """Summary

        Args:
            capacity (int): Description
        """
        self.capacity = capacity
        self.replay_buffer = deque(maxlen=self.capacity)

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
        states, actions, rewards, firsts, next_states = zip(
            *[self.replay_buffer[i] for i in range(len(self.replay_buffer))])

        return (np.array(states), np.array(actions),
                np.array(rewards, dtype=np.float32),
                np.array(firsts, dtype=np.bool), np.array(next_states))

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
        replay_buffer: Replay buffer
    """
    def __init__(self, replay_buffer: RolloutCollector,
                 episodes_per_batch: int, net: MLP, agent) -> None:
        """Summary

        Args:
            replay_buffer (RolloutCollector): Description
            episodes_per_batch (int): number of episodes per batch
            net (nn.Module): Policy network
            agent (Agent): Agent that interacts with env
        """
        self.replay_buffer = replay_buffer
        # self.episodes_per_batch = episodes_per_batch
        self.steps_per_batch = episodes_per_batch
        self.net = net
        self.agent = agent

    def populate(self) -> None:
        """
        Samples an entire episode

        """
        for _ in range(self.steps_per_batch):
            self.agent.play_step(self.net)

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
