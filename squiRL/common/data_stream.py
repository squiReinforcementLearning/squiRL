"""Contains RL experience buffers

Attributes:
    Experience (namedtuple): An environment step experience
"""
import numpy as np
from torch.utils.data.dataset import IterableDataset
from collections import deque
from collections import namedtuple
from squiRL.common.policies import MLP
import gym
from typing import Tuple

Experience = namedtuple('Experience',
                        ('state', 'action', 'reward', 'done', 'last_state'))


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
            experience (Experience): Tuple (state, action, reward, done,
            new_state)
        """
        self.replay_buffer.append(experience)

    def sample(self) -> Tuple:
        """Sample experience from buffer

        Returns:
            Tuple: Sampled experience
        """
        states, actions, rewards, dones, next_states = zip(
            *[self.replay_buffer[i] for i in range(len(self.replay_buffer))])

        return (np.array(states), np.array(actions),
                np.array(rewards, dtype=np.float32),
                np.array(dones, dtype=np.bool), np.array(next_states))

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
        env (gym.Env): OpenAI gym environment
        net (nn.Module): Policy network
        replay_buffer: Replay buffer
    """
    def __init__(self, replay_buffer: RolloutCollector, env: gym.Env, net: MLP,
                 agent) -> None:
        """Summary

        Args:
            replay_buffer (RolloutCollector): Description
            env (gym.Env): OpenAI gym environment
            net (nn.Module): Policy network
            agent (Agent): Agent that interacts with env
        """
        self.replay_buffer = replay_buffer
        self.env = env
        self.net = net
        self.agent = agent

    def populate(self) -> None:
        """
        Samples an entire episode

        """
        self.replay_buffer.empty_buffer()
        done = False
        while not done:
            reward, done = self.agent.play_step(self.net)

    def __iter__(self):
        """Iterates over sampled batch

        Yields:
            Tuple: Sampled experience
        """
        for i in range(1):
            self.populate()
            states, actions, rewards, dones, new_states = self.replay_buffer.sample(
            )
            yield (states, actions, rewards, dones, new_states)
