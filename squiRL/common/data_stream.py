"""Contains RL experience buffers

Attributes:
    Experience (namedtuple): An environment step experience
"""
from collections import deque
from collections import namedtuple
from typing import Tuple

import gym
import numpy as np
from torch.utils.data.dataset import IterableDataset
import torch.multiprocessing as mp
import torch

from squiRL.common.policies import MLP

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
    def __init__(self, capacity: int, state_shape: tuple, action_shape: tuple, should_share: bool = False) -> None:
        """Summary

        Args:
            capacity (int): Description
        """

        state_shape = [capacity] + list(state_shape)
        action_shape = [capacity] + list(action_shape)

        self.capacity = capacity
        self.count = torch.tensor([0], dtype=torch.int64)
        self.states = torch.zeros(state_shape, dtype=torch.float32)
        self.actions = torch.zeros(action_shape, dtype=torch.float32)
        self.rewards = torch.zeros((capacity), dtype=torch.float32)
        self.dones = torch.zeros((capacity), dtype=torch.bool)
        self.next_states = torch.zeros(state_shape, dtype=torch.float32)

        if should_share:
            self.count.share_memory_()
            self.states.share_memory_()
            self.actions.share_memory_()
            self.next_states.share_memory_()
            self.rewards.share_memory_()
            self.dones.share_memory_()

        self.lock = mp.Lock()

    def __len__(self) -> int:
        """Calculates length of buffer

        Returns:
            int: Length of buffer
        """
        return self.count.detach().numpy().item()

    def append(self, experience: Experience) -> None:
        """
        Add experience to the buffer

        Args:
            experience (Experience): Tuple (state, action, reward, done,
            last_state)
        """

        with self.lock:
            if self.count[0] < self.capacity:
               self.count[0] += 1

               # count keeps the exact length, but indexing starts from 0 so we decrease by 1
               nr = self.count[0] - 1

               self.states[nr] = torch.tensor(experience.state, dtype=torch.float32)
               self.actions[nr] = torch.tensor(experience.action, dtype=torch.float32)
               self.rewards[nr] = torch.tensor(experience.reward, dtype=torch.float32)
               self.dones[nr] = torch.tensor(experience.done, dtype=torch.bool)
               self.next_states[nr] = torch.tensor(experience.last_state, dtype=torch.float32)

            else:
                exit("RolloutCollector: Buffer is full but samples are being added to it")


    def sample(self) -> Tuple:
        """Sample experience from buffer

        Returns:
            Tuple: Sampled experience
        """

        # count keeps the exact length, but indexing starts from 0 so we decrease by 1
        nr = self.count[0] - 1
        return (self.states[:nr], self.actions[:nr], self.rewards[:nr], self.dones[:nr], self.next_states[:nr])

    def empty_buffer(self) -> None:
        """Empty replay buffer by resetting the count (so old data gets overwritten)
        """
        with self.lock:
            # the [0] is very important, otherwise we throw the tensor out and the int that replaces it won't get shared
            self.count[0] = 0


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
                 agent, episodes_per_batch: int = 1) -> None:
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
        self.episodes_per_batch = episodes_per_batch

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
        for i in range(self.episodes_per_batch):
            self.populate()
            states, actions, rewards, dones, new_states = self.replay_buffer.sample(
            )
            yield (states, actions, rewards, dones, new_states)
