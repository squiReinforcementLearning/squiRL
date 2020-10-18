"""Base agent class which handles interacting with the environment for
generating experience
"""
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
from squiRL.common.data_stream import Experience


class Agent:
    """
    Base Agent class handling the interaction with the environment

    Args:
        env: training environment

    Attributes:
        env (List[gym.Env]): List of OpenAI gym training environment
        obs (int): Array of env observation state
        replay_buffer (TYPE): Data collector for saving experience
    """
    def __init__(self, env: List[gym.Env], replay_buffer) -> None:
        """Initializes agent class

        Args:
            env (List[gym.Env]): List of OpenAI gym training environment
            replay_buffer (TYPE): Data collector for saving experience
        """
        self.envs = env
        self.obs = [None] * len(self.envs)
        self.replay_buffer = replay_buffer

        for i in range(len(self.envs)):
            self.env_idx = i
            self.reset()

        self.env_idx = 0

    def reset(self) -> None:
        """Resets the environment and updates the obs
        """
        self.obs[self.env_idx] = self.envs[self.env_idx].reset()

    def next_env(self) -> None:
        self.env_idx = (self.env_idx + 1) % len(self.envs)

    def process_obs(self, obs: int) -> torch.Tensor:
        """Converts obs np.array to torch.Tensor for passing through NN

        Args:
            obs (int): Array of env observation state

        Returns:
            torch.Tensor: Torch tensor of observation
        """
        return torch.from_numpy(obs).float().unsqueeze(0)

    def get_action(
        self,
        net: nn.Module,
    ) -> int:
        """
        Using the given network, decide what action to carry out

        Args:
            net (nn.Module): Policy network

        Returns:
            action (int): Action to be carried out
        """
        obs = self.obs[self.env_idx]
        assert obs is not None
        obs = self.process_obs(obs)

        action_logit = net(obs)
        probs = F.softmax(action_logit, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample().item()
        action = int(action)
        return action

    @torch.no_grad()
    def play_step(
        self,
        net: nn.Module,
    ) -> Tuple[float, bool]:
        """
        Carries out a single interaction step between the agent and the
        environment

        Args:
            net (nn.Module): Policy network

        Returns:
            reward (float): Reward received due to taken step
            done (bool): Indicates if a step is terminal
        """
        action = self.get_action(net)

        # do step in the environment
        new_obs, reward, done, _ = self.envs[self.env_idx].step(action)
        exp = Experience(self.obs[self.env_idx], action, reward, done, new_obs)
        self.replay_buffer.append(exp)

        self.obs[self.env_idx] = new_obs
        if done:
            self.reset()
        return reward, done
