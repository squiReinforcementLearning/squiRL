import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from squiRL.common.data_stream import Experience


class Agent:
    """
    Base Agent class handling the interaction with the environment

    Args:
        env: training environment
    """
    def __init__(self, env: gym.Env, replay_buffer) -> None:
        self.env = env
        self.replay_buffer = replay_buffer
        self.reset()

    def reset(self) -> None:
        """ Resets the environment and updates the obs"""
        self.obs = self.env.reset()

    def process_obs(self, obs: int) -> torch.Tensor:
        return torch.from_numpy(obs).float().unsqueeze(0)

    def get_action(
        self,
        net: nn.Module,
    ) -> Tuple[int, torch.Tensor]:
        """
        Using the given network, decide what action to carry out

        Args:(b
            net: policy network

        Returns:
            action
        """
        obs = self.process_obs(self.obs)

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
    ) -> Tuple[float, bool, torch.Tensor]:
        """
        Carries out a single interaction step between the agent and the
        environment

        Args:
            net: policy network
            epsilon: value to determine likelihood of taking a random action

        Returns:
            reward, done
        """
        action = self.get_action(net)

        # do step in the environment
        new_obs, reward, done, _ = self.env.step(action)
        exp = Experience(self.obs, action, reward, done, new_obs)
        self.replay_buffer.append(exp)

        self.obs = new_obs
        if done:
            self.reset()
        return reward, done
