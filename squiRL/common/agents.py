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
        probs = F.softmax(action_logit, dim=-1).to(device)
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
