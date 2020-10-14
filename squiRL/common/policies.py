"""Stores policies used for generating actions
"""
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Simple MLP network

    Args:
        obs_size (int): Observation/obs size of the environment
        n_actions (int): Number of discrete actions available in the
        environment
        layers (int): Size of hidden layers

    Attributes:
        fc (nn.ModuleList): Network architecture definition
        layers_n (int): Description
        layers_size (int): Number of layers
    """
    def __init__(self, obs_size: int, n_actions: int, layers: int = [64, 32]):
        """Initializes MLP policy network

        Args:
            obs_size (int): Observation/obs size of the environment
            n_actions (int): Number of discrete actions available in the
            environment
        """
        super(MLP, self).__init__()
        self.fc = nn.ModuleList()
        self.layers_size = len(layers)
        prev = obs_size
        for n in range(0, self.layers_size):
            self.fc.append(nn.Linear(prev, layers[n]))
            prev = layers[n]
        self.fc.append(nn.Linear(prev, n_actions))

    def forward(self, x):
        """Forward pass through NN

        Args:
            x (torch.Tensor): Env observation/state

        Returns:
            torch.Tensor: Action logit
        """
        x = x.type_as(next(self.parameters()))
        for n in range(self.layers_size):
            x = F.relu(self.fc[n](x))
        x = self.fc[self.layers_size](x)
        return x
