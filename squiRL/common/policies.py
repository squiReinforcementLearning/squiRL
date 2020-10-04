import torch.nn as nn
import torch.nn.functional as F


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
