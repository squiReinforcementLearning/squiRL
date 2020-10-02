from squiRL.vpg import MLP
import torch
import gym


def test_MLP():
    env = gym.make("CartPole-v0")
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    net = MLP(obs_size, n_actions)
    obs = torch.from_numpy(env.reset()).float().unsqueeze(0)
    output = net(obs)

    assert output.shape == torch.Size(
        [1, n_actions]), "Action shape is not equal to env action space size."
