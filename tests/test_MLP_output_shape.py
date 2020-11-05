from squiRL.common.policies import MLP
import torch
import gym3


def test_MLP():
    n_envs = 5
    env = gym3.vectorize_gym(num=n_envs, env_kwargs={"id": "CartPole-v0"})
    obs_size = env.ob_space.size
    n_actions = env.ac_space.eltype.n

    net = MLP(obs_size, n_actions)
    _, obs, _ = env.observe()
    obs = torch.from_numpy(obs).float().unsqueeze(0)
    output = net(obs)

    assert output.shape == torch.Size(
        [1, n_envs,
         n_actions]), "Action shape is not equal to env action space size."
