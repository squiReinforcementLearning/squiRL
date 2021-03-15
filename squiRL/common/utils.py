import torch
from torch.utils.data._utils import collate
from squiRL.common.data_stream import Experience


def collate_episodes(batch):
    """Manually processes collected batch of experience

    Args:
        batch (TYPE): Current mini batch of replay data

    Returns:
        TYPE: Processed mini batch of replay data
    """
    batch_dict = {
        j: [collate.default_convert(s[i]) for s in batch][0]
        for i, j in enumerate(Experience._fields)
    }

    return batch_dict.values()


def reward_to_go(rewards: torch.Tensor, gamma: float) -> torch.tensor:
    """Calculates reward to go over an entire episode

    Args:
        rewards (torch.Tensor): Episode rewards

    Returns:
        torch.tensor: Reward to go for each episode step
    """
    rewards = rewards.detach().cpu().numpy()
    res = []
    sum_r = 0.0
    for r in reversed(rewards):
        sum_r *= gamma
        sum_r += r
        res.append(sum_r)
    res = torch.tensor(list(reversed(res))).float()
    res = (res - res.mean()) / (res.std() + 1e-8)
    return res
