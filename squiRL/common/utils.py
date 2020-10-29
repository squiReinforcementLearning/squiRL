import numpy as np
import torch
from torch.utils.data._utils import collate


def collate_episodes(batch):

    """Manually processes collected batch of experience

    Args:
        batch (TYPE): Current mini batch of replay data

    Returns:
        TYPE: Processed mini batch of replay data
    """
    # extract from batch
    states = [s[0].squeeze() for s in batch][0]
    actions = [s[1].squeeze() for s in batch][0]
    rewards = [s[2].squeeze() for s in batch][0]
    firsts = [s[3].squeeze() for s in batch][0]
    next_states = [s[4].squeeze() for s in batch][0]

    # remove streams with no full episodes
    counts = [np.count_nonzero(firsts[:, i]) for i in range(firsts.shape[1])]
    no_episode_streams = np.where(np.array(counts) <= 1)[0].tolist()
    for j, i in enumerate(no_episode_streams):
        i = i - j
        states = np.delete(states, i, 1)
        actions = np.delete(actions, i, 1)
        rewards = np.delete(rewards, i, 1)
        firsts = np.delete(firsts, i, 1)
        next_states = np.delete(next_states, i, 1)

    # removes incomplete episodes from remaining streams
    inds = [np.nonzero(firsts[:, i]) for i in range(firsts.shape[1])]
    cleaned_states = []
    cleaned_actions = []
    cleaned_rewards = []
    cleaned_firsts = []
    cleaned_next_states = []
    for j, i in enumerate(inds):
        i = i[0]
        c_firsts = np.delete(firsts[:, j], np.s_[:i[0]], 0)
        c_firsts = np.delete(c_firsts, np.s_[i[-1] - i[0]:], 0)
        cleaned_firsts.append(c_firsts)
        c_states = np.delete(states[:, j], np.s_[:i[0]], 0)
        c_states = np.delete(c_states, np.s_[i[-1] - i[0]:], 0)
        cleaned_states.append(c_states)
        c_actions = np.delete(actions[:, j], np.s_[:i[0]], 0)
        c_actions = np.delete(c_actions, np.s_[i[-1] - i[0]:], 0)
        cleaned_actions.append(c_actions)
        c_rewards = np.delete(rewards[:, j], np.s_[:i[0]], 0)
        c_rewards = np.delete(c_rewards, np.s_[i[-1] - i[0]:], 0)
        cleaned_rewards.append(c_rewards)
        c_next_states = np.delete(next_states[:, j], np.s_[:i[0]], 0)
        c_next_states = np.delete(c_next_states, np.s_[i[-1] - i[0]:], 0)
        cleaned_next_states.append(c_next_states)

    # convert to torch tensors
    states = torch.cat(collate.default_convert(cleaned_states))
    actions = torch.cat(collate.default_convert(cleaned_actions))
    rewards = torch.cat(collate.default_convert(cleaned_rewards))
    firsts = torch.cat(collate.default_convert(cleaned_firsts))
    next_states = torch.cat(collate.default_convert(cleaned_next_states))

    return states, actions, rewards, firsts, next_states
