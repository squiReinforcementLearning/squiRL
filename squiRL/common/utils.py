import numpy as np
import torch
from torch.utils.data._utils import collate
from squiRL.common.data_stream import Experience
from collections import defaultdict


def collate_episodes(batch):
    """Manually processes collected batch of experience

    Args:
        batch (TYPE): Current mini batch of replay data

    Returns:
        TYPE: Processed mini batch of replay data
    """

    # extract from batch
    batch_dict = {}
    cleaned_batch_dict = defaultdict(list)
    for i, j in enumerate(Experience._fields):
        batch_dict[j] = [s[i].squeeze() for s in batch][0]
        cleaned_batch_dict[j] = []

    # remove streams with no full episodes
    counts = [
        np.count_nonzero(batch_dict['first'][:, i])
        for i in range(batch_dict['first'].shape[1])
    ]
    no_episode_streams = np.where(np.array(counts) <= 1)[0].tolist()
    for j, i in enumerate(no_episode_streams):
        i = i - j
        batch_dict = {k: np.delete(v, i, 1) for k, v in batch_dict.items()}

    # remove incomplete episodes from remaining streams
    inds = [
        np.nonzero(batch_dict['first'][:, i])
        for i in range(batch_dict['first'].shape[1])
    ]

    for j, i in enumerate(inds):
        i = i[0]
        c_all = {
            k: np.delete(v[:, j], np.s_[:i[0]], 0)
            for k, v in batch_dict.items()
        }
        c_all = {
            k: np.delete(v, np.s_[i[-1] - i[0]:], 0)
            for k, v in c_all.items()
        }
        for k, v in c_all.items():
            cleaned_batch_dict[k].append(v)

    # convert to torch tensors
    cleaned_batch_dict = {
        k: torch.cat(collate.default_convert(v))
        for k, v in cleaned_batch_dict.items()
    }
    return cleaned_batch_dict.values()
