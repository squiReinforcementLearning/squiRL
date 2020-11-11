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
