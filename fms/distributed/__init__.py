import torch


def rank_and_world(group=None):
    """
    Returns (rank, world_size) from the optionally-specified group, otherwise
    from the default group, or if non-distributed just returns (0, 1)
    """
    if torch.distributed.is_initialized() and group is None:
        group = torch.distributed.GroupMember.WORLD

    if group is None:
        world_size = 1
        rank = 0
    else:
        world_size = group.size()
        rank = group.rank()

    return rank, world_size
