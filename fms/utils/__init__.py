import os
from typing import Optional

from torch.distributed import ProcessGroup


def print0(*args, group: Optional[ProcessGroup] = None):
    """
    Print *args to stdout on rank 0 of the default process group, or an
    optionally specified process group.
    """
    if group is not None:
        rank = group.rank()
    else:
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    if rank == 0:
        print(*args)


def has_package(name):
    """
    Checks if a package is installed and available.
    """
    try:
        __import__(name)
    except ImportError:
        return False
    else:
        return True
