import math
import os

from torch.distributed import ProcessGroup


def print0(*args, group: ProcessGroup = None):
    """
    Print *args to stdout on rank 0 of the default process group, or an
    optionally specified process group.
    """
    if group is not None:
        rank = group.rank()
    else:
        rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
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


def smallest_power_greater_than(greater_than_num: int) -> int:
    """
    Gets the smallest integer power of 2 strictly greater than a number

    Parameters
    ----------
    greater_than_num: int
        the number that the power computation must be strictly greater than

    Returns
    -------
    int
        smallest integer power strictly greater than the given greater_than_num
    """
    return int(math.pow(2, int(math.log2(greater_than_num)) + 1))
