import pytest
import time
import torch

from fms.utils.tensors import ExpandableTensor


def test_expandable_tensor():
    t = torch.randn((4, 5, 8, 7))
    expandable = ExpandableTensor(t, 2, 10)

    to_append = torch.randn((4, 5, 3, 7))

    expected = t
    # 8 + 3*4 == 20
    for i in range(4):
        expected = torch.cat((expected, to_append), dim=2)
        expandable = expandable._append(to_append)

    torch.testing.assert_close(expandable, expected)

    # the "preallocated length" began at 10, then would have expanded 1x and is now exactly full.
    assert expandable._underlying_tensor.shape[2] == 20
    # expand again doubles it
    expandable = expandable._append(to_append)
    assert expandable._underlying_tensor.shape[2] == 40


def test_expandable_tensor_ops():
    t = torch.randn((4, 5, 8, 7))
    expandable = ExpandableTensor(t, 2, 10)
    # validate an arbitrary tensor operation
    expandable.add_(7)
    t.add_(7)
    torch.testing.assert_close(expandable, t)
    # in place op preserves type, no copy
    assert type(expandable) == ExpandableTensor


def test_cat():
    t = torch.randn((4, 5, 8, 7))
    expandable = ExpandableTensor(t, 2, 10)

    to_append = torch.randn((4, 5, 3, 7))

    expected = t
    # 8 + 3*4 == 20
    for i in range(4):
        expected = torch.cat((expected, to_append), dim=2)
        expandable = torch.cat((expandable, to_append), dim=2)

    torch.testing.assert_close(expandable, expected)
    assert type(expected) != ExpandableTensor
    assert type(expandable) == ExpandableTensor
    assert expandable._underlying_tensor.shape[2] == 20


def test_perf():
    iters = 100
    # not a serious benchmark but confirm that behavior correctly offers
    # some perf benefit, otherwise implementation must be incorrect.

    t = torch.randn((10, 1000, 500))
    next_val = torch.randn((10, 1000, 1))
    result = t
    start = time.time()
    for _ in range(iters):
        result = torch.cat((result, next_val), dim=2)
    regular_time = time.time() - start

    t = ExpandableTensor(t)
    result = t
    start = time.time()
    for _ in range(iters):
        result = torch.cat((result, next_val), dim=2)
    expandable_time = time.time() - start
    # requires that expandable tensor be at least 2x faster
    assert expandable_time < (regular_time / 2)
