import time

import pytest
import torch

from fms.utils.tensors import ExpandableTensor, PagedTensor


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
    original = expandable

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

    # original should not have changed
    assert original.shape == (4, 5, 8, 7)


# TODO: This is a flaky test we will need to address, most likely making tensors larger
@pytest.mark.skip
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

    result = t
    start = time.time()
    t = ExpandableTensor(t)
    for _ in range(iters):
        result = torch.cat((result, next_val), dim=2)
    expandable_time = time.time() - start
    # requires that expandable tensor be at least 20% faster
    assert expandable_time < (regular_time * 0.8)


def test_contiguous():
    t = torch.randn((5, 5))
    expandable = ExpandableTensor(t, preallocate_length=100)
    expandable = expandable.contiguous()

    torch.testing.assert_close(expandable, t)
    assert expandable.is_contiguous()
    assert type(expandable) != ExpandableTensor


def get_paged_tensor() -> PagedTensor:
    batches = 5
    # default page size is 16. using 15 x 7 ensures un-even paging, to ensure
    # copy works correctly.
    seq_len = 15 * 7
    emb_dim = 4
    t = torch.arange(batches * seq_len * emb_dim).view(batches, seq_len, emb_dim)
    pt = PagedTensor.from_tensor(t, 1)
    return pt


def test_paged():
    # simplest test, the number of items is evenly divisible by page size
    batches = 5
    seq_len = 16 * 7
    emb_dim = 4
    t = torch.arange(batches * seq_len * emb_dim).view(batches, seq_len, emb_dim)
    pt = PagedTensor.from_tensor(t, 1)
    extracted = pt._tensor()
    torch.testing.assert_close(t, extracted)


def test_paged_uneven():
    batches = 5
    # default page size is 16. using 15 x 7 ensures un-even paging, to ensure
    # copy works correctly.
    seq_len = 15 * 7
    emb_dim = 4
    t = torch.arange(batches * seq_len * emb_dim).view(batches, seq_len, emb_dim)
    pt = PagedTensor.from_tensor(t, 1)
    extracted = pt._tensor()
    torch.testing.assert_close(t, extracted)


def test_paged_cleanup():
    pt = get_paged_tensor()
    storage = pt.paged_storage
    del pt
    assert len(storage.refcounts)
    for refcount in storage.refcounts:
        assert refcount == 0


def test_pointwise():
    pt = get_paged_tensor()
    result = torch.add(pt, 4)
    # result = pt.add(4)
    assert isinstance(result, PagedTensor)
    assert result.shape == pt.shape
    assert torch.allclose(pt._tensor() + 4, result._tensor())

    # out of place
    pt += 4
    assert isinstance(pt, PagedTensor)
    assert torch.allclose(pt._tensor(), result._tensor())


def test_expand():
    pt = get_paged_tensor()
    free_pages = pt.paged_storage.free_pages()
    tensor = pt._tensor()
    initial_size = pt.paged_storage.storage.shape
    pt.paged_storage.expand()
    print(tensor.shape)
    assert tensor.shape == pt._tensor().shape
    print(pt.paged_storage.storage.shape, initial_size)
    assert pt.paged_storage.storage.shape[0] > initial_size[0]
    assert pt.paged_storage.free_pages() > free_pages


def test_remove():
    pt = get_paged_tensor()
    free_pages = pt.paged_storage.free_pages()
    tensor = pt._tensor()
    pt.remove(0, 0)
    new_tensor = pt._tensor()

    joined = tensor[1:, :, :]

    assert joined.shape == new_tensor.shape
    assert torch.allclose(joined, new_tensor)
    # make sure the deletion freed up space:
    assert pt.paged_storage.free_pages() > free_pages

    # a last row in batch
    pt = get_paged_tensor()
    free_pages = pt.paged_storage.free_pages()
    tensor = pt._tensor()
    pt.remove(4, 0)
    new_tensor = pt._tensor()
    joined = tensor[:-1, :, :]
    assert joined.shape == new_tensor.shape
    assert torch.allclose(joined, new_tensor)
    # make sure the deletion freed up space:
    assert pt.paged_storage.free_pages() > free_pages

    # an end value on last dynamic dim
    pt = get_paged_tensor()
    free_pages = pt.paged_storage.free_pages()
    tensor = pt._tensor()

    pt.remove(15 * 7 - 1, 1)

    new_tensor = pt._tensor()
    joined = tensor[:, :-1, :]

    assert joined.shape == new_tensor.shape
    assert torch.allclose(joined, new_tensor)

    # if we remove items up to page_size we should free up one page per batch.
    to_remove = 15 * 7 % 16
    last_idx = 15 * 7 - 2
    for idx in range(last_idx, last_idx - to_remove, -1):
        pt.remove(idx, 1)
    # removes one page per batch-dim
    assert free_pages + pt.shape[0] == pt.paged_storage.free_pages()


def test_compile():
    pt = get_paged_tensor()

    def do_stuff(t: PagedTensor):
        t = torch.add(t, 4)
        # t.paged_storage.expand()
        t = torch.mul(t, 2)
        return t

    result1 = do_stuff(pt)
    compiled = torch.compile(do_stuff, fullgraph=True)
    result2 = compiled(pt)

    assert result1.shape == result2.shape
    assert isinstance(result1, PagedTensor)
    assert isinstance(result2, PagedTensor)

    assert torch.allclose(result1._tensor(), result2._tensor())
