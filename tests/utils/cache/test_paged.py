import pytest
import torch


construction_test_data = [
    (10, 8, 0, 1),
    (2, 4, 2, 1),
    (10, 8, 0, 2),
    (10, 8, 4, 2),
]


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="must have cuda to run paged attention tests"
)
@pytest.mark.parametrize(
    "num_layers,num_heads,kv_heads,tensor_parallel_size", construction_test_data
)
def test_construction(num_layers, num_heads, kv_heads, tensor_parallel_size):
    from fms.utils.cache.paged import PagedKVCacheManager

    total_num_gpu_blocks = 100
    emb_dim = 100
    block_size = 16
    head_size = emb_dim // num_heads
    kv_cache_manager = PagedKVCacheManager(
        num_layers,
        num_heads,
        emb_dim,
        kv_heads,
        block_size=block_size,
        total_num_gpu_blocks=total_num_gpu_blocks,
        tensor_parallel_size=tensor_parallel_size,
    )

    assert len(kv_cache_manager.cache) == num_layers
    assert len(kv_cache_manager.free_blocks) == total_num_gpu_blocks
    assert len(kv_cache_manager.cbg_map) == 0
    assert kv_cache_manager.unused_keys.qsize() == total_num_gpu_blocks

    assert kv_cache_manager.head_size == head_size
    if kv_heads == 0:
        kv_heads = num_heads
    kv_heads_final = kv_heads // tensor_parallel_size if kv_heads > 1 else kv_heads
    num_heads_final = num_heads // tensor_parallel_size if num_heads > 1 else num_heads
    element_size = torch.tensor([], dtype=kv_cache_manager.dtype).element_size()
    x = block_size // element_size

    assert kv_cache_manager.kv_heads == kv_heads_final
    assert kv_cache_manager.num_heads == num_heads_final
    assert kv_cache_manager.cache[0][0].shape == (
        total_num_gpu_blocks,
        kv_heads_final,
        head_size // x,
        block_size,
        x,
    )
    assert kv_cache_manager.cache[0][1].shape == (
        total_num_gpu_blocks,
        kv_heads_final,
        head_size,
        block_size,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="must have cuda to run paged attention tests"
)
def test_allocate_tokens():
    from fms.utils.cache.paged import PagedKVCacheManager

    total_num_gpu_blocks = 100
    kv_cache_manager = PagedKVCacheManager(
        4, 4, 16, 0, total_num_gpu_blocks=total_num_gpu_blocks
    )

    # test prompt
    # 5 - 1 block
    # 18 - 2 blocks
    # 40 - 3 blocks
    sequence_lengths = [5, 18, 40]
    cache_data = kv_cache_manager.allocate_tokens(sequence_lengths)
    assert len(kv_cache_manager.free_blocks) == total_num_gpu_blocks - 6
    assert not cache_data.is_filled()  # this is the prompt so not yet filled
    assert cache_data.context_lengths is None
    assert cache_data.max_sequence_length == 40
    assert cache_data.sequence_ids == [0, 1, 2]
    block_mapping = torch.tensor(
        [[99, 0, 0], [98, 97, 0], [96, 95, 94]], dtype=torch.int32, device="cuda"
    )
    torch.testing.assert_allclose(cache_data.block_mapping, block_mapping)
    slot_mapping = torch.tensor(
        [
            [
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                1584,
                1585,
                1586,
                1587,
                1588,
            ],
            [
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                1568,
                1569,
                1570,
                1571,
                1572,
                1573,
                1574,
                1575,
                1576,
                1577,
                1578,
                1579,
                1580,
                1581,
                1582,
                1583,
                1552,
                1553,
            ],
            [
                1536,
                1537,
                1538,
                1539,
                1540,
                1541,
                1542,
                1543,
                1544,
                1545,
                1546,
                1547,
                1548,
                1549,
                1550,
                1551,
                1520,
                1521,
                1522,
                1523,
                1524,
                1525,
                1526,
                1527,
                1528,
                1529,
                1530,
                1531,
                1532,
                1533,
                1534,
                1535,
                1504,
                1505,
                1506,
                1507,
                1508,
                1509,
                1510,
                1511,
            ],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    torch.testing.assert_allclose(cache_data.slot_mapping, slot_mapping)
    position_ids = []
    for sequence_length in sequence_lengths:
        positions = [0 for _ in range(max(sequence_lengths) - sequence_length)] + [
            i for i in range(sequence_length)
        ]
        position_ids.append(positions)
    torch.testing.assert_allclose(
        cache_data.compute_position_ids(sequence_lengths),
        torch.tensor(position_ids, dtype=torch.long, device="cuda"),
    )

    # test generated tokens
    num_tokens_per_sequence = [12, 1, 1]
    cache_data = kv_cache_manager.allocate_tokens(
        num_tokens_per_sequence, sequence_ids=cache_data.sequence_ids
    )
    assert len(kv_cache_manager.free_blocks) == total_num_gpu_blocks - 7
    assert cache_data.is_filled()  # this is the prompt so not yet filled
    context_lengths = [l + r for l, r in zip(sequence_lengths, num_tokens_per_sequence)]
    assert torch.allclose(
        cache_data.context_lengths,
        torch.tensor(context_lengths, dtype=torch.int32, device="cuda"),
    )
    assert cache_data.max_sequence_length == 41
    assert cache_data.sequence_ids == [0, 1, 2]
    block_mapping = torch.tensor(
        [[99, 93, 0], [98, 97, 0], [96, 95, 94]], dtype=torch.int32, device="cuda"
    )
    torch.testing.assert_allclose(cache_data.block_mapping, block_mapping)
    slot_mapping = torch.tensor(
        [
            [1589, 1590, 1591, 1592, 1593, 1594, 1595, 1596, 1597, 1598, 1599, 1488],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1554],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1512],
        ],
        dtype=torch.int32,
        device="cuda:0",
    )
    torch.testing.assert_allclose(cache_data.slot_mapping, slot_mapping)
    position_ids = []
    for n_i, num_tokens in enumerate(num_tokens_per_sequence):
        positions = [0 for _ in range(max(num_tokens_per_sequence) - num_tokens)] + [
            sequence_lengths[n_i] + i for i in range(num_tokens)
        ]
        position_ids.append(positions)
    torch.testing.assert_allclose(
        cache_data.compute_position_ids(num_tokens_per_sequence),
        torch.tensor(position_ids, dtype=torch.long, device="cuda"),
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="must have cuda to run paged attention tests"
)
def test_free_sequences():
    from fms.utils.cache.paged import PagedKVCacheManager

    total_num_gpu_blocks = 100
    sequence_lengths = [5, 18, 40]
    kv_cache_manager = PagedKVCacheManager(
        4, 4, 16, 0, total_num_gpu_blocks=total_num_gpu_blocks
    )
    assert len(kv_cache_manager.free_blocks) == total_num_gpu_blocks
    assert kv_cache_manager.unused_keys.qsize() == total_num_gpu_blocks
    assert len(kv_cache_manager.cbg_map) == 0

    # test prompt
    # 5 - 1 block
    # 18 - 2 blocks
    # 40 - 3 blocks
    cache_data = kv_cache_manager.allocate_tokens(sequence_lengths)
    assert len(kv_cache_manager.free_blocks) == total_num_gpu_blocks - 6
    assert kv_cache_manager.unused_keys.qsize() == total_num_gpu_blocks - len(
        sequence_lengths
    )
    assert len(kv_cache_manager.cbg_map) == len(sequence_lengths)

    kv_cache_manager.free_sequences(cache_data.sequence_ids)
    assert len(kv_cache_manager.free_blocks) == total_num_gpu_blocks
    assert kv_cache_manager.unused_keys.qsize() == total_num_gpu_blocks
    assert len(kv_cache_manager.cbg_map) == 0
