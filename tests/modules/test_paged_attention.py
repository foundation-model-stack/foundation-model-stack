import pytest
import torch
import torch.nn.functional as F


# Default block size for paged attention
_DEFAULT_SPARSE_BLOCK_SIZE = 128

from fms.modules.attention import MultiHeadAttention
from fms.models.llama import LLaMAConfig, LLaMA

def test_paged_attention_basic():
    """Test basic paged attention functionality matches regular attention."""
    batch_size, seq_len, n_heads, head_dim = 2, 512, 8, 64
    
    # Create attention module with paged attention config
    attn = MultiHeadAttention(
        emb_dim=n_heads * head_dim,
        emb_kq=head_dim,
        emb_v=head_dim,
        nheads=n_heads,
        kvheads=n_heads,
        paged_attention_config={"block_size": 128}
    ).cuda()

    # Create random inputs
    q = torch.randn(batch_size, seq_len, n_heads * head_dim, device="cuda", requires_grad=True)
    k = torch.randn(batch_size, seq_len, n_heads * head_dim, device="cuda", requires_grad=True)
    v = torch.randn(batch_size, seq_len, n_heads * head_dim, device="cuda", requires_grad=True)
    
    # Run with paged attention
    out_paged = attn(q, k, v, attn_algorithm="paged")
    
    # Run with regular attention
    out_regular = attn(q, k, v)
    
    # Outputs should be close
    torch.testing.assert_close(out_paged, out_regular, rtol=1e-3, atol=1e-3)

def test_paged_attention_kv_cache():
    """Test paged attention with KV cache."""
    batch_size, seq_len, n_heads, head_dim = 2, 256, 8, 64
    
    attn = MultiHeadAttention(
        emb_dim=n_heads * head_dim,
        emb_kq=head_dim,
        emb_v=head_dim,
        nheads=n_heads,
        kvheads=n_heads,
        paged_attention_config={"block_size": 128}
    ).cuda()

    # Initial sequence
    q1 = torch.randn(batch_size, seq_len, n_heads * head_dim, device="cuda")
    k1 = torch.randn(batch_size, seq_len, n_heads * head_dim, device="cuda")
    v1 = torch.randn(batch_size, seq_len, n_heads * head_dim, device="cuda")
    
    # Next token
    q2 = torch.randn(batch_size, 1, n_heads * head_dim, device="cuda")
    
    # Run with cache
    out1, cache = attn(q1, k1, v1, attn_algorithm="paged", use_cache=True)
    out2, _ = attn(
        q2, 
        past_key_value_state=cache,
        attn_algorithm="paged",
        use_cache=True
    )
    
    # Verify shapes
    assert out1.shape == (batch_size, seq_len, n_heads * head_dim)
    assert out2.shape == (batch_size, 1, n_heads * head_dim)

def test_paged_attention_block_sizes():
    """Test different block sizes."""
    batch_size, seq_len, n_heads, head_dim = 2, 512, 8, 64
    
    # Test power of 2 block sizes
    for block_size in [32, 64, 128]:
        attn = MultiHeadAttention(
            emb_dim=n_heads * head_dim,
            emb_kq=head_dim,
            emb_v=head_dim,
            nheads=n_heads,
            kvheads=n_heads,
            paged_attention_config={"block_size": block_size}
        ).cuda()
        
        q = torch.randn(batch_size, seq_len, n_heads * head_dim, device="cuda")
        k = torch.randn(batch_size, seq_len, n_heads * head_dim, device="cuda")
        v = torch.randn(batch_size, seq_len, n_heads * head_dim, device="cuda")
        
        # Should run without errors
        out = attn(q, k, v, attn_algorithm="paged")
        assert out.shape == (batch_size, seq_len, n_heads * head_dim)
    
    # Test invalid block size
    with pytest.raises(ValueError, match="must be a power of 2"):
        attn = MultiHeadAttention(
            emb_dim=n_heads * head_dim,
            emb_kq=head_dim,
            emb_v=head_dim,
            nheads=n_heads,
            kvheads=n_heads,
            paged_attention_config={"block_size": 100}  # Not power of 2
        ).cuda()
        out = attn(q, k, v, attn_algorithm="paged")

def test_llama_paged_attention():
    """Test paged attention in LLaMA model."""
    config = LLaMAConfig(
        emb_dim=512,
        nheads=8,
        nlayers=2,
        paged_attention=True,
        paged_attention_block_size=128,
        max_blocks_per_sequence=16
    )
    
    model = LLaMA(config).cuda()
    batch_size, seq_len = 2, 512
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device="cuda")
    
    # Regular forward pass
    out1 = model(input_ids)
    
    # Forward pass with explicit paged attention
    out2 = model(input_ids, attn_algorithm="paged")
    
    # Outputs should be close
    torch.testing.assert_close(out1, out2, rtol=1e-3, atol=1e-3)

def _call_flex(attn, q, k, v):
    """
    Helper that forces MultiHeadAttention to take the FlexAttention codepath
    by using attn_algorithm="paged" but without KV caching.
    """
    return attn(q, k, v, attn_algorithm="paged").detach()

def test_paged_attention_memory():
    """Peak‑memory comparison after *incremental decoding* using Flex on both paths."""
    batch_size, seq_len, n_heads, head_dim = 2, 2048, 8, 64
    blk_size = 128

    attn_base = MultiHeadAttention(
        emb_dim=n_heads * head_dim,
        emb_kq=head_dim,
        emb_v=head_dim,
        nheads=n_heads,
        kvheads=n_heads,
        paged_attention_config={"block_size": blk_size}
    ).cuda()

    attn_paged = MultiHeadAttention(
        emb_dim=n_heads * head_dim,
        emb_kq=head_dim,
        emb_v=head_dim,
        nheads=n_heads,
        kvheads=n_heads,
        paged_attention_config={"block_size": blk_size}
    ).cuda()

    # Warm-up so the global KV cache exists
    dummy = torch.randn(batch_size, 1, n_heads * head_dim, device="cuda")
    attn_base(dummy, dummy, dummy, attn_algorithm="paged", use_cache=True)

    # Free the pages we just allocated, but keep the cache tensors resident
    for b in range(batch_size):
        attn_base._paged_mgr.erase(torch.tensor([b], device="cuda"))
        
    # ------------------------------------------------------------------
    # Baseline (Flex, no KV cache)
    # ------------------------------------------------------------------
    q0 = torch.randn(batch_size, seq_len, n_heads * head_dim, device="cuda")
    k0 = torch.randn_like(q0)
    v0 = torch.randn_like(q0)

    # Reset the manager so the baseline starts with a clean page pool sized
    # for the *big* sequence, not the 1‑token warm‑up.
    attn_base._paged_mgr = None
    attn_base._k_cache = None
    attn_base._v_cache = None

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _ = _call_flex(attn_base, q0, k0, v0)                       # step‑0

    # Reset again so that the second baseline call starts with a fresh
    # page table; this prevents capacity from doubling (2048 → 4097).
    attn_base._paged_mgr = None
    attn_base._k_cache = None
    attn_base._v_cache = None

    q_full = torch.randn(batch_size, seq_len + 1, n_heads * head_dim, device="cuda")
    k_full = torch.randn_like(q_full)
    v_full = torch.randn_like(q_full)
    _ = _call_flex(attn_base, q_full, k_full, v_full)
    
    regular_peak = torch.cuda.max_memory_allocated()

    # ------------------------------------------------------------------
    # Paged attention (Flex + KV cache)
    # ------------------------------------------------------------------
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _, cache = attn_paged(q0, k0, v0, attn_algorithm="paged", use_cache=True)  # step‑0
    q1 = torch.randn(batch_size, 1, n_heads * head_dim, device="cuda")
    _ = attn_paged(
        q1,
        attn_algorithm="paged",
        past_key_value_state=cache,
        use_cache=True
    )                                                      # step‑1
    paged_peak = torch.cuda.max_memory_allocated()

    # allow 25 MB head-room
    tolerance = 25 * 2**20   # 25 MiB
    assert paged_peak <= regular_peak + tolerance, (
        f"paged {paged_peak} – regular {regular_peak} exceeds {tolerance}-byte slack"
    )

def test_paged_attention_max_blocks():
    """Test max blocks limit."""
    batch_size, seq_len, n_heads, head_dim = 2, 1024, 8, 64
    block_size = 64
    max_blocks = 8  # Only allow 8 blocks = 512 tokens
    
    attn = MultiHeadAttention(
        emb_dim=n_heads * head_dim,
        emb_kq=head_dim,
        emb_v=head_dim,
        nheads=n_heads,
        kvheads=n_heads,
        paged_attention_config={
            "block_size": block_size,
            "max_blocks": max_blocks
        }
    ).cuda()

    q = torch.randn(batch_size, seq_len, n_heads * head_dim, device="cuda")
    k = torch.randn(batch_size, seq_len, n_heads * head_dim, device="cuda")
    v = torch.randn(batch_size, seq_len, n_heads * head_dim, device="cuda")
    
    # Should raise error due to sequence length exceeding max blocks
    with pytest.raises(ValueError, match="exceeds maximum allowed blocks"):
        out = attn(q, k, v, attn_algorithm="paged") 