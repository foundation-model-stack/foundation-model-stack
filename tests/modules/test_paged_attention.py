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


def test_paged_attention_memory_flash():
    """
    Peak‑memory comparison that mirrors real decoding:
      • baseline uses Flash/mem‑efficient SDP (no KV cache)
      • paged path reuses its KV cache
    We measure only the incremental step (prompt peak is discarded) so the
    scratch buffer is identical and any difference comes from extra working
    memory.  Paged should win by at least 10 MB.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for peak‑memory test")

    batch_size, seq_len, n_heads, head_dim = 2, 2048, 8, 64
    blk_size = 128
    atol_mb = 10

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

    # ------------------------------------------------------------------
    # 1. Run the prompt once on both modules (not measured)
    # ------------------------------------------------------------------
    prompt_q = torch.randn(batch_size, seq_len, n_heads * head_dim, device="cuda")
    prompt_k = torch.randn_like(prompt_q)
    prompt_v = torch.randn_like(prompt_q)

    _ = attn_base(prompt_q, prompt_k, prompt_v, attn_algorithm="math")  # baseline prompt
    _, cache = attn_paged(prompt_q, prompt_k, prompt_v,
                          attn_algorithm="paged", use_cache=True)        # paged prompt

    # ------------------------------------------------------------------
    # 2. Measure peak memory for the incremental step only
    # ------------------------------------------------------------------
    torch.cuda.reset_peak_memory_stats()
    incr_q = torch.randn(batch_size, 1, n_heads * head_dim, device="cuda")
    incr_k = torch.randn_like(incr_q)
    incr_v = torch.randn_like(incr_q)

    # Baseline must feed full prompt +1 again
    full_q = torch.cat([prompt_q, incr_q], dim=1)
    full_k = torch.cat([prompt_k, incr_k], dim=1)
    full_v = torch.cat([prompt_v, incr_v], dim=1)
    _ = attn_base(full_q, full_k, full_v, attn_algorithm="math")
    baseline_peak = torch.cuda.max_memory_allocated()

    torch.cuda.reset_peak_memory_stats()
    _ = attn_paged(
        incr_q,
        attn_algorithm="paged",
        past_key_value_state=cache,
        use_cache=True
    )
    paged_peak = torch.cuda.max_memory_allocated()

    diff_mb = (baseline_peak - paged_peak) / 2**20
    assert paged_peak + atol_mb * 2**20 < baseline_peak, (
        f"Paged should peak at least {atol_mb} MB lower than baseline "
        f"but baseline={baseline_peak/2**20:.1f} MB paged={paged_peak/2**20:.1f} MB "
        f"(diff={diff_mb:.1f} MB)"
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