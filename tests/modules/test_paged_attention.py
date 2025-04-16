import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import FlexAttention

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

def test_paged_attention_memory():
    """Test memory efficiency of paged attention."""
    batch_size, seq_len, n_heads, head_dim = 2, 2048, 8, 64
    
    attn = MultiHeadAttention(
        emb_dim=n_heads * head_dim,
        emb_kq=head_dim,
        emb_v=head_dim,
        nheads=n_heads,
        kvheads=n_heads,
        paged_attention_config={"block_size": 128}
    ).cuda()

    q = torch.randn(batch_size, seq_len, n_heads * head_dim, device="cuda")
    k = torch.randn(batch_size, seq_len, n_heads * head_dim, device="cuda")
    v = torch.randn(batch_size, seq_len, n_heads * head_dim, device="cuda")
    
    # Measure memory usage with regular attention
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    out_regular = attn(q, k, v)
    regular_memory = torch.cuda.max_memory_allocated()
    
    # Measure memory usage with paged attention
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    out_paged = attn(q, k, v, attn_algorithm="paged")
    paged_memory = torch.cuda.max_memory_allocated()
    
    # Paged attention should use less memory
    assert paged_memory < regular_memory
    
    # But outputs should still match
    torch.testing.assert_close(out_paged, out_regular, rtol=1e-3, atol=1e-3)

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