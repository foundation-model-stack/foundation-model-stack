import torch

from fms.modules.positions import PositionEncoder
from fms.modules.layernorm import LayerNormParameterized
from torch import Tensor, nn
from torch.nn import functional as F
from typing import List, Optional, Tuple


def scan(state, g):
    # state: b n d h
    # g: 1/b n h h
    state = state.clone()
    g = g.clone()
    s = state.size()
    g0 = g.size(0)
    logl = s[1].bit_length() - 1
    # Up sweep: create ruler ticks
    for i in range(logl):
        span = 2**(i+1)
        state = state.view(s[0], -1, span, *s[2:])  # b -1 span d h
        g = g.view(g0, -1, span, s[3], s[3])  # 1 -1 span h h
        newstate = state[:,:,span//2-1].matmul(g[:,:,-1])
        newgate = g[:,:,span//2-1].matmul(g[:,:,-1])
        state[:,:,-1] += newstate
        g[:,:,-1] = newgate
        
    # Down sweep: fill in blanks
    state = state.view(*s)
    g = g.view(g0, s[1], s[3], s[3])
    state = nn.functional.pad(state, (0,0,0,0,1,0))
    g = nn.functional.pad(g, (0,0,0,0,1,0))[:,:-1]
    remainder = state[:,-1:]
    state = state[:,:-1]
    for i in range(logl-1):
        span = 2**(logl-i-1)
        state = state.view(s[0], -1, span, *s[2:])  # b -1 span d h
        g = g.view(g0, -1, span, s[3], s[3])  # b -1 span h h
        state[:,:,span//2] = state[:,:,span//2] + state[:,:,0].matmul(g[:,:,span//2])
        g[:,:,span//2] = g[:,:,0].matmul(g[:,:,span//2])
    state = torch.cat([state.view(*s)[:,1:], remainder], dim=1)
    return state


class MatScan(torch.autograd.Function):
    @staticmethod
    def forward(state, gate):
        return scan(state, gate)

    @staticmethod
    def setup_context(ctx, inputs, output):
        state, gate = inputs
        ctx.save_for_backward(gate)

    @staticmethod
    def backward(ctx, grad):
        gate = ctx.saved_tensors[0]

        # Gate-accumulate grads
        gflip = gate.flip([1]).transpose(2,3)
        gatesum = scan(grad.flip([1]), gflip.roll(1, dims=1)).flip([1])  # b n d h

        return gatesum, None


class SlidingWindowAttention(nn.Module):
    def __init__(
        self,
        emb_dim,
        emb_kq,
        emb_v,
        nheads,
        kvheads,
        p_dropout=None,
        use_bias=False,
        position_encoder: Optional[PositionEncoder] = None,
    ):
        super(SlidingWindowAttention, self).__init__()
        self.nheads = nheads
        self.kvheads = kvheads
        self.emb_dim = emb_dim
        self.emb_kq_per_head = emb_kq
        self.emb_v_per_head = emb_v
        self.p_dropout = p_dropout if p_dropout is not None else 0.0
        self.use_bias = use_bias
        self.query = nn.Linear(
            self.emb_dim, self.nheads * self.emb_kq_per_head, bias=use_bias
        )
        self.key = nn.Linear(
            self.emb_dim, self.kvheads * self.emb_kq_per_head, bias=use_bias
        )
        self.value = nn.Linear(
            self.emb_dim, self.kvheads * self.emb_v_per_head, bias=use_bias
        )
        self.dense = nn.Linear(
            self.nheads * self.emb_v_per_head, self.emb_dim, bias=use_bias
        )
        if self.p_dropout:
            self.attn_dropout = nn.Dropout(self.p_dropout)
        self.position_encoder = position_encoder
        # Avoiding graph breaks
        self.previous_flash: bool = torch.backends.cuda.flash_sdp_enabled()
        self.previous_mem_efficient: bool = (
            torch.backends.cuda.mem_efficient_sdp_enabled()
        )
        self.previous_math: bool = torch.backends.cuda.math_sdp_enabled()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
                if self.use_bias:
                    m.bias.data.zero_()

    def forward(
        self,
        q,
        k,
        v,
        mask: Optional[Tensor] = None,
        position_ids=None,
        attn_algorithm=None,
        past_key_value_state: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache=False,
        is_self=True,
        is_causal_mask=False,
    ):
        """
        past_key_value_state: tuple
            the cache to be used in attention of the form (<self/cross>_key, <self/cross>_value)
        position_ids: Optional[torch.LongTensor]
            The position of each of the tokens encoded in q and k. Used for RoPE embeddings
        use_cache: bool
            if True, the kv states for self/cross attention will be saved, otherwise they will not be saved
        is_self: bool
            if True, this will perform self attention, otherwise this will perform cross attention. Note: This will
            only be used in the case that use_cache=True. This may be removed in future

        Returns
        -------
        tensor or tuple
            If use_cache=False, only the hidden state will be returned as a tensor. If use_cache=True, a tuple will be
            returned in the form (hidden_state, cache) where hidden_state is a tensor and cache is of the form specified
            in past_key_value_state
        """

        # q, k, v: batch_size x seq_len x emb_dim
        # mask: batch_size x seq_len x seq_len
        batch_size, q_len, _ = q.size()
        kv_len = k.size(1)

        # split emb_dim as nheads*emb_dim_per_head
        # b x h x qlen x ds
        queries = self.query(q).view(
            batch_size, q_len, self.nheads, self.emb_kq_per_head
        )

        # if this is self attention, we always recompute
        # cross attention only gets computed when a cache does not exist
        # if we dont have the cache yet, we need to compute
        # d x (h x ds)
        # b x kvlen x d
        # b x kvlen x h x ds
        # b x h x kvlen x ds
        if is_self or past_key_value_state is None:
            keys = self.key(k).view(
                batch_size, kv_len, self.kvheads, self.emb_kq_per_head
            )

            values = self.value(v).view(
                batch_size, kv_len, self.kvheads, self.emb_v_per_head
            )

            # You want to apply rotary embeddings pre-cache
            if self.position_encoder is not None:
                queries, keys = self.position_encoder.adjusted_qk(
                    queries, keys, position_ids, past_key_value_state, use_cache
                )

        queries = queries.transpose(2, 1) / (self.emb_kq_per_head**(1/4))
        keys = keys.transpose(2, 1) / (self.emb_kq_per_head**(1/4))
        values = values.transpose(2, 1)  # compatible with QK.T

        # if you want to use caching and past_key_value_state is not None meaning you have values in your cache
        if use_cache and past_key_value_state is not None:
            if is_self:
                keys = torch.cat((past_key_value_state[0], keys), dim=2)
                values = torch.cat((past_key_value_state[1], values), dim=2)
            else:
                keys = past_key_value_state[0]
                values = past_key_value_state[1]

        # Merge rel pos bias and mask into single float mask
        if mask is not None:
            # Our expected mask format is bs x q_len x k_len, so to make it broadcastable
            # we need to create the nheads dimension
            while len(mask.size()) != 4:  # expects bs (x nheads) x q_len x kv_len
                mask = mask.unsqueeze(1)

        if self.position_encoder is not None:
            attn_mask: Optional[Tensor] = self.position_encoder.adjusted_mask(
                mask, queries, keys, past_key_value_state, use_cache
            )
        else:
            attn_mask = mask

        # Expand kv so black-box attn will work
        expansion = self.nheads // self.kvheads
        # k/v: b h l d
        if expansion != 1:
            keys_e = keys.unsqueeze(2).expand(-1, -1, expansion, -1, -1).flatten(1, 2)
            values_e = (
                values.unsqueeze(2).expand(-1, -1, expansion, -1, -1).flatten(1, 2)
            )
        else:
            keys_e = keys
            values_e = values

        qk = queries.matmul(keys_e.transpose(2,3))
        m = torch.ones(qk.size(2), qk.size(3), device=qk.device, dtype=qk.dtype).tril()
        m = m - torch.ones_like(m).tril(diagonal=-32)
        m = m.log()
        qk = qk.add(m).softmax(3)  # b h l l
        attn = qk.matmul(values_e)  # b h l d

        # attn = F.scaled_dot_product_attention(
        #     queries,
        #     keys_e,
        #     values_e,
        #     attn_mask=attn_mask,
        #     dropout_p=self.p_dropout if self.training else 0.0,
        #     is_causal=is_causal_mask,
        # )
        
        # attn: bs x seq_len x nheads*emb_v_per_head
        # attn: b x h x qlen x ds
        # attn after permute: b x qlen x h x ds
        # b x qlen x (d)
        attn = (
            attn.transpose(2, 1)
            .contiguous()
            .view(batch_size, q_len, self.nheads * self.emb_v_per_head)
        )
        out = self.dense(attn)

        # if use_cache=True, we return the hidden_state as well as the kv cache
        if use_cache:
            return out, (keys, values)
        else:
            return out
        

class ScanCacheAttention(nn.Module):
    def __init__(
        self,
        emb_dim,
        emb_kq,
        emb_v,
        nheads,
        kvheads,
        p_dropout=None,
        use_bias=False,
        position_encoder: Optional[PositionEncoder] = None,
    ):
        super(ScanCacheAttention, self).__init__()
        self.nheads = nheads
        self.kvheads = kvheads
        self.emb_dim = emb_dim
        self.emb_kq_per_head = emb_kq
        self.emb_v_per_head = emb_v
        self.p_dropout = p_dropout if p_dropout is not None else 0.0
        self.use_bias = use_bias
        self.query = nn.Linear(
            self.emb_dim, self.nheads * self.emb_kq_per_head, bias=use_bias
        )
        self.key = nn.Linear(
            self.emb_dim, self.kvheads * self.emb_kq_per_head, bias=use_bias
        )
        self.value = nn.Linear(
            self.emb_dim, self.kvheads * self.emb_v_per_head, bias=use_bias
        )
        self.dense = nn.Linear(
            self.nheads * self.emb_v_per_head, self.emb_dim, bias=use_bias
        )
        if self.p_dropout:
            self.attn_dropout = nn.Dropout(self.p_dropout)
        self.position_encoder = position_encoder
        # Avoiding graph breaks
        self.previous_flash: bool = torch.backends.cuda.flash_sdp_enabled()
        self.previous_mem_efficient: bool = (
            torch.backends.cuda.mem_efficient_sdp_enabled()
        )
        self.previous_math: bool = torch.backends.cuda.math_sdp_enabled()

        gates = self.make_gates()
        # g = torch.tensor([1 - 2**(-i/32*5-1) for i in range(32)])
        # gates = torch.diag(g)
        # for i in range(1, g.size(0)):
        #     gates[i-1,i] = (1-g[i])
        
        self.register_buffer("gates", gates)
        self.scan = MatScan.apply
        self.ln_k = LayerNormParameterized(emb_kq, use_high_precision_pow=True)
        self.ln_v = LayerNormParameterized(emb_v, use_high_precision_pow=True)
        self.key_pos = nn.Parameter(torch.zeros(emb_v, 32))
        self.query_pos = nn.Parameter(torch.zeros(emb_kq))


    def make_gates(self):
        n = 1024  # Roughly, total cache window length (actually somewhat smaller)
        f = 4  # Repetitions of each power of 2 per entry
        interval = sum([torch.arange(n).remainder(2**x).sub(2**x-1).sign().add(1) 
                        for x in range(n.bit_length())])
        d = (n.bit_length()-3)*f
        m = torch.zeros(n,d,d)
        for i in range(n):
            key = interval[i]*f
            for j in range(key+1, d):
                m[i,j,j] = 1
            if key < d:
                m[i,key,key] = .5**.5
                m[i,key,key-1] = .5**.5
            for j in range(1,min(key,d)):
                m[i,j,j-1] = 1
        return m.transpose(1,2)  # 1k 32 32

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
                if self.use_bias:
                    m.bias.data.zero_()
            elif isinstance(m, LayerNormParameterized):
                m.reset_parameters()
        nn.init.trunc_normal_(self.key_pos, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.query_pos, mean=0.0, std=0.02)

    def forward(
        self,
        q,
        k,
        v,
        mask: Optional[Tensor] = None,
        position_ids=None,
        attn_algorithm=None,
        past_key_value_state: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache=False,
        is_self=True,
        is_causal_mask=False,
    ):
        """
        past_key_value_state: tuple
            the cache to be used in attention of the form (<self/cross>_key, <self/cross>_value)
        position_ids: Optional[torch.LongTensor]
            The position of each of the tokens encoded in q and k. Used for RoPE embeddings
        use_cache: bool
            if True, the kv states for self/cross attention will be saved, otherwise they will not be saved
        is_self: bool
            if True, this will perform self attention, otherwise this will perform cross attention. Note: This will
            only be used in the case that use_cache=True. This may be removed in future

        Returns
        -------
        tensor or tuple
            If use_cache=False, only the hidden state will be returned as a tensor. If use_cache=True, a tuple will be
            returned in the form (hidden_state, cache) where hidden_state is a tensor and cache is of the form specified
            in past_key_value_state
        """

        # q, k, v: batch_size x seq_len x emb_dim
        # mask: batch_size x seq_len x seq_len
        batch_size, q_len, _ = q.size()
        kv_len = k.size(1)

        # split emb_dim as nheads*emb_dim_per_head
        # b x h x qlen x ds
        queries = self.query(q).view(
            batch_size, q_len, self.nheads, self.emb_kq_per_head
        )

        # if this is self attention, we always recompute
        # cross attention only gets computed when a cache does not exist
        # if we dont have the cache yet, we need to compute
        # d x (h x ds)
        # b x kvlen x d
        # b x kvlen x h x ds
        # b x h x kvlen x ds
        if is_self or past_key_value_state is None:
            keys = self.key(k).view(
                batch_size, kv_len, self.kvheads, self.emb_kq_per_head
            )

            values = self.value(v).view(
                batch_size, kv_len, self.kvheads, self.emb_v_per_head
            )

            # # You want to apply rotary embeddings pre-cache
            # if self.position_encoder is not None:
            #     queries, keys = self.position_encoder.adjusted_qk(
            #         queries, keys, position_ids, past_key_value_state, use_cache
            #     )

        queries = queries / (self.emb_kq_per_head**(1/4))  # b l h d
        keys = keys / (self.emb_kq_per_head**(1/4))  # b l h d

        # Build scan cache
        # k/v: b l h d
        keys = keys.view(batch_size, kv_len, -1)
        values = values.view(batch_size, kv_len, -1)
        gate = self.gates.repeat(4,1,1)[None]  # 1 4096 32 32
        # gate = self.gates[None,None]  # 1 1 32 32
        # gate = gate.expand(batch_size, kv_len, -1, -1)  # b l 32 32
        keys = keys.unsqueeze(3)  # b l d 1
        keys = F.pad(keys, (0, 32-1))  # b l d 32
        keys = self.scan(keys, gate).view(batch_size, kv_len, self.kvheads, self.emb_kq_per_head, -1)  # b l h d 32
        keys = keys + self.key_pos  # b l h d 32
        keys = self.ln_k(keys.transpose(3,4))  # b l h 32 d
        values = values.unsqueeze(3)  # b l d 1
        values = F.pad(values, (0, 32-1))  # b l d 32
        values = self.scan(values, gate).view(batch_size, kv_len, self.kvheads, self.emb_v_per_head, -1)  # b l h d 32
        values = self.ln_v(values.transpose(3,4))  # b l h 32 d
        queries = queries + self.query_pos  # b l h d

        # if you want to use caching and past_key_value_state is not None meaning you have values in your cache
        if use_cache and past_key_value_state is not None:
            if is_self:
                keys = torch.cat((past_key_value_state[0], keys), dim=2)
                values = torch.cat((past_key_value_state[1], values), dim=2)
            else:
                keys = past_key_value_state[0]
                values = past_key_value_state[1]

        # Merge rel pos bias and mask into single float mask
        if mask is not None:
            # Our expected mask format is bs x q_len x k_len, so to make it broadcastable
            # we need to create the nheads dimension
            while len(mask.size()) != 4:  # expects bs (x nheads) x q_len x kv_len
                mask = mask.unsqueeze(1)

        if self.position_encoder is not None:
            attn_mask: Optional[Tensor] = self.position_encoder.adjusted_mask(
                mask, queries, keys, past_key_value_state, use_cache
            )
        else:
            attn_mask = mask

        # Expand kv so black-box attn will work
        expansion = self.nheads // self.kvheads
        # k/v: b l h 32 d
        if expansion != 1:
            keys_e = keys.unsqueeze(3).expand(-1, -1, -1, expansion, -1, -1).flatten(2, 3)
            values_e = (
                values.unsqueeze(3).expand(-1, -1, -1, expansion, -1, -1).flatten(2, 3)
            )
        else:
            keys_e = keys
            values_e = values


        qk = torch.einsum("blhd,blhed->blhe", queries, keys_e)  # b l h 32
        qk = qk.softmax(3)
        attn = torch.einsum("blhe,blhed->blhd", qk, values_e)  # b l h d

        # qk = queries.matmul(keys_e.transpose(2,3))
        # m = torch.ones(qk.size(2), qk.size(3), device=qk.device, dtype=qk.dtype).tril()
        # m = m - torch.ones_like(m).tril(diagonal=-32)
        # m = m.log()
        # qk = qk.add(m).softmax(3)  # b h l l
        # attn = qk.matmul(values_e)  # b h l d

        # attn = F.scaled_dot_product_attention(
        #     queries,
        #     keys_e,
        #     values_e,
        #     attn_mask=attn_mask,
        #     dropout_p=self.p_dropout if self.training else 0.0,
        #     is_causal=is_causal_mask,
        # )

        # attn: bs x seq_len x nheads*emb_v_per_head
        # attn: b x h x qlen x ds
        # attn after permute: b x qlen x h x ds
        # b x qlen x (d)
        attn = (
            attn.reshape(batch_size, q_len, self.nheads * self.emb_v_per_head)
        )
        out = self.dense(attn)

        # if use_cache=True, we return the hidden_state as well as the kv cache
        if use_cache:
            return out, (keys, values)
        else:
            return out