# Reference: https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py
import os
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn


def init_process_group():
    if dist.is_available() and "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


rank, world_size = init_process_group()


@dataclass
class ModelArgs:
    max_batch_size: int = 8
    max_seq_len: int = 4096 * 4
    dtype: Literal["bf16", "fp8"] = "bf16"
    vocab_size: int = 102400
    dim: int = 2048
    inter_dim: int = 10944
    moe_inter_dim: int = 1408
    n_layers: int = 27
    n_dense_layers: int = 1
    n_heads: int = 16

    # MoE
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    n_activated_experts: int = 6
    n_expert_groups: int = 1
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    route_scale: float = 1.0
    n_limited_groups: int = 1

    # MLA
    q_lora_rank: int = 0
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128

    rope_theta: int = 10000


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)


class _ColumnParallelLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(self, x: torch.Tensor, weight: torch.Tensor, bias) -> torch.Tensor:
        y = x @ weight
        if bias is not None:
            y = y + bias
        ctx.save_for_backward(x, weight, bias)
        return y
    
    def backward(ctx, dout):
        # dx = g @ w.T
        # dw = x.T @ g
        # dout.shape: [B, out_features // TP]
        x, weight, bias = ctx.saved_tensors        
        # [B, out_features // TP] @ [out_features // TP, in_features] -> [B, in_features]
        dx = dout @ weight.T 
        dist.all_reduce(dx, op=dist.ReduceOp.SUM)

        x = ctx.input_tensor
        # [in_features, B] @ [B, out_features] -> [in_features, out_features]
        dw = x.T @ dout
        dbias = dout.sum(dim=0) if self.bias is not None else None
        return dx, dw, db

class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features // world_size))

class RowParallelLinear(Linear):
    def __init__(
        self, in_features: int, out_features: int, bias: bool = False, dtype=None
    ):
        super().__init__(in_features // world_size, out_features, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # y = x @ w, we need all reduce here
        y = linear(x, self.weight, self.bias)
        if world_size > 1:
            dist.all_reduce(y)
        if self.bias is not None:
            y += self.bias
        return y


class MLA(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads  # matters when world_size > 1
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim

        if self.q_lora_rank == 0:
            # dim: 2048, qk_head_dim: 192
            self.wq = ColumnParallelLinear(self.dim, self.n_heads * self.qk_head_dim)
        else:
            self.wq_a = Linear(self.dim, self.q_lora_rank)
            self.q_norm = nn.RMSNorm(self.q_lora_rank)
            self.wq_b = ColumnParallelLinear(
                self.q_lora_rank, self.n_heads * self.qk_head_dim
            )

        # kv_lora_rank: 512, qk_rope_head_dim: 64
        self.wkv_a = Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
        self.kv_norm = nn.RMSNorm(self.kv_lora_rank)
        self.wkv_b = ColumnParallelLinear(
            self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim)
        )

        self.wo = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim)

        self.softmax_scale = (self.qk_head_dim) ** (-0.5)

        self.register_buffer(
            "k_cache",
            torch.zeros(
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_heads,
                self.qk_head_dim,
            ),
            persistent=False,
        )
        self.register_buffer(
            "v_cache",
            torch.zeros(
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_heads,
                self.v_head_dim,
            ),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x)))

        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        q_pe = apply_rotary_emb(q_pe, freqs_cis)

        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)

        q = torch.cat([q_nope, q_pe], dim=-1)
        kv = self.wkv_b(self.kv_norm(kv))
        kv = kv.view(
            bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)

        self.k_cache[:bsz, start_pos:end_pos] = k
        self.v_cache[:bsz, start_pos:end_pos] = v

        scores = (
            torch.einsum("bshd,bthd->bsht", q, self.k_cache[:bsz, :end_pos])
            * self.softmax_scale
        )

        if mask is not None:
            scores += mask.unsqueeze(1)
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
        x = torch.einsum("bsht,bthd->bshd", scores, self.v_cache[:bsz, :end_pos])
        x = self.wo(x.flatten(2))
        return x


def precompute_freqs_cis(args: ModelArgs) -> torch.Tensor:
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    base = args.rope_theta

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


class MLP(nn.Module):
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim)
        self.w2 = nn.Linear(inter_dim, dim)
        self.w3 = nn.Linear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Gate(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.n_groups = args.n_expert_groups
        self.topk_groups = args.n_limited_groups
        self.score_func = "softmax"
        self.route_scale = args.route_scale
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        self.bias = (
            nn.Parameter(torch.empty(args.n_routed_experts, dtype=torch.float32))
            if self.dim == 7168
            else None
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Outputs weights and indices.
            weights are the distribution over number of routed experts (topk)
                weights.shape: [bs, seqlen, topk]
            indices represent selected experts
                indices.shape: [bs, seqlen, topk]
        """
        scores = F.linear(x, self.weight)
        # scores.shape: [bs, seq_len, dim]
        scores = scores.softmax(dim=-1, dtype=torch.float32)

        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        indices = torch.topk(scores, self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        # softmax normalization
        weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        return weights.type_as(x), indices


class Expert(nn.Module):
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim)
        self.w2 = nn.Linear(inter_dim, dim)
        self.w3 = nn.Linear(dim, inter_dim)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_routed_experts = args.n_routed_experts
        self.n_local_experts = args.n_routed_experts
        self.n_activated_experts = args.n_activated_experts
        self.experts_start_idx = rank * self.n_local_experts
        self.expertss_end_idx = self.experts_start_idx + self.n_local_experts
        self.gate = Gate(args)

        l: list[torch.Tensor] = []
        for i in range(self.n_routed_experts):
            if self.experts_start_idx <= i < self.expertss_end_idx:
                l.append(Expert(args.dim, args.moe_inter_dim))
        self.experts = nn.ModuleList(l)
        self.shared_expert = MLP(args.dim, args.n_shared_experts * args.moe_inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        # we are flattening bs, seqlen into one dim
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x)
        y = torch.zeros_like(x)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts)
        for idx in range(self.experts_start_idx, self.expertss_end_idx):
            if counts[idx] == 0:
                continue
            expert = self.experts[idx]
            idx, top = torch.where(indices == idx)
            y[idx] += expert(x[idx]) * weights[idx, top, None]
        z = self.shared_expert(x)
        if world_size > 1:
            dist.all_reduce(y)
        return (y + z).view(shape)


def test_moe():
    args = ModelArgs()
    device = "cuda"
    moe = MoE(args).to(device)
    x = torch.rand(4, 512, args.dim, device=device)
    out = moe(x)
    print(f"out.shape: {out.shape}")


def test_gate():
    args = ModelArgs()
    device = "cuda"
    gate = Gate(args).to(device)
    x = torch.rand(4, 512, args.dim, device=device)
    weights, indices = gate(x)
    print(f"weights.shape: {weights.shape}")
    print(f"indices.shape: {indices.shape}")


def test_attention():
    args = ModelArgs()
    device = "cuda"
    attention = MLA(args).to(device)
    x = torch.rand(8, 512, args.dim, device=device)
    freqs_cis = precompute_freqs_cis(args)
    start_pos = 0
    seqlen = x.size(1)
    # [seqlen, qk_rope_head_dim//2]
    freqs_cis = freqs_cis[start_pos : start_pos + seqlen].to(device)
    mask = torch.full((seqlen, seqlen), float("-inf"), device=x.device).triu_(1)
    out = attention(x, start_pos, freqs_cis, mask).to(device)
    print(out.shape)


# test_moe()
# test_gate()
test_attention()
dist.destroy_process_group()
