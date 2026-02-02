import math

import torch
from torch import nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wq = nn.Linear(config.n_embd, config.n_embd)
        self.wk = nn.Linear(config.n_embd, config.n_embd)
        self.wv = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.block_size = config.block_size

        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x, kv_cache=None):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # [bs, n, T, n_head]
        k = self.wk(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        new_kv_cache = (k, v)

        att = (q @ k.transpose(-2, -1)) / (math.sqrt(q.shape[-1]))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, -float("inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return y, new_kv_cache


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = LayerNorm(config.n_embd, bias=True)
        self.attn = CausalSelfAttention(config)
        self.ln2 = LayerNorm(config.n_embd, bias=True)
        self.mlp = MLP(config)

    def forward(self, x, kv_cache):
        attn_out, new_kv_cache = self.attn(self.ln1(x), kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, new_kv_cache


class Config:
    n_embd = 256
    n_head = 16
    block_size = 128
    vocab_size = 512
    n_layer = 4


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList(Block(config) for _ in range(config.n_layer)),
            ln_f=LayerNorm(config.n_embd, bias=True),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx, kv_caches=None, use_cache=False):
        B, T = idx.shape

        if kv_caches is not None and len(kv_caches) > 0:
            past_length = kv_caches[0][0].size(2)
            pos = torch.arange(past_length, past_length + T, device=idx.device)
        else:
            pos = torch.arange(0, T, device=idx.device)

        tok_emb = self.transformer.wte(idx)  # (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb
        for block_idx, block in enumerate(self.transformer.h):
            cache = kv_caches[block_idx] if kv_caches is not None else None
            x, new_cache = block(x, cache)
            if use_cache:
                kv_caches.append(new_cache)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x[:, [-1], :])
        return logits, kv_caches if use_cache else None

    def generate(self, idx, max_tokens, top_k):
        kv_caches = None

        for _ in range(max_tokens):
            if kv_caches is None:
                idx_cond = idx if idx.size(1) < self.config.block_size else idx[:, -self.config.block_size]
            else:
                idx_cond = idx[:, [-1]]

            logits, kv_caches = self(idx_cond, kv_caches, use_cache=True)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

input_tensor = torch.randint(0, 512, (8, 128), device="cuda")
model = GPT(Config()).to("cuda")
model.eval()
output = model(input_tensor)
