"""A small decoder-only transformer built from plain ``nn.Linear`` layers.

It exists so the full pipeline (train, calibrate, quantize, evaluate) runs in seconds on
a CPU, in tests and in CI, without downloading anything.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class TinyLMConfig:
    vocab_size: int = 128
    dim: int = 128
    n_layers: int = 2
    n_heads: int = 4
    mlp_ratio: int = 4
    max_len: int = 256


class Block(nn.Module):
    def __init__(self, cfg: TinyLMConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.norm1 = nn.LayerNorm(cfg.dim)
        self.q_proj = nn.Linear(cfg.dim, cfg.dim)
        self.k_proj = nn.Linear(cfg.dim, cfg.dim)
        self.v_proj = nn.Linear(cfg.dim, cfg.dim)
        self.o_proj = nn.Linear(cfg.dim, cfg.dim)
        self.norm2 = nn.LayerNorm(cfg.dim)
        self.up_proj = nn.Linear(cfg.dim, cfg.mlp_ratio * cfg.dim)
        self.down_proj = nn.Linear(cfg.mlp_ratio * cfg.dim, cfg.dim)

    def forward(self, x: Tensor) -> Tensor:
        b, t, d = x.shape
        h = self.norm1(x)

        def heads(z: Tensor) -> Tensor:
            return z.view(b, t, self.n_heads, d // self.n_heads).transpose(1, 2)

        attn = F.scaled_dot_product_attention(
            heads(self.q_proj(h)), heads(self.k_proj(h)), heads(self.v_proj(h)), is_causal=True
        )
        x = x + self.o_proj(attn.transpose(1, 2).reshape(b, t, d))
        return x + self.down_proj(F.gelu(self.up_proj(self.norm2(x))))


class TinyLM(nn.Module):
    def __init__(self, cfg: TinyLMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.dim)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = nn.LayerNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

    def forward(self, ids: Tensor) -> Tensor:
        t = ids.shape[-1]
        if t > self.cfg.max_len:
            raise ValueError(f"sequence length {t} exceeds max_len {self.cfg.max_len}")
        x = self.tok_emb(ids) + self.pos_emb(torch.arange(t, device=ids.device))
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.norm(x))


def train_tiny_lm(
    tokens: Tensor,
    cfg: TinyLMConfig,
    *,
    steps: int = 300,
    batch_size: int = 32,
    seq_len: int = 64,
    lr: float = 3e-3,
    seed: int = 0,
) -> TinyLM:
    """Train a :class:`TinyLM` on random crops of ``tokens`` with AdamW and cosine decay."""
    torch.manual_seed(seed)
    model = TinyLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    g = torch.Generator().manual_seed(seed)
    n = tokens.numel() - seq_len - 1
    model.train()
    for _ in range(steps):
        starts = torch.randint(0, n, (batch_size,), generator=g)
        idx = starts.unsqueeze(1) + torch.arange(seq_len + 1)
        batch = tokens[idx]
        logits = model(batch[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), batch[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    model.eval()
    return model
