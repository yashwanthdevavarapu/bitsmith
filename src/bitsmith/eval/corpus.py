"""A synthetic corpus with a known optimal perplexity.

Tokens follow a first order Markov chain where every token has ``branching`` possible
successors with Dirichlet distributed probabilities. Because the generating process is
known, the entropy rate and therefore the best achievable perplexity can be computed
exactly. That gives benchmarks a fixed floor to compare against, with no downloads.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass
class MarkovCorpus:
    vocab_size: int = 128
    branching: int = 4
    concentration: float = 0.5
    seed: int = 0
    transition: Tensor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not 1 <= self.branching <= self.vocab_size:
            raise ValueError("branching must be in [1, vocab_size]")
        g = torch.Generator().manual_seed(self.seed)
        t = torch.zeros(self.vocab_size, self.vocab_size, dtype=torch.float64)
        alpha = torch.full((self.branching,), self.concentration, dtype=torch.float64)
        dirichlet = torch.distributions.Dirichlet(alpha)
        torch.manual_seed(self.seed)  # Dirichlet sampling uses the global generator
        for i in range(self.vocab_size):
            succ = torch.randperm(self.vocab_size, generator=g)[: self.branching]
            t[i, succ] = dirichlet.sample()
        self.transition = t

    def stationary(self, iters: int = 2000) -> Tensor:
        p = torch.full((self.vocab_size,), 1.0 / self.vocab_size, dtype=torch.float64)
        for _ in range(iters):
            p = 0.5 * p + 0.5 * (p @ self.transition)  # lazy chain: converges for periodic chains too
        return p / p.sum()

    def entropy_rate(self) -> float:
        """Entropy per token in nats under the stationary distribution."""
        t = self.transition
        row_h = -(torch.where(t > 0, t * t.log(), torch.zeros_like(t))).sum(dim=1)
        return float(self.stationary() @ row_h)

    def optimal_perplexity(self) -> float:
        return math.exp(self.entropy_rate())

    def sample(self, n_tokens: int, seed: int | None = None) -> Tensor:
        g = torch.Generator().manual_seed(self.seed + 1 if seed is None else seed)
        out = torch.empty(n_tokens, dtype=torch.long)
        state = int(torch.randint(self.vocab_size, (1,), generator=g))
        cdf = self.transition.cumsum(dim=1)
        u = torch.rand(n_tokens, generator=g, dtype=torch.float64)
        for i in range(n_tokens):
            out[i] = state
            nxt = int(torch.searchsorted(cdf[state], u[i]))
            state = min(nxt, self.vocab_size - 1)
        return out
