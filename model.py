from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F

from config import ModelConfig


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if n_embd % num_heads != 0:
            raise ValueError("n_embd must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_size = n_embd // num_heads
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = dropout
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, channels = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(channels, dim=2)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=True,
        )
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        return self.resid_dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, n_embd: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.sa = MultiHeadAttention(config.n_embd, config.n_head, config.dropout)
        self.ffwd = FeedForward(config.n_embd, config.dropout)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, config: ModelConfig | None = None, **overrides: int | float) -> None:
        super().__init__()
        if config is not None and overrides:
            raise ValueError("Provide either a ModelConfig or keyword overrides, not both")
        self.config = config or ModelConfig(**overrides)
        self.block_size = self.config.block_size
        self.token_embedding_table = nn.Embedding(vocab_size, self.config.n_embd)
        self.position_embedding_table = nn.Embedding(self.block_size, self.config.n_embd)
        self.blocks = nn.Sequential(*[Block(self.config) for _ in range(self.config.n_layer)])
        self.ln_f = nn.LayerNorm(self.config.n_embd)
        self.lm_head = nn.Linear(self.config.n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, seq_len = idx.shape
        if seq_len > self.block_size:
            raise ValueError(f"sequence length {seq_len} exceeds block_size {self.block_size}")

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(seq_len, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits, None

        batch_size, target_seq_len, vocab_size = logits.shape
        loss = F.cross_entropy(logits.reshape(batch_size * target_seq_len, vocab_size), targets.reshape(batch_size * target_seq_len))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None and top_k >= 1:
                top_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < top_values[:, [-1]], float("-inf"))
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            if eos_token_id is not None and torch.all(idx_next == eos_token_id):
                break
        return idx
