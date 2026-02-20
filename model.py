"""Minimal transformer for modular arithmetic grokking experiments."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, _ = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, T, -1)
        return self.out(out)


class TransformerBlock(nn.Module):

    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GrokTransformer(nn.Module):
    """
    Transformer for modular arithmetic: (a, b) -> (a op b) mod p.

    Two input tokens from vocabulary {0, ..., p-1}, two positional
    embeddings, N transformer layers, linear head to p classes.
    """

    def __init__(self, p, d_model=128, n_heads=4, n_layers=2, d_ff=None):
        super().__init__()
        self.p = p
        self.d_model = d_model

        if d_ff is None:
            d_ff = 4 * d_model

        self.tok_emb = nn.Embedding(p, d_model)
        self.pos_emb = nn.Embedding(2, d_model)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, p)

    def forward(self, x, return_features=False):
        """
        Args:
            x: (B, 2) tensor of token indices in {0, ..., p-1}
            return_features: if True, return (logits, features) where
                features is the pre-head representation.
        """
        B, T = x.shape
        pos = torch.arange(T, device=x.device)

        h = self.tok_emb(x) + self.pos_emb(pos)

        for layer in self.layers:
            h = layer(h)

        h = self.ln_f(h)
        features = h.mean(dim=1)
        logits = self.head(features)

        if return_features:
            return logits, features
        return logits
