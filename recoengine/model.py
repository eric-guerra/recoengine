"""Two-tower retrieval model (PyTorch).

  user tower: mean of the embeddings of the user's recent items (last 20), through a small MLP
  item tower: item-id embedding + category embedding, through a small MLP

The user tower takes *history*, not a user id, so brand-new users get recommendations after one click
and the model never needs retraining to serve a new user: their history is read from the feature store.
Training uses sampled softmax with in-batch negatives and a log-popularity correction (logQ), so the
model doesn't just learn "recommend popular things".
"""
from __future__ import annotations

import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import Item

HISTORY = 20


class TwoTower(nn.Module):
    def __init__(self, n_items: int, n_categories: int, dim: int = 32):
        super().__init__()
        self.item_emb = nn.Embedding(n_items + 1, dim, padding_idx=n_items)  # last index = padding
        self.cat_emb = nn.Embedding(n_categories, dim)
        self.item_mlp = nn.Sequential(nn.Linear(2 * dim, 64), nn.ReLU(), nn.Linear(64, dim))
        self.user_mlp = nn.Sequential(nn.Linear(dim, 64), nn.ReLU(), nn.Linear(64, dim))
        self.pad = n_items
        self.register_buffer("cat_lookup", torch.zeros(n_items + 1, dtype=torch.long))

    def set_categories(self, cats: list[int]):
        self.cat_lookup[: len(cats)] = torch.tensor(cats)

    def item_vectors(self, item_ids: torch.Tensor, cats: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.item_emb(item_ids), self.cat_emb(cats)], dim=-1)
        return F.normalize(self.item_mlp(x), dim=-1)

    def user_vectors(self, histories: torch.Tensor, cats: torch.Tensor | None = None) -> torch.Tensor:
        # History items go through the *item tower* (shared weights), so users and items live in one space
        # and a single click already places a new user next to similar items.
        mask = (histories != self.pad).float().unsqueeze(-1)
        hc = self.cat_lookup[histories]
        vecs = self.item_vectors(histories.clamp(max=self.pad), hc) * mask
        pooled = vecs.sum(1) / mask.sum(1).clamp(min=1)
        return F.normalize(pooled + self.user_mlp(pooled), dim=-1)


def pad_history(seq: list[int], pad: int) -> list[int]:
    h = seq[-HISTORY:]
    return [pad] * (HISTORY - len(h)) + h


def training_examples(train: dict[int, list[int]]):
    """Every prefix → next item (teaches the model to predict the next click from recent history)."""
    ex = []
    for seq in train.values():
        for t in range(1, len(seq)):
            ex.append((seq[:t], seq[t]))
    return ex


def train_model(catalog: list[Item], train: dict[int, list[int]], *, epochs: int = 20, batch: int = 256, lr: float = 3e-3,
                temperature: float = 0.07, dim: int = 32, logq: bool = True, seed: int = 0) -> TwoTower:
    torch.manual_seed(seed)
    random.seed(seed)
    n_items = len(catalog)
    n_cat = max(i.category for i in catalog) + 1
    model = TwoTower(n_items, n_cat, dim=dim)
    model.set_categories([i.category for i in catalog])
    cats = torch.tensor([i.category for i in catalog])
    counts = np.ones(n_items)
    for s in train.values():
        for i in s:
            counts[i] += 1
    log_q = torch.tensor(np.log(counts / counts.sum()), dtype=torch.float32)

    ex = training_examples(train)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        random.shuffle(ex)
        for b in range(0, len(ex), batch):
            chunk = ex[b:b + batch]
            hist = torch.tensor([pad_history(h, model.pad) for h, _ in chunk])
            pos = torch.tensor([p for _, p in chunk])
            u = model.user_vectors(hist)
            v = model.item_vectors(pos, cats[pos])
            logits = (u @ v.T) / temperature - (log_q[pos].unsqueeze(0) if logq else 0)  # in-batch negatives, popularity-corrected
            # Same item can appear twice in a batch; mask those false negatives.
            same = pos.unsqueeze(0) == pos.unsqueeze(1)
            logits = logits.masked_fill(same & ~torch.eye(len(chunk), dtype=torch.bool), -1e9)
            loss = F.cross_entropy(logits, torch.arange(len(chunk)))
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def export_item_embeddings(model: TwoTower, catalog: list[Item]) -> np.ndarray:
    ids = torch.arange(len(catalog))
    cats = torch.tensor([i.category for i in catalog])
    return model.item_vectors(ids, cats).numpy()


@torch.no_grad()
def user_embedding(model: TwoTower, history: list[int]) -> np.ndarray:
    return model.user_vectors(torch.tensor([pad_history(history, model.pad)]))[0].numpy()


def recall_at_k(recs: dict[int, list[int]], test: dict[int, int], k: int = 10) -> float:
    return sum(test[u] in recs[u][:k] for u in test) / len(test)


def ndcg_at_k(recs: dict[int, list[int]], test: dict[int, int], k: int = 10) -> float:
    s = 0.0
    for u, target in test.items():
        if target in recs[u][:k]:
            s += 1 / math.log2(recs[u].index(target) + 2)
    return s / len(test)
