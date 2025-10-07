"""Serving: history → user vector → nearest items → business filters → category diversification."""
from __future__ import annotations

import time

import numpy as np
from fastapi import FastAPI, Query

from .data import Item
from .features import FeatureStore


def mmr(candidates: list[int], scores: np.ndarray, emb: np.ndarray, k: int, lam: float = 0.75) -> list[int]:
    """Maximal Marginal Relevance: trade a little relevance for variety (no wall of 10 near-identical items)."""
    chosen: list[int] = []
    pool = list(candidates)
    while pool and len(chosen) < k:
        def value(i):
            sim = max((float(emb[i] @ emb[j]) for j in chosen), default=0.0)
            return lam * scores[i] - (1 - lam) * sim
        best = max(pool, key=value)
        chosen.append(best)
        pool.remove(best)
    return chosen


class Recommender:
    def __init__(self, store: FeatureStore, catalog: list[Item], user_tower, popular: list[int]):
        self.store, self.catalog, self.user_tower, self.popular = store, catalog, user_tower, popular
        self.version, self.emb = store.item_embeddings()

    def refresh_model(self):
        v, emb = self.store.item_embeddings()
        if v != self.version:
            self.version, self.emb = v, emb

    def recommend(self, user_id: str, k: int = 10, category: int | None = None, diversify: bool = True) -> dict:
        history = self.store.recent(user_id)
        if not history:  # cold start: popular in-stock items
            cands = [i for i in self.popular if category is None or self.catalog[i].category == category]
            stock = self.store.in_stock(cands[: k * 3])
            return {"items": [i for i, ok in zip(cands[: k * 3], stock) if ok][:k], "strategy": "popular", "model": self.version}
        u = self.user_tower(history)
        scores = self.emb @ u
        seen = set(history)
        order = [int(i) for i in np.argsort(-scores) if int(i) not in seen and (category is None or self.catalog[int(i)].category == category)]
        top = order[: k * 5]
        stock = self.store.in_stock(top)
        top = [i for i, ok in zip(top, stock) if ok]
        items = mmr(top, scores, self.emb, k) if diversify else top[:k]
        return {"items": items, "strategy": "two_tower", "model": self.version}


def create_app(rec: Recommender) -> FastAPI:
    app = FastAPI(title="RecoEngine")

    @app.get("/recommendations/{user_id}")
    def recommendations(user_id: str, k: int = Query(10, ge=1, le=50), category: int | None = None):
        t0 = time.perf_counter()
        rec.refresh_model()
        out = rec.recommend(user_id, k, category)
        return {**out, "took_ms": round((time.perf_counter() - t0) * 1000, 2)}

    return app
