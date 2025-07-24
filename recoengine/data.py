"""Synthetic shopping sessions with real structure: users have latent tastes over item "styles", so
history is predictive and a learned model can beat popularity. Interactions are time-ordered so the
evaluation is leave-last-out (predict each user's most recent item from everything before it)."""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Item:
    id: int
    category: int
    style: int
    popularity: float


def make_catalog(n_items: int = 600, n_categories: int = 12, n_styles: int = 20, seed: int = 0) -> list[Item]:
    rng = random.Random(seed)
    return [Item(i, rng.randrange(n_categories), rng.randrange(n_styles), rng.paretovariate(1.5)) for i in range(n_items)]


def make_interactions(catalog: list[Item], n_users: int = 1500, per_user: tuple[int, int] = (6, 20), seed: int = 1) -> dict[int, list[int]]:
    """user -> item ids in time order. 80% of picks follow the user's 2 favourite styles, 20% popularity."""
    rng = random.Random(seed)
    styles = sorted({i.style for i in catalog})
    by_style: dict[int, list[Item]] = {}
    for it in catalog:
        by_style.setdefault(it.style, []).append(it)
    pop_items = [i.id for i in catalog]
    pop_w = [i.popularity for i in catalog]
    out = {}
    for u in range(n_users):
        favs = rng.sample(styles, 2)
        seq: list[int] = []
        for _ in range(rng.randint(*per_user)):
            if rng.random() < 0.8:
                pool = by_style[rng.choice(favs)]
                it = rng.choices(pool, weights=[p.popularity for p in pool])[0].id
            else:
                it = rng.choices(pop_items, weights=pop_w)[0]
            if it not in seq:
                seq.append(it)
        out[u] = seq
    return out


def split_leave_last_out(inter: dict[int, list[int]]):
    train = {u: s[:-1] for u, s in inter.items() if len(s) >= 3}
    test = {u: s[-1] for u, s in inter.items() if len(s) >= 3}
    return train, test
