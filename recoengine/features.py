"""Online feature store on Redis, updated from the Kafka click/purchase stream.

Keys:
  user:{id}:recent   LIST of item ids, newest first, capped at 50 (LPUSH + LTRIM, atomic in a pipeline)
  item:{id}:stock    "1"/"0" in-stock flag, fed by the inventory topic
  model:items        item embedding matrix for the active model version (bytes) + version string

Consumers are idempotent: each event carries an id, remembered for 24h with SET NX, so a Kafka
redelivery doesn't double-count a click.
"""
from __future__ import annotations

import json
from typing import Protocol

import numpy as np

RECENT_CAP = 50


class Redisish(Protocol):
    def pipeline(self): ...
    def lrange(self, k, a, b): ...
    def mget(self, keys): ...
    def get(self, k): ...
    def set(self, k, v, nx: bool = False, ex: int | None = None): ...


class FeatureStore:
    def __init__(self, r: Redisish):
        self.r = r

    def recent(self, user_id: str, n: int = 20) -> list[int]:
        return [int(x) for x in self.r.lrange(f"user:{user_id}:recent", 0, n - 1)][::-1]  # oldest → newest for the model

    def in_stock(self, item_ids: list[int]) -> list[bool]:
        vals = self.r.mget([f"item:{i}:stock" for i in item_ids])
        return [v != b"0" and v != "0" for v in vals]  # unknown → assume in stock

    def apply_event(self, raw: str) -> bool:
        """Kafka consumer handler for `shop.events.v1`. Returns False for duplicates/ignored events."""
        e = json.loads(raw)
        if not self.r.set(f"evt:{e['eventId']}", "1", nx=True, ex=86_400):
            return False
        p = self.r.pipeline()
        if e["type"] in ("view", "add_to_cart", "purchase"):
            key = f"user:{e['userId']}:recent"
            p.lrem(key, 0, e["itemId"])  # move-to-front instead of duplicates
            p.lpush(key, e["itemId"])
            p.ltrim(key, 0, RECENT_CAP - 1)
            p.expire(key, 90 * 86_400)
        elif e["type"] == "stock":
            p.set(f"item:{e['itemId']}:stock", "1" if e["inStock"] else "0")
        else:
            return False
        p.execute()
        return True

    def publish_item_embeddings(self, emb: np.ndarray, version: str):
        p = self.r.pipeline()
        p.set("model:items", emb.astype(np.float32).tobytes())
        p.set("model:items:shape", json.dumps(list(emb.shape)))
        p.set("model:version", version)  # readers check this to hot-swap
        p.execute()

    def item_embeddings(self) -> tuple[str, np.ndarray]:
        shape = json.loads(self.r.get("model:items:shape"))
        emb = np.frombuffer(self.r.get("model:items"), dtype=np.float32).reshape(shape)
        v = self.r.get("model:version")
        return (v.decode() if isinstance(v, bytes) else v), emb


class MemoryRedis:
    """Enough of redis-py for the feature store and tests."""

    def __init__(self):
        self.kv, self.lists = {}, {}

    def set(self, k, v, nx=False, ex=None):
        if nx and k in self.kv:
            return None
        self.kv[k] = v
        return True

    def get(self, k):
        return self.kv.get(k)

    def mget(self, keys):
        return [self.kv.get(k) for k in keys]

    def lrange(self, k, a, b):
        lst = self.lists.get(k, [])
        return lst[a: None if b == -1 else b + 1]

    def pipeline(self):
        r = self
        ops = []

        class P:
            def __getattr__(self, name):
                return lambda *a, **kw: ops.append((name, a, kw))

            def execute(self):
                for name, a, kw in ops:
                    getattr(r, "_" + name, getattr(r, name, None))(*a, **kw)

        return P()

    def _lrem(self, k, _count, v):
        self.lists[k] = [x for x in self.lists.get(k, []) if str(x) != str(v)]

    def _lpush(self, k, v):
        self.lists.setdefault(k, []).insert(0, str(v))

    def _ltrim(self, k, a, b):
        self.lists[k] = self.lists.get(k, [])[a:b + 1]

    def _expire(self, k, s):
        pass
