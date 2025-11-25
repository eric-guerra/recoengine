"""Kafka → feature store. Runs as its own deployment (scaled by consumer lag), separate from serving.

Offsets are committed only after the Redis pipeline for the batch has executed (at-least-once); the
event-id dedupe in FeatureStore.apply_event makes redelivery harmless.
"""
from __future__ import annotations

import os

from .features import FeatureStore


def run():
    import redis
    from confluent_kafka import Consumer

    store = FeatureStore(redis.Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0")))
    c = Consumer({
        "bootstrap.servers": os.environ.get("KAFKA_BROKERS", "localhost:9092"),
        "group.id": "reco-features",
        "enable.auto.commit": False,
        "auto.offset.reset": "latest",  # features are about *recent* behaviour; no need to replay history
        "isolation.level": "read_committed",
    })
    c.subscribe(["shop.events.v1", "inventory.stock.v1"])
    try:
        while True:
            msgs = c.consume(num_messages=500, timeout=1.0)
            if not msgs:
                continue
            for m in msgs:
                if m.error():
                    continue
                store.apply_event(m.value().decode())
            c.commit(asynchronous=False)
    finally:
        c.close()


if __name__ == "__main__":
    run()
