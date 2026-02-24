import json
from collections import Counter

import numpy as np
import pytest
from fastapi.testclient import TestClient

from recoengine.data import make_catalog, make_interactions, split_leave_last_out
from recoengine.features import FeatureStore, MemoryRedis
from recoengine.model import export_item_embeddings, ndcg_at_k, recall_at_k, train_model, user_embedding
from recoengine.serve import Recommender, create_app, mmr


@pytest.fixture(scope="module")
def trained():
    catalog = make_catalog()
    inter = make_interactions(catalog)
    train, test = split_leave_last_out(inter)
    model = train_model(catalog, train)
    return catalog, train, test, model


def popularity_recs(train, catalog, k=10):
    counts = Counter(i for s in train.values() for i in s)
    ranked = [i for i, _ in counts.most_common()] + [i.id for i in catalog if i.id not in counts]
    return {u: [i for i in ranked if i not in set(s)][:k] for u, s in train.items()}


def model_recs(model, catalog, train, k=10):
    emb = export_item_embeddings(model, catalog)
    out = {}
    for u, s in train.items():
        scores = emb @ user_embedding(model, s)
        seen = set(s)
        out[u] = [int(i) for i in np.argsort(-scores) if int(i) not in seen][:k]
    return out


def test_two_tower_beats_popularity_on_leave_last_out(trained):
    catalog, train, test, model = trained
    pop, ours = popularity_recs(train, catalog), model_recs(model, catalog, train)
    r_pop, r_ours = recall_at_k(pop, test), recall_at_k(ours, test)
    print(f"recall@10 popularity={r_pop:.3f} two-tower={r_ours:.3f}  ndcg@10 {ndcg_at_k(pop, test):.3f} → {ndcg_at_k(ours, test):.3f}")
    assert r_ours > r_pop * 2.5, (r_pop, r_ours)


def test_new_user_gets_personalised_results_after_one_click(trained):
    catalog, _, _, model = trained
    emb = export_item_embeddings(model, catalog)
    clicked = next(i for i in catalog if i.style == 3)
    scores = emb @ user_embedding(model, [clicked.id])
    top = [int(i) for i in np.argsort(-scores) if int(i) != clicked.id][:10]
    same_style = sum(catalog[i].style == clicked.style for i in top)
    assert same_style >= 4, same_style  # base rate is ~1 in 20


def event(eid, uid, iid, typ="view", **kw):
    return json.dumps({"eventId": eid, "userId": uid, "itemId": iid, "type": typ, **kw})


def test_feature_store_updates_are_idempotent_and_recency_ordered():
    fs = FeatureStore(MemoryRedis())
    assert fs.apply_event(event("e1", "u1", 5))
    assert fs.apply_event(event("e2", "u1", 7))
    assert not fs.apply_event(event("e2", "u1", 7)), "redelivery ignored"
    assert fs.apply_event(event("e3", "u1", 5))  # re-view moves to front, no duplicate
    assert fs.recent("u1") == [7, 5]
    fs.apply_event(json.dumps({"eventId": "s1", "itemId": 7, "type": "stock", "inStock": False}))
    assert fs.in_stock([7, 5]) == [False, True]


def test_serving_filters_seen_and_out_of_stock_and_diversifies(trained):
    catalog, _, _, model = trained
    fs = FeatureStore(MemoryRedis())
    fs.publish_item_embeddings(export_item_embeddings(model, catalog), "v1")
    popular = [i.id for i in sorted(catalog, key=lambda x: -x.popularity)]
    rec = Recommender(fs, catalog, lambda h: user_embedding(model, h), popular)
    for n, item in enumerate([i.id for i in catalog if i.style == 2][:5]):
        fs.apply_event(event(f"v{n}", "u9", item))
    first = rec.recommend("u9", diversify=False)["items"]
    fs.apply_event(json.dumps({"eventId": "oos", "itemId": first[0], "type": "stock", "inStock": False}))
    out = rec.recommend("u9")
    assert out["strategy"] == "two_tower" and len(out["items"]) == 10
    assert not set(out["items"]) & set(fs.recent("u9"))
    assert first[0] not in out["items"]
    cold = rec.recommend("nobody")
    assert cold["strategy"] == "popular" and cold["items"][0] == popular[0]


def test_model_hot_swap_and_api(trained):
    catalog, _, _, model = trained
    fs = FeatureStore(MemoryRedis())
    emb = export_item_embeddings(model, catalog)
    fs.publish_item_embeddings(emb, "v1")
    rec = Recommender(fs, catalog, lambda h: user_embedding(model, h), list(range(len(catalog))))
    client = TestClient(create_app(rec))
    assert client.get("/recommendations/x").json()["model"] == "v1"
    fs.publish_item_embeddings(emb, "v2")
    assert client.get("/recommendations/x", params={"k": 5}).json()["model"] == "v2"
    assert client.get("/recommendations/x", params={"k": 500}).status_code == 422


def test_mmr_trades_relevance_for_variety():
    emb = np.array([[1, 0], [0.99, 0.14], [0, 1]], dtype=np.float32)
    scores = np.array([1.0, 0.98, 0.7])
    assert mmr([0, 1, 2], scores, emb, 2, lam=0.5) == [0, 2]
    assert mmr([0, 1, 2], scores, emb, 2, lam=1.0) == [0, 1]
