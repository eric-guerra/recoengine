# RecoEngine

Portfolio reference implementation of a 2025 project, built to show real-time "recommended for you" for an e-commerce site. It combines a PyTorch two-tower model, a Redis online feature store fed from Kafka, and a FastAPI serving layer with business filters.

**Stack:** Python 3.11, PyTorch 2.3, Redis (feature store), Kafka (`confluent-kafka`), FastAPI, NumPy.

## Model (`recoengine/model.py`)
- **Two towers in one embedding space.** The item tower combines the item id and category embeddings through an MLP. The user tower pools the *item-tower vectors* of the user's last 20 interactions (shared weights) and adds a residual MLP.
  - Because users are defined by their history rather than an id, a brand-new user is personalised after **one click**, and serving never needs retraining for new users.
- **Training:**
  - Next-item prediction over every prefix of each session.
  - **Sampled softmax with in-batch negatives**, a **logQ popularity correction** so it doesn't just learn "recommend bestsellers", and masking of duplicate positives within a batch.
- **Evaluation:** leave-last-out; predict each user's most recent item. Hyperparameters were chosen on a **separate validation split** (second-to-last item).

| | Recall@10 | NDCG@10 |
|---|---|---|
| Popularity baseline | 0.070 | 0.036 |
| **Two-tower** | **0.207** | **0.112** |

What actually moved the numbers, measured on the validation split:
- **Sharing the item tower** in the user tower took recall@10 from 0.15 to 0.21, and one-click users went from 1 to 10 same-style items in their top 10.
- **logQ correction:** 0.198 with it vs 0.135 without.
- **Epochs:** 6 → 20 epochs gave 0.148 → 0.215.

## Online path
- **Feature store** (`features.py`, Redis):
  - `user:{id}:recent` is a capped list, newest first. A re-view moves the item to the front instead of duplicating it.
  - Item stock flags come from the inventory topic.
  - Item embeddings are published with a **model version**, which servers check so they can hot-swap without restarting.
- **Stream consumer** (`stream.py`):
  - Kafka `shop.events.v1` and `inventory.stock.v1` feed Redis pipelines.
  - Offsets are committed after the writes.
  - Event-id `SET NX` dedupe makes redelivery harmless.
- **Serving** (`serve.py`):
  - History → user vector → dot product against item embeddings.
  - Seen items and **out-of-stock items** are removed.
  - **MMR diversification** prevents a wall of near-identical items.
  - Cold start falls back to popular in-stock items. The response reports the strategy and model version.

## Test
```bash
pip install -r requirements.txt
pytest -q -s   # 6 tests: beats popularity by >2.5x recall@10, one-click personalisation, idempotent and recency-ordered features,
               # seen/out-of-stock filtering, cold start, model hot swap, API validation, MMR
```
