"""Evaluation: RMSE, MAE (rating accuracy) and MAP@10, Precision@10,
Recall@10, NDCG@10, catalog coverage (ranking quality).

Ranking protocol
----------------
For each test user, the candidate set is every movie in the training
catalog that the user has NOT rated in train. Models score all candidates;
the top-10 form the recommendation list. A held-out test movie is
*relevant* if its actual rating >= 3.5 (per problem statement). MAP@10
averages, over users with >= 1 relevant held-out item, the average
precision of the top-10 list against that user's relevant set.
"""
import time

import numpy as np
import pandas as pd

RELEVANCE_THRESHOLD = 3.5
K = 10


def rmse(y, yhat):
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def mae(y, yhat):
    return float(np.mean(np.abs(y - yhat)))


def top_k_for_user(model, user_id, candidates, k=K):
    scores = model.score_items(user_id, candidates)
    top = np.argsort(-scores)[:k]
    return candidates[top], scores[top]


def ranking_metrics(model, train, test, k=K, max_users=None, seed=42):
    """Compute MAP@k, Precision@k, Recall@k, NDCG@k, coverage."""
    catalog = np.sort(train.movie_id.unique())
    seen = train.groupby("user_id")["movie_id"].agg(set)
    relevant = (test[test.rating >= RELEVANCE_THRESHOLD]
                .groupby("user_id")["movie_id"].agg(set))
    users = relevant.index.to_numpy()
    if max_users and len(users) > max_users:
        users = np.random.default_rng(seed).choice(users, max_users,
                                                   replace=False)
    aps, precs, recs, ndcgs = [], [], [], []
    recommended_items = set()
    for u in users:
        cand = catalog[~np.isin(catalog, list(seen.get(u, set())))]
        rec_ids, _ = top_k_for_user(model, u, cand, k)
        recommended_items.update(rec_ids.tolist())
        rel = relevant[u]
        hits = np.array([m in rel for m in rec_ids])
        # Average precision @ k
        if hits.any():
            prec_at = np.cumsum(hits) / (np.arange(k) + 1)
            ap = (prec_at * hits).sum() / min(len(rel), k)
        else:
            ap = 0.0
        aps.append(ap)
        precs.append(hits.sum() / k)
        recs.append(hits.sum() / len(rel))
        # NDCG@k (binary relevance)
        dcg = (hits / np.log2(np.arange(k) + 2)).sum()
        idcg = (1 / np.log2(np.arange(min(len(rel), k)) + 2)).sum()
        ndcgs.append(dcg / idcg)
    return {
        f"MAP@{k}": float(np.mean(aps)),
        f"Precision@{k}": float(np.mean(precs)),
        f"Recall@{k}": float(np.mean(recs)),
        f"NDCG@{k}": float(np.mean(ndcgs)),
        "Coverage": len(recommended_items) / len(catalog),
        "n_eval_users": len(users),
    }


def evaluate_model(model, train, test, max_users=None, fit=True):
    res = {"model": model.name}
    if fit:
        t0 = time.time()
        model.fit(train)
        res["fit_seconds"] = round(time.time() - t0, 1)
    t0 = time.time()
    preds = model.predict(test)
    res["predict_seconds"] = round(time.time() - t0, 1)
    res["RMSE"] = round(rmse(test.rating.to_numpy(), preds), 4)
    res["MAE"] = round(mae(test.rating.to_numpy(), preds), 4)
    t0 = time.time()
    rk = ranking_metrics(model, train, test, max_users=max_users)
    res["rank_seconds"] = round(time.time() - t0, 1)
    res.update({k: round(v, 4) if isinstance(v, float) else v
                for k, v in rk.items()})
    return res
