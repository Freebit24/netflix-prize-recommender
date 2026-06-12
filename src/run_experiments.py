"""End-to-end experiment runner: trains all models, evaluates, generates
sample recommendations, and writes results/ artifacts used by the report.

Supports checkpointed runs (--models / --ckpt-dir) so each model can be
trained and evaluated in a separate invocation on constrained machines.
"""
import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd

from src.models import BiasBaseline, ItemCF, SVD
from src.evaluate import evaluate_model
from src.recommend import top_k, explain_recs, user_profile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--out", default="results")
    ap.add_argument("--max-eval-users", type=int, default=2000,
                    help="cap ranking-metric users for speed (sampled)")
    ap.add_argument("--models", default="BiasBaseline,ItemCF,SVD",
                    help="comma-separated subset to run (checkpointed)")
    ap.add_argument("--ckpt-dir", default=None,
                    help="where fitted models are pickled (default: --out)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ckpt = os.path.join(args.ckpt_dir or args.out, "models.pkl")

    train = pd.read_parquet(f"{args.data_dir}/train.parquet")
    test = pd.read_parquet(f"{args.data_dir}/test.parquet")
    titles = pd.read_parquet(f"{args.data_dir}/titles.parquet")

    all_models = {"BiasBaseline": BiasBaseline(), "ItemCF": ItemCF(k=40),
                  "SVD": SVD(n_factors=50, n_epochs=25)}
    wanted = args.models.split(",")
    # checkpointing: merge with previously saved metrics / models
    fitted = {}
    if os.path.exists(ckpt):
        try:
            with open(ckpt, "rb") as f:
                fitted = pickle.load(f)
        except Exception:
            fitted = {}
    rows = (pd.read_csv(f"{args.out}/metrics.csv").to_dict("records")
            if os.path.exists(f"{args.out}/metrics.csv") else [])
    rows = [r for r in rows if r["model"] not in wanted]
    for name in wanted:
        m = all_models[name]
        print(f"=== {m.name} ===")
        rows.append(evaluate_model(m, train, test,
                                   max_users=args.max_eval_users))
        fitted[m.name] = m
        print(rows[-1])
    metrics = pd.DataFrame(rows)
    metrics.to_csv(f"{args.out}/metrics.csv", index=False)
    with open(ckpt, "wb") as f:
        pickle.dump(fitted, f)
    print(metrics.to_string(index=False))
    if not {"SVD", "ItemCF"} <= set(fitted):
        return

    # ---- sample recommendations (best model = SVD; explanations via ItemCF)
    svd, icf = fitted["SVD"], fitted["ItemCF"]
    upc = train.user_id.value_counts()
    sample_users = {
        "heavy_rater": int(upc.index[10]),
        "median_user": int(upc.index[len(upc) // 2]),
        "light_user": int(upc[upc.between(16, 25)].index[0]),
    }
    recs_out = {}
    for label, uid in sample_users.items():
        profile = user_profile(uid, train, titles)
        recs = top_k(svd, uid, train, titles, k=10)
        expl = explain_recs(icf, uid, recs.movie_id.tolist(), titles)
        heldout = (test[(test.user_id == uid) & (test.rating >= 3.5)]
                   .merge(titles, on="movie_id").title.tolist())
        recs_out[label] = {
            "user_id": uid,
            "n_train_ratings": int(upc[uid]),
            "profile": profile.astype(str).to_dict("records"),
            "recommendations": recs.astype(str).to_dict("records"),
            "explanations": expl.astype(str).to_dict("records"),
            "heldout_relevant": [str(t) for t in heldout],
        }
    with open(f"{args.out}/sample_recommendations.json", "w") as f:
        json.dump(recs_out, f, indent=2)
    print("Saved results to", args.out)


if __name__ == "__main__":
    main()
