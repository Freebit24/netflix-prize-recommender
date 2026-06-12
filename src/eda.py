"""Exploratory data analysis: figures + summary stats saved to results/."""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"figure.dpi": 130, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False})
C = "#E50914"  # netflix red


def run(df, titles, out_dir, label=""):
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    stats = {}

    n_users, n_movies = df.user_id.nunique(), df.movie_id.nunique()
    stats["n_ratings"] = int(len(df))
    stats["n_users"] = int(n_users)
    stats["n_movies"] = int(n_movies)
    stats["density_pct"] = round(100 * len(df) / (n_users * n_movies), 3)
    stats["mean_rating"] = round(float(df.rating.mean()), 3)
    stats["date_range"] = [str(df.date.min().date()), str(df.date.max().date())]

    # 1. Rating distribution
    fig, ax = plt.subplots(figsize=(6, 3.5))
    counts = df.rating.value_counts().sort_index()
    ax.bar(counts.index, counts.values / 1e6, color=C)
    ax.set_xlabel("Rating"); ax.set_ylabel("Ratings (millions)")
    ax.set_title("Rating distribution")
    fig.tight_layout(); fig.savefig(f"{fig_dir}/rating_distribution.png"); plt.close(fig)
    stats["rating_distribution_pct"] = {
        int(k): round(100 * v / len(df), 1) for k, v in counts.items()}

    # 2. User activity (ratings per user)
    upc = df.user_id.value_counts()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(upc.values, bins=np.logspace(0, np.log10(upc.max()), 50), color=C)
    ax.set_xscale("log"); ax.set_xlabel("Ratings per user (log)")
    ax.set_ylabel("Users"); ax.set_title("User activity — long tail")
    fig.tight_layout(); fig.savefig(f"{fig_dir}/user_activity.png"); plt.close(fig)
    stats["ratings_per_user"] = {"median": int(upc.median()),
                                 "mean": round(float(upc.mean()), 1),
                                 "max": int(upc.max())}

    # 3. Movie popularity (ratings per movie) + top movies
    mpc = df.movie_id.value_counts()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(np.arange(1, len(mpc) + 1), mpc.values, color=C)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Movie rank (log)"); ax.set_ylabel("Ratings (log)")
    ax.set_title("Content popularity — power law")
    fig.tight_layout(); fig.savefig(f"{fig_dir}/movie_popularity.png"); plt.close(fig)
    tmap = titles.set_index("movie_id")["title"]
    stats["top_movies"] = [
        {"title": tmap.get(m, str(m)), "n_ratings": int(c),
         "mean_rating": round(float(df[df.movie_id == m].rating.mean()), 2)}
        for m, c in mpc.head(10).items()]
    stats["pop_concentration_pct"] = round(
        100 * mpc.head(int(0.1 * len(mpc))).sum() / len(df), 1)

    # 4. Temporal trends
    monthly = df.set_index("date").resample("ME")["rating"].agg(["count", "mean"])
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 4.5), sharex=True)
    axes[0].plot(monthly.index, monthly["count"] / 1e3, color=C)
    axes[0].set_ylabel("Ratings (k)/month"); axes[0].set_title("Temporal trends")
    axes[1].plot(monthly.index, monthly["mean"], color="#444")
    axes[1].set_ylabel("Mean rating"); axes[1].set_xlabel("Date")
    fig.tight_layout(); fig.savefig(f"{fig_dir}/temporal_trends.png"); plt.close(fig)
    stats["mean_rating_first_year"] = round(
        float(monthly["mean"].iloc[:12].mean()), 3)
    stats["mean_rating_last_year"] = round(
        float(monthly["mean"].iloc[-12:].mean()), 3)

    # 5. Popularity vs quality
    g = df.groupby("movie_id")["rating"].agg(["count", "mean"])
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.scatter(g["count"], g["mean"], s=4, alpha=0.25, color=C)
    ax.set_xscale("log"); ax.set_xlabel("Ratings per movie (log)")
    ax.set_ylabel("Mean rating"); ax.set_title("Popularity vs quality")
    fig.tight_layout(); fig.savefig(f"{fig_dir}/popularity_vs_quality.png"); plt.close(fig)
    stats["pop_quality_corr"] = round(
        float(np.corrcoef(np.log10(g["count"]), g["mean"])[0, 1]), 3)

    with open(os.path.join(out_dir, f"eda_stats{label}.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/ratings_sample.parquet")
    ap.add_argument("--titles", default="data/processed/titles.parquet")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    run(pd.read_parquet(args.data), pd.read_parquet(args.titles), args.out)
