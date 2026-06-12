"""Data pipeline: parse raw Netflix Prize files, subsample, and split.

Raw format (combined_data_*.txt):
    MovieID:
    UserID,Rating,YYYY-MM-DD
    ...
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

RAW_GLOB = "combined_data_*.txt"
TITLES_FILE = "movie_titles.csv"


def parse_raw(raw_dir: str) -> pd.DataFrame:
    """Parse all combined_data_*.txt files into a single DataFrame."""
    frames = []
    for fp in sorted(glob.glob(os.path.join(raw_dir, RAW_GLOB))):
        print(f"Parsing {fp} ...")
        users, movies, ratings, dates = [], [], [], []
        movie_id = -1
        with open(fp) as f:
            for line in f:
                line = line.rstrip("\n")
                if line.endswith(":"):
                    movie_id = int(line[:-1])
                else:
                    u, r, d = line.split(",")
                    users.append(int(u))
                    movies.append(movie_id)
                    ratings.append(int(r))
                    dates.append(d)
        frames.append(pd.DataFrame({
            "user_id": np.array(users, dtype=np.int32),
            "movie_id": np.array(movies, dtype=np.int16),
            "rating": np.array(ratings, dtype=np.int8),
            "date": pd.to_datetime(dates),
        }))
    df = pd.concat(frames, ignore_index=True)
    print(f"Parsed {len(df):,} ratings, {df.user_id.nunique():,} users, "
          f"{df.movie_id.nunique():,} movies")
    return df


def load_titles(raw_dir: str) -> pd.DataFrame:
    """movie_titles.csv: movie_id, year, title (title may contain commas)."""
    rows = []
    with open(os.path.join(raw_dir, TITLES_FILE), encoding="latin-1") as f:
        for line in f:
            mid, year, title = line.rstrip("\n").split(",", 2)
            rows.append((int(mid), None if year == "NULL" else int(year), title))
    return pd.DataFrame(rows, columns=["movie_id", "year", "title"])


def subsample(df: pd.DataFrame, n_users: int = 8000, n_movies: int = 4000,
              min_user_ratings: int = 20, seed: int = 42) -> pd.DataFrame:
    """Keep the n_movies most-rated movies, then sample n_users among users
    with >= min_user_ratings on those movies. Keeps the matrix dense enough
    to learn from while preserving the long-tail shape within the sample."""
    top_movies = df.movie_id.value_counts().head(n_movies).index
    df = df[df.movie_id.isin(top_movies)]
    counts = df.user_id.value_counts()
    eligible = counts[counts >= min_user_ratings].index.to_numpy()
    rng = np.random.default_rng(seed)
    keep = rng.choice(eligible, size=min(n_users, len(eligible)), replace=False)
    out = df[df.user_id.isin(keep)].reset_index(drop=True)
    print(f"Subsample: {len(out):,} ratings, {out.user_id.nunique():,} users, "
          f"{out.movie_id.nunique():,} movies")
    return out


def temporal_split(df: pd.DataFrame, test_frac: float = 0.2,
                   min_train: int = 5):
    """Per-user temporal split: most recent test_frac of each user's ratings
    go to test. Users with < min_train ratings stay entirely in train."""
    df = df.sort_values(["user_id", "date"], kind="mergesort")
    grp = df.groupby("user_id", sort=False)
    n = grp["rating"].transform("size")
    rank = grp.cumcount()
    cutoff = np.ceil(n * (1 - test_frac)).astype(int)
    is_test = (rank >= cutoff) & (n >= min_train)
    train = df[~is_test].reset_index(drop=True)
    test = df[is_test].reset_index(drop=True)
    # Drop test rows whose movie never appears in train (unrateable cold items)
    test = test[test.movie_id.isin(train.movie_id.unique())].reset_index(drop=True)
    print(f"Split: train={len(train):,}  test={len(test):,}")
    return train, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--n-users", type=int, default=8000)
    ap.add_argument("--n-movies", type=int, default=4000)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = parse_raw(args.raw_dir)
    df.to_parquet(os.path.join(args.out_dir, "ratings_full.parquet"))
    sample = subsample(df, args.n_users, args.n_movies)
    sample.to_parquet(os.path.join(args.out_dir, "ratings_sample.parquet"))
    train, test = temporal_split(sample)
    train.to_parquet(os.path.join(args.out_dir, "train.parquet"))
    test.to_parquet(os.path.join(args.out_dir, "test.parquet"))
    load_titles(args.raw_dir).to_parquet(os.path.join(args.out_dir, "titles.parquet"))
    print("Done.")


if __name__ == "__main__":
    main()
