"""Item-based collaborative filtering with adjusted-cosine similarity.

Similarity is computed between item columns after subtracting each user's
mean rating (removes per-user scale effects). Predictions are a similarity-
weighted average of the target user's ratings over the k most similar
co-rated items, anchored on the bias baseline as fallback.

Interpretable by construction: the neighbors that drive a prediction are
the explanation ("recommended because you rated X and Y highly").
"""
import numpy as np
import pandas as pd
from scipy import sparse

from .baseline import BiasBaseline


class ItemCF:
    name = "ItemCF"

    def __init__(self, k: int = 40, shrink: float = 100.0,
                 min_overlap: int = 3):
        self.k = k
        self.shrink = shrink          # shrinks similarities with low support
        self.min_overlap = min_overlap

    def fit(self, train: pd.DataFrame):
        self.fallback = BiasBaseline().fit(train)
        self.user_ids = np.sort(train.user_id.unique())
        self.movie_ids = np.sort(train.movie_id.unique())
        self.uidx = {u: i for i, u in enumerate(self.user_ids)}
        self.midx = {m: i for i, m in enumerate(self.movie_ids)}

        rows = train.user_id.map(self.uidx).to_numpy()
        cols = train.movie_id.map(self.midx).to_numpy()
        vals = train.rating.to_numpy(dtype=np.float32)
        n_u, n_m = len(self.user_ids), len(self.movie_ids)

        self.R = sparse.csr_matrix((vals, (rows, cols)), shape=(n_u, n_m))
        user_mean = np.asarray(self.R.sum(1)).ravel() / np.maximum(
            self.R.getnnz(1), 1)
        self.user_mean = user_mean

        # Mean-center nonzero entries per user
        Rc = self.R.copy().astype(np.float32)
        Rc.data -= np.repeat(user_mean, np.diff(self.R.indptr))
        Rc = Rc.tocsc()

        # Item-item adjusted cosine with shrinkage on co-rating counts
        norms = np.sqrt(np.asarray(Rc.multiply(Rc).sum(0)).ravel())
        sim = (Rc.T @ Rc).toarray()
        overlap = (self.R.T.astype(bool).astype(np.float32)
                   @ self.R.astype(bool).astype(np.float32)).toarray()
        denom = np.outer(norms, norms) + 1e-9
        sim = sim / denom * (overlap / (overlap + self.shrink))
        sim[overlap < self.min_overlap] = 0.0
        np.fill_diagonal(sim, 0.0)
        self.sim = sim.astype(np.float32)
        self.R_csr = self.R.tocsr()
        return self

    def _predict_one(self, u: int, m: int) -> float:
        base = self.fallback.mu + self.fallback.b_u.get(
            u, 0.0) + self.fallback.b_i.get(m, 0.0)
        ui, mi = self.uidx.get(u), self.midx.get(m)
        if ui is None or mi is None:
            return float(np.clip(base, 1, 5))
        start, end = self.R_csr.indptr[ui], self.R_csr.indptr[ui + 1]
        rated = self.R_csr.indices[start:end]
        ratings = self.R_csr.data[start:end]
        sims = self.sim[mi, rated]
        if len(rated) > self.k:
            top = np.argpartition(-np.abs(sims), self.k)[:self.k]
            sims, ratings = sims[top], ratings[top]
        wsum = np.abs(sims).sum()
        if wsum < 1e-6:
            return float(np.clip(base, 1, 5))
        dev = ratings - self.user_mean[ui]
        pred = self.user_mean[ui] + (sims @ dev) / wsum
        return float(np.clip(pred, 1, 5))

    def _score_targets(self, user_id: int, target_idx: np.ndarray) -> np.ndarray:
        """Vectorized prediction for one user over many target items."""
        ui = self.uidx.get(user_id)
        bi = np.array([self.fallback.b_i.get(int(self.movie_ids[t]), 0.0)
                       for t in target_idx])
        base = np.clip(self.fallback.mu
                       + self.fallback.b_u.get(user_id, 0.0) + bi, 1, 5)
        if ui is None:
            return base
        start, end = self.R_csr.indptr[ui], self.R_csr.indptr[ui + 1]
        rated = self.R_csr.indices[start:end]
        ratings = self.R_csr.data[start:end]
        S = self.sim[np.ix_(target_idx, rated)]          # (T, n_rated)
        if len(rated) > self.k:                          # keep top-k per row
            thresh = -np.partition(-np.abs(S), self.k - 1, axis=1)[:, self.k - 1]
            S = np.where(np.abs(S) >= thresh[:, None], S, 0.0)
        wsum = np.abs(S).sum(1)
        dev = ratings - self.user_mean[ui]
        pred = self.user_mean[ui] + (S @ dev) / np.maximum(wsum, 1e-9)
        return np.where(wsum < 1e-6, base, np.clip(pred, 1, 5))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        df = df.reset_index(drop=True)
        out = np.empty(len(df))
        for u, g in df.groupby("user_id", sort=False):
            tgt = np.array([self.midx.get(m, -1) for m in g.movie_id])
            ok = tgt >= 0
            res = np.full(len(g), np.nan)
            if ok.any():
                res[ok] = self._score_targets(u, tgt[ok])
            if (~ok).any():  # unseen movies -> baseline fallback
                res[~ok] = self.fallback.predict(g[~ok])
            out[g.index.to_numpy()] = res
        return out

    def score_items(self, user_id: int, movie_ids: np.ndarray) -> np.ndarray:
        tgt = np.array([self.midx.get(int(m), -1) for m in movie_ids])
        ok = tgt >= 0
        out = np.full(len(movie_ids), self.fallback.mu)
        if ok.any():
            out[ok] = self._score_targets(user_id, tgt[ok])
        return out

    def explain(self, user_id: int, movie_id: int, top_n: int = 3):
        """Return [(neighbor_movie_id, similarity, user_rating)] that drove
        the recommendation of movie_id for user_id."""
        ui, mi = self.uidx.get(user_id), self.midx.get(movie_id)
        if ui is None or mi is None:
            return []
        start, end = self.R_csr.indptr[ui], self.R_csr.indptr[ui + 1]
        rated = self.R_csr.indices[start:end]
        ratings = self.R_csr.data[start:end]
        sims = self.sim[mi, rated]
        order = np.argsort(-sims)[:top_n]
        return [(int(self.movie_ids[rated[j]]), float(sims[j]),
                 float(ratings[j])) for j in order if sims[j] > 0]

    def similar_items(self, movie_id: int, top_n: int = 10):
        mi = self.midx.get(movie_id)
        if mi is None:
            return []
        order = np.argsort(-self.sim[mi])[:top_n]
        return [(int(self.movie_ids[j]), float(self.sim[mi, j]))
                for j in order]
