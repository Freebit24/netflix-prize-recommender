"""FunkSVD-style biased matrix factorization trained with SGD.

r_hat(u,i) = mu + b_u + b_i + p_u . q_i

The technique that defined the Netflix Prize (Funk 2006; Koren et al. 2009).
Learns latent factors capturing taste dimensions (genre affinity, tone,
era, ...) directly from the rating matrix. Pure NumPy; vectorized SGD over
shuffled epochs is fast enough at our sample scale.
"""
import numpy as np
import pandas as pd

try:  # optional JIT acceleration (~100x); pure-NumPy fallback below
    from numba import njit
except ImportError:  # pragma: no cover
    njit = None


def _sgd_epoch(u, m, r, order, mu, bu, bi, P, Q, lr, reg):
    for j in order:
        uu, mm = u[j], m[j]
        pred = mu + bu[uu] + bi[mm] + P[uu] @ Q[mm]
        e = r[j] - pred
        bu[uu] += lr * (e - reg * bu[uu])
        bi[mm] += lr * (e - reg * bi[mm])
        pu = P[uu].copy()
        P[uu] += lr * (e * Q[mm] - reg * pu)
        Q[mm] += lr * (e * pu - reg * Q[mm])


if njit is not None:
    _sgd_epoch = njit(cache=True)(_sgd_epoch)


class SVD:
    name = "SVD"

    def __init__(self, n_factors: int = 50, n_epochs: int = 25,
                 lr: float = 0.007, reg: float = 0.05, seed: int = 42):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.seed = seed

    def fit(self, train: pd.DataFrame, verbose: bool = False):
        rng = np.random.default_rng(self.seed)
        self.user_ids = np.sort(train.user_id.unique())
        self.movie_ids = np.sort(train.movie_id.unique())
        self.uidx = {u: i for i, u in enumerate(self.user_ids)}
        self.midx = {m: i for i, m in enumerate(self.movie_ids)}

        u = train.user_id.map(self.uidx).to_numpy()
        m = train.movie_id.map(self.midx).to_numpy()
        r = train.rating.to_numpy(dtype=np.float64)

        n_u, n_m, k = len(self.user_ids), len(self.movie_ids), self.n_factors
        self.mu = r.mean()
        self.bu = np.zeros(n_u)
        self.bi = np.zeros(n_m)
        self.P = rng.normal(0, 0.1, (n_u, k))
        self.Q = rng.normal(0, 0.1, (n_m, k))

        idx = np.arange(len(r))
        u = u.astype(np.int64)
        m = m.astype(np.int64)
        for epoch in range(self.n_epochs):
            rng.shuffle(idx)
            _sgd_epoch(u, m, r, idx, self.mu, self.bu, self.bi,
                       self.P, self.Q, self.lr, self.reg)
            if verbose:
                pred = (self.mu + self.bu[u] + self.bi[m]
                        + np.einsum("ij,ij->i", self.P[u], self.Q[m]))
                print(f"epoch {epoch + 1}: train RMSE = "
                      f"{np.sqrt(np.mean((r - pred) ** 2)):.4f}")
        return self

    def _predict_arrays(self, users, movies) -> np.ndarray:
        out = np.full(len(users), self.mu)
        for j, (uu, mm) in enumerate(zip(users, movies)):
            ui, mi = self.uidx.get(uu), self.midx.get(mm)
            p = self.mu
            if ui is not None:
                p += self.bu[ui]
            if mi is not None:
                p += self.bi[mi]
            if ui is not None and mi is not None:
                p += self.P[ui] @ self.Q[mi]
            out[j] = p
        return np.clip(out, 1, 5)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self._predict_arrays(df.user_id.to_numpy(),
                                    df.movie_id.to_numpy())

    def score_items(self, user_id: int, movie_ids: np.ndarray) -> np.ndarray:
        ui = self.uidx.get(user_id)
        if ui is None:
            bi = np.array([self.bi[self.midx[m]] if m in self.midx else 0.0
                           for m in movie_ids])
            return np.clip(self.mu + bi, 1, 5)
        mi = np.array([self.midx[m] for m in movie_ids])
        scores = (self.mu + self.bu[ui] + self.bi[mi]
                  + self.Q[mi] @ self.P[ui])
        return np.clip(scores, 1, 5)
