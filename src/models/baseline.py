"""Bias baseline: r_hat(u,i) = mu + b_u + b_i (regularized means).

A deliberately simple anchor model. On Netflix-style data it is hard to
beat by a wide margin, which makes it the honest yardstick for fancier
models (Koren, 2009).
"""
import numpy as np
import pandas as pd


class BiasBaseline:
    name = "BiasBaseline"

    def __init__(self, reg: float = 10.0):
        self.reg = reg

    def fit(self, train: pd.DataFrame):
        self.mu = train.rating.mean()
        # Regularized item bias
        g = train.groupby("movie_id")["rating"]
        self.b_i = ((g.sum() - g.count() * self.mu)
                    / (g.count() + self.reg)).to_dict()
        # Regularized user bias on residuals
        resid = train.rating - self.mu - train.movie_id.map(self.b_i).fillna(0)
        g = resid.groupby(train.user_id)
        self.b_u = (g.sum() / (g.count() + self.reg)).to_dict()
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        pred = (self.mu
                + df.user_id.map(self.b_u).fillna(0).to_numpy()
                + df.movie_id.map(self.b_i).fillna(0).to_numpy())
        return np.clip(pred, 1, 5)

    def score_items(self, user_id: int, movie_ids: np.ndarray) -> np.ndarray:
        bu = self.b_u.get(user_id, 0.0)
        bi = np.array([self.b_i.get(m, 0.0) for m in movie_ids])
        return np.clip(self.mu + bu + bi, 1, 5)
