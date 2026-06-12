"""Top-K recommendation generation with optional explanations."""
import numpy as np
import pandas as pd


def top_k(model, user_id, train, titles, k=10):
    """Top-k unseen movies for a user, with titles and predicted scores."""
    catalog = np.sort(train.movie_id.unique())
    seen = set(train.loc[train.user_id == user_id, "movie_id"])
    cand = catalog[~np.isin(catalog, list(seen))]
    scores = model.score_items(user_id, cand)
    order = np.argsort(-scores)[:k]
    out = pd.DataFrame({"movie_id": cand[order],
                        "pred_score": np.round(scores[order], 2)})
    return out.merge(titles, on="movie_id", how="left")


def explain_recs(item_cf, user_id, rec_movie_ids, titles, top_n=3):
    """For each recommended movie, name the user's highly-rated movies most
    similar to it: 'Because you rated X and Y highly.'"""
    tmap = titles.set_index("movie_id")["title"].to_dict()
    lines = []
    for mid in rec_movie_ids:
        nbrs = item_cf.explain(user_id, int(mid), top_n=top_n)
        because = [f"{tmap.get(n, n)} (you rated it {r:.0f}/5)"
                   for n, s, r in nbrs if r >= 4]
        lines.append({
            "movie_id": int(mid),
            "title": tmap.get(int(mid), str(mid)),
            "because": "; ".join(because) if because else
                       "popular among users with similar taste",
        })
    return pd.DataFrame(lines)


def user_profile(user_id, train, titles, n=8):
    """A user's top-rated training movies (for qualitative analysis)."""
    h = (train[train.user_id == user_id]
         .sort_values(["rating", "date"], ascending=False).head(n))
    return h.merge(titles, on="movie_id", how="left")[
        ["title", "year", "rating", "date"]]
