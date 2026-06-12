# Netflix Prize: Personalized Recommendation System

A recommendation engine built on the [Netflix Prize Dataset](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data) that learns user preferences, predicts unseen ratings, and generates explained Top-10 recommendations.

## Approach

Three models with a common `fit / predict / score_items` interface:

| Model | Idea | Why it's here |
|---|---|---|
| **BiasBaseline** | μ + user bias + item bias | Honest anchor, surprisingly strong on this data |
| **ItemCF** | Adjusted-cosine item-item kNN | Interpretable; powers "because you watched X" explanations |
| **SVD** | Biased matrix factorization (FunkSVD, SGD) | The Netflix Prize workhorse; best accuracy |

**Split:** per-user temporal — each user's most recent 20% of ratings are held out (mimics predicting the future).
**Metrics:** RMSE, MAE (rating accuracy); MAP@10, Precision@10, Recall@10, NDCG@10, coverage (ranking). Relevance = held-out rating ≥ 3.5.

## Repository structure

```
src/
  data_pipeline.py    # parse raw files → parquet, subsample, temporal split
  eda.py              # EDA figures + stats → results/
  models/             # baseline.py, item_cf.py, svd.py
  evaluate.py         # RMSE, MAE, MAP@10, NDCG@10, coverage
  recommend.py        # Top-K generation + explanations
  run_experiments.py  # end-to-end: train, evaluate, sample recs
results/              # metrics.csv, figures/, sample_recommendations.json
report/               # technical report (PDF)
slides/               # presentation (PDF)
```

## Reproduce

```bash
pip install -r requirements.txt

# 1. Download data (needs Kaggle credentials)
python -c "import kagglehub; print(kagglehub.dataset_download('netflix-inc/netflix-prize-data'))"

# 2. Parse + subsample + split (point --raw-dir at the downloaded folder)
python src/data_pipeline.py --raw-dir <downloaded_path>

# 3. EDA
python src/eda.py

# 4. Train + evaluate + generate recommendations
python -m src.run_experiments
```

The full dataset (100M ratings) is parsed once; experiments run on a reproducible subsample (seed=42) of ~8,000 active users × 4,000 most-rated movies, as permitted by the problem statement.

## Results

Per-user temporal split (last 20% held out): 1,359,913 train / 336,027 test ratings. Ranking metrics over 2,000 sampled users, all-unseen-catalog candidates, relevance = rating ≥ 3.5.

| Model | RMSE | MAE | MAP@10 | NDCG@10 | Coverage | Fit time |
|---|---|---|---|---|---|---|
| Bias baseline | 0.9165 | 0.7168 | 0.0306 | 0.058 | 7.8% | 0.1 s |
| Item-based CF | 0.8972 | 0.6940 | **0.0363** | **0.073** | **67.3%** | 5.1 s |
| SVD (k=50) | **0.8627** | **0.6694** | 0.0196 | 0.044 | 15.7% | 34.5 s |

Headline: the best rating predictor (SVD) is the worst ranker. Full discussion in `report/Technical_Report.pdf`. Sample Top-10 recommendations with "because you rated X" explanations are in `results/sample_recommendations.json`.
