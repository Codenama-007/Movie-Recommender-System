"""
build_model.py — precompute the TF-IDF text model AND multiple rating-
prediction models adapted from the reference paper, plus their accuracy
metrics.

Run this once (and again whenever imdb_movies_clean.csv changes):
    pip install pandas numpy scikit-learn joblib xgboost
    python build_multi_model.py

This is a SEPARATE, independent pipeline from build_model.py / app.py. It
reads the same imdb_movies_clean.csv but writes to its own ./model_multi/
folder, so running it never touches or breaks the original ./model/ used
by app.py. Its own front end is main.py (not app.py).

--------------------------------------------------------------------------
WHY THIS ISN'T LITERALLY THE PAPER'S MODELS
--------------------------------------------------------------------------
The paper's XGBoost / BaselineOnly / KNNBaseline / SVD / SVD++ all predict
a specific USER's rating for a specific movie, trained on real
(user, movie, rating) triples. Our IMDb data has no per-user ratings —
only one aggregate rating + vote count per movie. So each model here is
re-purposed as a MOVIE-LEVEL rating regressor: given a movie's content
features (genre/title text -> SVD latent factors, vote count, year),
predict its actual IMDb rating. This keeps "accuracy" honest — it's
measured with a real train/test split of real movies and real ratings —
while adapting the paper's modeling ideas as closely as the data allows.

"User-based KNNBaseline" from the paper is dropped entirely: there is no
honest content-based substitute for user-user similarity when there are
no users.

Models produced (9):
    BaselineOnly, KNNBaseline_Item, SVD, SVD++, XGBoost,
    XGB_BSL, XGB_BSL_KNN, XGB_BSL_KNN_MF, XGB_KNN_MF

Produces (./model_multi/):
    vectorizer.joblib, matrix.joblib             TF-IDF (unchanged)
    svd_components.joblib, svd_latent.joblib     latent factors (unchanged)
    movies.joblib                                movie table + one
                                                  pred_<model> column per
                                                  model (production scores,
                                                  refit on all data)
    metrics.json                                 held-out RMSE/MAPE per
                                                  model, from a genuine
                                                  train/meta/test split
"""

import os
import json
import time
import datetime
import numpy as np
import pandas as pd
import joblib
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

DATA_PATH = "imdb_movies_clean.csv"
MODEL_DIR = "model_multi"

GENRE_WEIGHT = 3          # repeat genre tokens so they outweigh title words
SVD_COMPONENTS = 100       # TF-IDF latent factors
BASELINE_REG = 100         # damping ("virtual votes") for genre-bias baseline
KNN_K = 20                 # neighbors for the item-based KNN model
KNN_REFERENCE_SIZE = 20000  # cap on how many movies KNN searches against.
                             # Brute-force cosine KNN costs O(n_query * n_reference).
                             # At ~150k movies, querying against ALL of them (n_reference
                             # = n_query = 150k) is ~2.25 trillion multiply-adds -- this is
                             # what makes the script take hours. Capping the searchable
                             # reference set doesn't meaningfully hurt quality (KNN just
                             # needs a large enough representative pool of neighbors, not
                             # literally every movie) but cuts the cost proportionally.
RANDOM_STATE = 42

MODEL_NAMES = [
    "BaselineOnly", "KNNBaseline_Item", "SVD", "SVD++", "XGBoost",
    "XGB_BSL", "XGB_BSL_KNN", "XGB_BSL_KNN_MF", "XGB_KNN_MF",
]


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.clip(np.abs(y_true), 1e-6, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


# ---------------------------------------------------------------------
# Feature building
# ---------------------------------------------------------------------

def build_dense_features(df: pd.DataFrame, latent: np.ndarray):
    """latent factors + log(votes) + normalized year -> one feature matrix."""
    log_votes = np.log1p(df["votes"].to_numpy(dtype=float))
    year = df["year"].astype(float)
    year = year.fillna(year.median())
    year_norm = ((year - year.min()) / (year.max() - year.min() + 1e-9)).to_numpy()
    base = np.hstack([latent, log_votes.reshape(-1, 1), year_norm.reshape(-1, 1)])
    return base, log_votes


# ---------------------------------------------------------------------
# BaselineOnly — genre-level bias, learned on train only
# ---------------------------------------------------------------------

def fit_baseline(genres_clean: pd.Series, rating: np.ndarray, votes: np.ndarray):
    mu = float(np.average(rating, weights=votes))
    genre_vote_sum, genre_dev_sum = {}, {}
    for g_str, r, v in zip(genres_clean, rating, votes):
        for g in g_str.split():
            genre_vote_sum[g] = genre_vote_sum.get(g, 0.0) + v
            genre_dev_sum[g] = genre_dev_sum.get(g, 0.0) + v * (r - mu)
    genre_bias = {
        g: genre_dev_sum[g] / (BASELINE_REG + genre_vote_sum[g])
        for g in genre_vote_sum
    }
    return mu, genre_bias


def predict_baseline(mu: float, genre_bias: dict, genres_clean: pd.Series):
    preds = np.empty(len(genres_clean))
    for i, g_str in enumerate(genres_clean):
        genres = g_str.split()
        biases = [genre_bias.get(g, 0.0) for g in genres]
        preds[i] = mu + (np.mean(biases) if biases else 0.0)
    return preds


# ---------------------------------------------------------------------
# KNNBaseline (Item) — similarity-weighted average of nearest neighbors
# ---------------------------------------------------------------------

def fit_knn(train_latent: np.ndarray, train_rating: np.ndarray, k: int = KNN_K,
            reference_size: int = KNN_REFERENCE_SIZE):
    """Fits on a capped, randomly-sampled reference set rather than the full
    input, so query cost stays roughly constant instead of growing with the
    size of the dataset. Returns (nn_index, reference_ratings)."""
    if len(train_latent) > reference_size:
        rng = np.random.default_rng(RANDOM_STATE)
        ref_idx = rng.choice(len(train_latent), size=reference_size, replace=False)
    else:
        ref_idx = np.arange(len(train_latent))
    ref_latent = train_latent[ref_idx]
    ref_rating = train_rating[ref_idx]
    nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute", n_jobs=-1)
    nn.fit(ref_latent)
    return nn, ref_rating, ref_idx


def predict_knn(nn: NearestNeighbors, ref_rating: np.ndarray, query_latent: np.ndarray,
                 k: int = KNN_K, ref_idx: np.ndarray = None, query_idx: np.ndarray = None):
    """ref_idx: original-dataset indices of the (possibly subsampled) reference
    points the NN index was fit on. query_idx: original-dataset indices of the
    query rows. When both are given (the production self-query case, where a
    movie may itself be part of the reference sample), fetch k+1 neighbors and
    zero out any neighbor that IS the query movie itself, so a movie doesn't
    just echo its own rating back."""
    exclude_self = ref_idx is not None and query_idx is not None
    n_fetch = k + 1 if exclude_self else k
    n_fetch = min(n_fetch, len(ref_rating))
    dist, idx = nn.kneighbors(query_latent, n_neighbors=n_fetch)
    sims = np.clip(1 - dist, 1e-6, None)
    if exclude_self:
        neighbor_orig_idx = ref_idx[idx]
        mask = neighbor_orig_idx == query_idx.reshape(-1, 1)
        sims = np.where(mask, 0.0, sims)
    ratings = ref_rating[idx]
    weight_sum = sims.sum(axis=1)
    weight_sum[weight_sum == 0] = 1e-6
    return (sims * ratings).sum(axis=1) / weight_sum


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"[load] {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df["title_clean"] = df["title_clean"].fillna("")
    df["genres_clean"] = df["genres_clean"].fillna("")
    df["rating"] = df["rating"].fillna(0.0)
    df["votes"] = df["votes"].fillna(0)
    df["combined_text"] = (
        df["title_clean"] + " " + (df["genres_clean"] + " ") * GENRE_WEIGHT
    ).str.strip()
    df = df.reset_index(drop=True)
    n = len(df)

    # --- TF-IDF + SVD latent factors (unchanged from before) -----------
    t0 = time.time()
    print(f"[fit] TF-IDF on {n:,} rows ...")
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)
    matrix = vectorizer.fit_transform(df["combined_text"])
    print(f"      done in {time.time() - t0:.1f}s, vocab size {matrix.shape[1]:,}")

    n_components = min(SVD_COMPONENTS, matrix.shape[1] - 1)
    t0 = time.time()
    print(f"[svd] fitting TruncatedSVD with {n_components} components ...")
    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    svd.fit(matrix)
    print(f"      done in {time.time() - t0:.1f}s, variance explained: {svd.explained_variance_ratio_.sum():.1%}")
    latent = normalize(svd.transform(matrix)).astype(np.float32)
    components = svd.components_.astype(np.float32)

    dense_features, log_votes = build_dense_features(df, latent)
    rating = df["rating"].to_numpy(dtype=float)
    votes = df["votes"].to_numpy(dtype=float)

    # --- Train / meta / test split of REAL movies -----------------------
    idx_all = np.arange(n)
    idx_train, idx_temp = train_test_split(idx_all, test_size=0.4, random_state=RANDOM_STATE)
    idx_meta, idx_test = train_test_split(idx_temp, test_size=0.5, random_state=RANDOM_STATE)
    print(f"[split] train={len(idx_train):,}  meta={len(idx_meta):,}  test={len(idx_test):,}")

    def sub(idx):
        return dict(
            X=dense_features[idx], latent=latent[idx], rating=rating[idx],
            votes=votes[idx], genres=df["genres_clean"].iloc[idx],
        )

    train, meta, test = sub(idx_train), sub(idx_meta), sub(idx_test)

    metrics = {}
    base_preds_meta, base_preds_test, base_preds_full = {}, {}, {}

    # --- 1) BaselineOnly (genre bias, fit on train only) -----------------
    print("[fit] BaselineOnly ..."); t0 = time.time()
    mu, genre_bias = fit_baseline(train["genres"], train["rating"], train["votes"])
    base_preds_meta["BaselineOnly"] = predict_baseline(mu, genre_bias, meta["genres"])
    base_preds_test["BaselineOnly"] = predict_baseline(mu, genre_bias, test["genres"])
    metrics["BaselineOnly"] = dict(
        rmse=rmse(test["rating"], base_preds_test["BaselineOnly"]),
        mape=mape(test["rating"], base_preds_test["BaselineOnly"]),
    )
    print(f"      done in {time.time() - t0:.1f}s")

    # --- 2) KNNBaseline (Item), fit on train only ------------------------
    t0 = time.time()
    print(f"[fit] KNNBaseline_Item (reference pool capped at {KNN_REFERENCE_SIZE:,}) ...")
    knn_train, ref_rating_train, _ref_idx_train = fit_knn(train["latent"], train["rating"])
    base_preds_meta["KNNBaseline_Item"] = predict_knn(knn_train, ref_rating_train, meta["latent"])
    base_preds_test["KNNBaseline_Item"] = predict_knn(knn_train, ref_rating_train, test["latent"])
    metrics["KNNBaseline_Item"] = dict(
        rmse=rmse(test["rating"], base_preds_test["KNNBaseline_Item"]),
        mape=mape(test["rating"], base_preds_test["KNNBaseline_Item"]),
    )
    print(f"      done in {time.time() - t0:.1f}s")

    # --- 3) SVD (matrix factorization: linear regression on latent) -----
    print("[fit] SVD ..."); t0 = time.time()
    svd_reg = LinearRegression().fit(train["latent"], train["rating"])
    base_preds_meta["SVD"] = svd_reg.predict(meta["latent"])
    base_preds_test["SVD"] = svd_reg.predict(test["latent"])
    metrics["SVD"] = dict(
        rmse=rmse(test["rating"], base_preds_test["SVD"]),
        mape=mape(test["rating"], base_preds_test["SVD"]),
    )
    print(f"      done in {time.time() - t0:.1f}s")

    # --- 4) SVD++ (latent + implicit "votes" signal) ---------------------
    print("[fit] SVD++ ..."); t0 = time.time()
    svdpp_feat = lambda d: np.hstack([d["latent"], np.log1p(d["votes"]).reshape(-1, 1)])
    svdpp_reg = LinearRegression().fit(svdpp_feat(train), train["rating"])
    base_preds_meta["SVD++"] = svdpp_reg.predict(svdpp_feat(meta))
    base_preds_test["SVD++"] = svdpp_reg.predict(svdpp_feat(test))
    metrics["SVD++"] = dict(
        rmse=rmse(test["rating"], base_preds_test["SVD++"]),
        mape=mape(test["rating"], base_preds_test["SVD++"]),
    )
    print(f"      done in {time.time() - t0:.1f}s")

    # --- 5) XGBoost (base) ------------------------------------------------
    print("[fit] XGBoost ..."); t0 = time.time()
    def make_xgb():
        return XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, n_jobs=-1,
        )
    xgb_base = make_xgb().fit(train["X"], train["rating"])
    base_preds_meta["XGBoost"] = xgb_base.predict(meta["X"])
    base_preds_test["XGBoost"] = xgb_base.predict(test["X"])
    metrics["XGBoost"] = dict(
        rmse=rmse(test["rating"], base_preds_test["XGBoost"]),
        mape=mape(test["rating"], base_preds_test["XGBoost"]),
    )
    print(f"      done in {time.time() - t0:.1f}s")

    # --- 6-9) Combo models: XGBoost fit on META, using base models' -----
    #          out-of-sample predictions as extra features (no leakage:
    #          those predictions came from models trained on TRAIN only).
    def stacked(d_X, preds_dict, keys):
        extra = np.column_stack([preds_dict[k] for k in keys])
        return np.hstack([d_X, extra])

    combos = {
        "XGB_BSL": ["BaselineOnly"],
        "XGB_BSL_KNN": ["BaselineOnly", "KNNBaseline_Item"],
        "XGB_BSL_KNN_MF": ["BaselineOnly", "KNNBaseline_Item", "SVD"],
        "XGB_KNN_MF": ["KNNBaseline_Item", "SVD"],
    }
    combo_models = {}
    for name, keys in combos.items():
        t0 = time.time()
        print(f"[fit] {name} ...")
        X_meta_stacked = stacked(meta["X"], base_preds_meta, keys)
        X_test_stacked = stacked(test["X"], base_preds_test, keys)
        model = make_xgb().fit(X_meta_stacked, meta["rating"])
        pred_test = model.predict(X_test_stacked)
        metrics[name] = dict(rmse=rmse(test["rating"], pred_test), mape=mape(test["rating"], pred_test))
        combo_models[name] = model
        print(f"      done in {time.time() - t0:.1f}s")

    print("\n[metrics] (held-out test split, lower RMSE/MAPE = more accurate)")
    for name in MODEL_NAMES:
        m = metrics[name]
        print(f"   {name:<18s} RMSE={m['rmse']:.4f}   MAPE={m['mape']:.2f}%")

    # ---------------------------------------------------------------------
    # Refit every model on the FULL dataset for production per-movie
    # predictions (used by app.py's dropdown reranking). Held-out accuracy
    # above remains the number reported for comparison.
    # ---------------------------------------------------------------------
    print("\n[refit] production models on full dataset ...")
    full_X, full_latent, full_rating, full_votes = dense_features, latent, rating, votes
    full_genres = df["genres_clean"]

    mu_f, genre_bias_f = fit_baseline(full_genres, full_rating, full_votes)
    base_preds_full["BaselineOnly"] = predict_baseline(mu_f, genre_bias_f, full_genres)

    t0 = time.time()
    print(f"      KNNBaseline_Item production predictions (reference pool capped at {KNN_REFERENCE_SIZE:,}) ...")
    knn_full, ref_rating_full, ref_idx_full = fit_knn(full_latent, full_rating)
    query_idx_full = np.arange(n)
    base_preds_full["KNNBaseline_Item"] = predict_knn(
        knn_full, ref_rating_full, full_latent, ref_idx=ref_idx_full, query_idx=query_idx_full
    )
    print(f"      done in {time.time() - t0:.1f}s")

    svd_reg_f = LinearRegression().fit(full_latent, full_rating)
    base_preds_full["SVD"] = svd_reg_f.predict(full_latent)

    svdpp_reg_f = LinearRegression().fit(svdpp_feat({"latent": full_latent, "votes": full_votes}), full_rating)
    base_preds_full["SVD++"] = svdpp_reg_f.predict(svdpp_feat({"latent": full_latent, "votes": full_votes}))

    xgb_base_f = make_xgb().fit(full_X, full_rating)
    base_preds_full["XGBoost"] = xgb_base_f.predict(full_X)

    for name, keys in combos.items():
        X_full_stacked = stacked(full_X, base_preds_full, keys)
        model_f = make_xgb().fit(X_full_stacked, full_rating)
        base_preds_full[name] = model_f.predict(X_full_stacked)

    for name in MODEL_NAMES:
        df[f"pred_{name}"] = base_preds_full[name]

    # --- Save --------------------------------------------------------------
    keep_cols = ["tconst", "title", "genres", "rating", "votes", "year", "type"] + \
                [f"pred_{n}" for n in MODEL_NAMES]
    movies = df[keep_cols]

    print("\n[save] writing model files ...")
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer.joblib"))
    joblib.dump(matrix, os.path.join(MODEL_DIR, "matrix.joblib"))
    joblib.dump(movies, os.path.join(MODEL_DIR, "movies.joblib"))
    joblib.dump(components, os.path.join(MODEL_DIR, "svd_components.joblib"))
    joblib.dump(latent, os.path.join(MODEL_DIR, "svd_latent.joblib"))

    metrics_out = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "note": (
            "Models predict a MOVIE's rating from content features (no "
            "per-user ratings exist in the IMDb dataset), evaluated on a "
            "held-out test split of real movies. 'User-based KNNBaseline' "
            "from the reference paper is omitted (no content-based "
            "equivalent without real users)."
        ),
        "split_sizes": {"train": len(idx_train), "meta": len(idx_meta), "test": len(idx_test)},
        "models": metrics,
    }
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics_out, f, indent=2)

    print(f"[done] saved to ./{MODEL_DIR}/ (including metrics.json)")


if __name__ == "__main__":
    main()