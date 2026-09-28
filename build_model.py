"""
build_model.py — precompute the model so the Streamlit app doesn't have to
refit anything on startup.

Run this once (and again whenever imdb_movies_clean.csv changes):
    pip install pandas numpy scikit-learn joblib
    python build_model.py

Pipeline (adapted from the collaborative-filtering paper, using only the
aggregate IMDb data we have, i.e. one rating + vote count per title):

    1. TF-IDF          text -> sparse vectors (title + genres)
    2. Baseline        mu + b_i : global mean rating plus a damped per-movie
                       bias (the paper's "BaselineOnly" predictor, minus the
                       user bias because we have no per-user ratings)
    3. Truncated SVD   latent-factor representation of every movie
                       (the paper's matrix-factorization idea, applied to
                       the movie x term matrix instead of user x movie)

Produces (all in ./model/):
    vectorizer.joblib      fitted TfidfVectorizer
    matrix.joblib          TF-IDF matrix (sparse)
    movies.joblib          columns needed at query time (incl. `baseline`)
    svd_components.joblib  k x vocab projection used to map a query into latent space
    svd_latent.joblib      every movie in latent space (n x k, unit-length rows)
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

DATA_PATH = "imdb_movies_clean.csv"
MODEL_DIR = "model"

GENRE_WEIGHT = 3        # repeat genre tokens so they outweigh title words
SVD_COMPONENTS = 100    # number of latent factors
BASELINE_REG = 100      # damping: how many "virtual votes" pull a rating toward the global mean


def compute_baseline(df: pd.DataFrame):
    """Baseline rating per movie: mu + b_i.

    mu  = global mean rating, weighted by votes (the paper's "average rating
          in the training data").
    b_i = n * (rating - mu) / (BASELINE_REG + n)
          i.e. the movie's deviation from the mean, shrunk toward 0 when the
          movie has few votes. A 9.5 from 12 votes gets pulled toward the
          mean; a 9.0 from 2 million votes barely moves.
    """
    rating = df["rating"].to_numpy(dtype=float)
    votes = df["votes"].to_numpy(dtype=float)

    mu = float(np.average(rating, weights=votes))
    b_i = votes * (rating - mu) / (BASELINE_REG + votes)
    return mu, mu + b_i


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"[load] {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df["title_clean"] = df["title_clean"].fillna("")
    df["genres_clean"] = df["genres_clean"].fillna("")
    df["rating"] = df["rating"].fillna(0.0)
    df["votes"] = df["votes"].fillna(0)

    df["combined_text"] = (
        df["title_clean"] + " " +
        (df["genres_clean"] + " ") * GENRE_WEIGHT
    ).str.strip()

    df = df.reset_index(drop=True)

    # 1) TF-IDF ---------------------------------------------------------
    print(f"[fit] TF-IDF on {len(df):,} rows ...")
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
    )
    matrix = vectorizer.fit_transform(df["combined_text"])
    print(f"      matrix shape: {matrix.shape}")

    # 2) Baseline rating ------------------------------------------------
    print("[baseline] computing mu + b_i ...")
    mu, baseline = compute_baseline(df)
    df["baseline"] = baseline
    print(f"      global mean rating (mu): {mu:.3f}")

    # 3) Truncated SVD (latent factors) ---------------------------------
    n_components = min(SVD_COMPONENTS, matrix.shape[1] - 1)
    print(f"[svd] fitting TruncatedSVD with {n_components} components ...")
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd.fit(matrix)
    print(f"      variance explained: {svd.explained_variance_ratio_.sum():.1%}")

    latent = normalize(svd.transform(matrix)).astype(np.float32)   # unit rows -> dot = cosine
    components = svd.components_.astype(np.float32)

    # Save ---------------------------------------------------------------
    keep_cols = ["tconst", "title", "genres", "rating", "votes", "year", "type", "baseline"]
    movies = df[keep_cols]

    print("[save] writing model files ...")
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer.joblib"))
    joblib.dump(matrix, os.path.join(MODEL_DIR, "matrix.joblib"))
    joblib.dump(movies, os.path.join(MODEL_DIR, "movies.joblib"))
    joblib.dump(components, os.path.join(MODEL_DIR, "svd_components.joblib"))
    joblib.dump(latent, os.path.join(MODEL_DIR, "svd_latent.joblib"))

    print(f"[done] saved to ./{MODEL_DIR}/")


if __name__ == "__main__":
    main()