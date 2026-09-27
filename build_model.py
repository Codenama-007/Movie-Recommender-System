"""
build_model.py — precompute the TF-IDF model so the Streamlit app doesn't
have to refit it on every cold start.

Run this once (and again whenever imdb_movies_clean.csv changes):
    pip install pandas scikit-learn joblib
    python build_model.py

Produces:
    model/vectorizer.joblib   -> fitted TfidfVectorizer
    model/matrix.joblib       -> TF-IDF matrix (sparse) for all rows
    model/movies.joblib       -> the DataFrame columns needed at query time
"""

import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_PATH = "imdb_movies_clean.csv"
MODEL_DIR = "model"
GENRE_WEIGHT = 3  # keep in sync with app.py


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

    print(f"[fit] TF-IDF on {len(df):,} rows ...")
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
    )
    matrix = vectorizer.fit_transform(df["combined_text"])

    keep_cols = ["tconst", "title", "genres", "rating", "votes", "year", "type"]
    movies = df[keep_cols]

    print("[save] writing model files ...")
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer.joblib"))
    joblib.dump(matrix, os.path.join(MODEL_DIR, "matrix.joblib"))
    joblib.dump(movies, os.path.join(MODEL_DIR, "movies.joblib"))

    print(f"[done] saved to ./{MODEL_DIR}/")


if __name__ == "__main__":
    main()