"""
main.py — Multi-Model Movie Recommender (Streamlit)

A SEPARATE app from app.py, for comparing recommendations across the
multiple rating-prediction models built by build_multi_model.py (loaded
from ./model_multi/). Running this never touches app.py or ./model/.

Each movie is scored by blending two signals:
    1. TF-IDF + SVD-latent text relevance -> keyword/genre match to the
       query (same text-matching approach as app.py)
    2. The SELECTED model's predicted rating -> a quality prior, so a
       movie the chosen model rates highly edges out one it rates lower,
       when relevance is close (multiplicative, so an irrelevant movie
       can never win purely on predicted rating)

The sidebar shows every model's held-out accuracy (RMSE/MAPE, from
model_multi/metrics.json) so models can be compared directly.

Run:
    pip install streamlit pandas numpy scikit-learn joblib requests python-dotenv
    streamlit run main.py
"""

import os
import json
import joblib
import numpy as np
import requests
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

load_dotenv()

MODEL_DIR = "model_multi"
PLACEHOLDER_POSTER = "https://placehold.co/300x445?text=No+Poster"

TFIDF_WEIGHT = 0.7     # keyword/genre text relevance
LATENT_WEIGHT = 0.3    # SVD latent-factor text relevance
QUALITY_WEIGHT = 0.3   # how strongly the selected model's predicted rating nudges ranking

OMDB_API_KEY = os.getenv("OMDB_API_KEY", "")

MODEL_DISPLAY_NAMES = {
    "BaselineOnly": "Baseline Only (genre bias)",
    "KNNBaseline_Item": "KNNBaseline (Item)",
    "SVD": "SVD (Matrix Factorization)",
    "SVD++": "SVD++",
    "XGBoost": "XGBoost",
    "XGB_BSL": "XGBoost + Baseline",
    "XGB_BSL_KNN": "XGBoost + Baseline + KNN",
    "XGB_BSL_KNN_MF": "XGBoost + Baseline + KNN + MF",
    "XGB_KNN_MF": "XGBoost + KNN + MF",
}
MODEL_KEYS = list(MODEL_DISPLAY_NAMES.keys())


@st.cache_data(show_spinner=False)
def fetch_poster(tconst: str, api_key: str) -> str:
    if not api_key:
        return PLACEHOLDER_POSTER
    try:
        resp = requests.get(
            "https://www.omdbapi.com/",
            params={"i": tconst, "apikey": api_key},
            timeout=5,
        )
        data = resp.json()
        poster = data.get("Poster")
        if poster and poster != "N/A":
            return poster
    except requests.RequestException:
        pass
    return PLACEHOLDER_POSTER


@st.cache_resource
def load_model():
    vectorizer = joblib.load(os.path.join(MODEL_DIR, "vectorizer.joblib"))
    matrix = joblib.load(os.path.join(MODEL_DIR, "matrix.joblib"))
    df = joblib.load(os.path.join(MODEL_DIR, "movies.joblib"))
    components = joblib.load(os.path.join(MODEL_DIR, "svd_components.joblib"))
    latent = joblib.load(os.path.join(MODEL_DIR, "svd_latent.joblib"))
    with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
        metrics = json.load(f)
    return df, vectorizer, matrix, components, latent, metrics


def clean_query(text: str) -> str:
    return text.lower().strip()


def recommend(query: str, df: pd.DataFrame, vectorizer, matrix, components, latent,
              model_key: str, top_n: int = 10, min_rating: float = 0.0):
    query_clean = clean_query(query)
    if not query_clean:
        return pd.DataFrame()

    query_vec = vectorizer.transform([query_clean])
    if query_vec.nnz == 0:
        return pd.DataFrame()

    sim_tfidf = cosine_similarity(query_vec, matrix).ravel()

    q_latent = normalize(query_vec @ components.T)
    sim_latent = np.clip(latent @ q_latent.ravel(), 0.0, None)

    relevance = TFIDF_WEIGHT * sim_tfidf + LATENT_WEIGHT * sim_latent

    pred_col = f"pred_{model_key}"
    model_pred = df[pred_col].to_numpy()
    quality = (1 - QUALITY_WEIGHT) + QUALITY_WEIGHT * np.clip(model_pred / 10.0, 0.0, 1.0)
    score = relevance * quality

    rating = df["rating"].to_numpy()
    candidates = np.where((relevance > 0) & (rating >= min_rating))[0]
    if candidates.size == 0:
        return pd.DataFrame()

    order = candidates[np.argsort(-score[candidates], kind="stable")][:top_n]

    result = df.iloc[order].copy()
    result["similarity"] = relevance[order]
    result["model_pred"] = model_pred[order]
    result["score"] = score[order]
    return result[
        ["tconst", "title", "genres", "rating", "votes", "year", "type",
         "similarity", "model_pred", "score"]
    ]


# ---------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------

st.set_page_config(page_title="Multi-Model Movie Recommender", page_icon="🎛️", layout="wide")
st.title("🎛️ Multi-Model Movie Recommender")
st.caption(
    "Compare recommendations across multiple rating-prediction models "
    "(adapted from the reference paper). Type a genre/mood, pick a model."
)

try:
    df, vectorizer, matrix, components, latent, metrics = load_model()
except FileNotFoundError:
    st.error(
        f"Couldn't find model files in `./{MODEL_DIR}/`. Run "
        "`python build_multi_model.py` first (after `prepare_imdb_data.py`), "
        "then restart this app."
    )
    st.stop()

with st.sidebar:
    st.subheader("Settings")
    omdb_key = st.text_input(
        "OMDb API key", value=OMDB_API_KEY, type="password",
        help="Loaded from .env (OMDB_API_KEY).",
    )
    if not OMDB_API_KEY:
        st.warning("No OMDB_API_KEY found in .env — posters will show placeholders.")

    model_key = st.selectbox(
        "Model",
        options=MODEL_KEYS,
        format_func=lambda k: MODEL_DISPLAY_NAMES[k],
        help="Which rating-prediction model reranks the matched movies.",
    )
    m = metrics["models"][model_key]
    st.metric("Held-out RMSE", f"{m['rmse']:.3f}")
    st.metric("Held-out MAPE", f"{m['mape']:.1f}%")
    st.caption("Lower is more accurate. Measured on a held-out test split of real movies.")

    with st.expander("Compare all models"):
        rows = [
            {"Model": MODEL_DISPLAY_NAMES[k], "RMSE": metrics["models"][k]["rmse"],
             "MAPE": metrics["models"][k]["mape"]}
            for k in MODEL_KEYS
        ]
        cmp_df = pd.DataFrame(rows).sort_values("RMSE")
        st.dataframe(cmp_df, hide_index=True, use_container_width=True)

    top_n = st.slider("Results", min_value=5, max_value=20, value=10)
    min_rating = st.slider("Min rating", min_value=0.0, max_value=10.0, value=6.0, step=0.5)

query = st.text_input("What are you in the mood for?", placeholder="e.g. sci-fi thriller")

if query:
    results = recommend(query, df, vectorizer, matrix, components, latent,
                        model_key=model_key, top_n=top_n, min_rating=min_rating)
    if results.empty:
        st.warning("No matches found — try different or broader keywords, or lower the min rating.")
    else:
        st.subheader(f"Top {len(results)} matches for \u201c{query}\u201d — ranked by {MODEL_DISPLAY_NAMES[model_key]}")
        if not omdb_key:
            st.caption("Add an OMDb API key in the sidebar to load poster images.")

        cols_per_row = 5
        rows_ = [results.iloc[i:i + cols_per_row] for i in range(0, len(results), cols_per_row)]

        for row in rows_:
            cols = st.columns(cols_per_row)
            for col, (_, movie) in zip(cols, row.iterrows()):
                with col:
                    with st.container(border=True):
                        poster_url = fetch_poster(movie["tconst"], omdb_key)
                        st.image(poster_url, use_container_width=True)
                        st.markdown(f"**{movie['title']}** ({int(movie['year']) if pd.notna(movie['year']) else '—'})")
                        st.caption(f"⭐ actual {movie['rating']:.1f}  ·  model predicted {movie['model_pred']:.1f}")
                        st.caption(movie["genres"])
else:
    st.info("Enter something above to get recommendations.")