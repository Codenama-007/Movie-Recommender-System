"""
app.py — Movie Recommender (Streamlit)

Loads the precomputed model in ./model/ (built by build_model.py) and returns
the closest matches to whatever the user types in (e.g. "sci-fi thriller").

Each movie is scored by blending three signals:
    1. TF-IDF cosine similarity   -> exact keyword / genre overlap
    2. SVD latent-factor similarity -> semantically related movies that share
                                       no exact words with the query
    3. Baseline rating (mu + b_i)  -> a gentle quality prior, so a well-rated,
                                       well-voted movie edges out an obscure
                                       one when relevance is close

Run:
    pip install streamlit pandas numpy scikit-learn joblib requests python-dotenv
    streamlit run app.py
"""

import os
import joblib
import numpy as np
import requests
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

load_dotenv()  # reads .env in the working directory, if present

MODEL_DIR = "model"
PLACEHOLDER_POSTER = "https://placehold.co/300x445?text=No+Poster"

# Blend weights used in recommend()
TFIDF_WEIGHT = 0.7     # keyword/genre overlap (the original ranking signal)
LATENT_WEIGHT = 0.3    # SVD latent-factor similarity
QUALITY_WEIGHT = 0.3   # how strongly the baseline rating nudges the final score (0 = ignore)

# Loaded from .env (OMDB_API_KEY=...). See build_model.py note / README.
OMDB_API_KEY = os.getenv("OMDB_API_KEY", "")


@st.cache_data(show_spinner=False)
def fetch_poster(tconst: str, api_key: str) -> str:
    """Fetch a poster URL from OMDb by IMDb id. Cached per (tconst, key)
    so each movie is only fetched once per session."""
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


# ---------------------------------------------------------------------
# Load precomputed model (built by build_model.py). Cached so it's only
# read from disk once per app session.
# ---------------------------------------------------------------------

@st.cache_resource
def load_model():
    vectorizer = joblib.load(os.path.join(MODEL_DIR, "vectorizer.joblib"))
    matrix = joblib.load(os.path.join(MODEL_DIR, "matrix.joblib"))
    df = joblib.load(os.path.join(MODEL_DIR, "movies.joblib"))
    components = joblib.load(os.path.join(MODEL_DIR, "svd_components.joblib"))
    latent = joblib.load(os.path.join(MODEL_DIR, "svd_latent.joblib"))
    return df, vectorizer, matrix, components, latent


def clean_query(text: str) -> str:
    return text.lower().strip()


def recommend(query: str, df: pd.DataFrame, vectorizer, matrix, components, latent,
              top_n: int = 10, min_rating: float = 0.0):
    query_clean = clean_query(query)
    if not query_clean:
        return pd.DataFrame()

    query_vec = vectorizer.transform([query_clean])
    if query_vec.nnz == 0:
        # None of the words appear in the vocabulary (e.g. "asdfgh")
        return pd.DataFrame()

    # 1) Keyword / genre overlap: TF-IDF cosine similarity (original signal)
    sim_tfidf = cosine_similarity(query_vec, matrix).ravel()

    # 2) Latent-factor similarity: project the query into the same SVD space
    #    as the movies, then cosine similarity (rows are unit length, so a
    #    dot product is enough). Negative values are clipped to 0.
    q_latent = normalize(query_vec @ components.T)
    sim_latent = np.clip(latent @ q_latent.ravel(), 0.0, None)

    relevance = TFIDF_WEIGHT * sim_tfidf + LATENT_WEIGHT * sim_latent

    # 3) Baseline quality prior in [1 - QUALITY_WEIGHT, 1]. Multiplying (rather
    #    than adding) means an irrelevant movie can never win on rating alone.
    baseline = df["baseline"].to_numpy()
    quality = (1 - QUALITY_WEIGHT) + QUALITY_WEIGHT * np.clip(baseline / 10.0, 0.0, 1.0)
    score = relevance * quality

    rating = df["rating"].to_numpy()
    candidates = np.where((relevance > 0) & (rating >= min_rating))[0]
    if candidates.size == 0:
        return pd.DataFrame()

    order = candidates[np.argsort(-score[candidates], kind="stable")][:top_n]

    result = df.iloc[order].copy()
    result["similarity"] = relevance[order]
    result["score"] = score[order]
    return result[
        ["tconst", "title", "genres", "rating", "votes", "year", "type", "similarity", "score"]
    ]


# ---------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")
st.title("🎬 Movie Recommender")
st.caption("Type a mood, genre, or keywords (e.g. \"sci-fi thriller\", \"romantic comedy\") and get closest matches from IMDb data.")

try:
    df, vectorizer, matrix, components, latent = load_model()
except FileNotFoundError:
    st.error(
        f"Couldn't find model files in `./{MODEL_DIR}/`. Run "
        "`python build_model.py` first (after `prepare_imdb_data.py`), "
        "then restart this app."
    )
    st.stop()

with st.sidebar:
    st.subheader("Settings")
    omdb_key = st.text_input(
        "OMDb API key",
        value=OMDB_API_KEY,
        type="password",
        help="Loaded from .env (OMDB_API_KEY). Override here if you want to test a different key.",
    )
    if not OMDB_API_KEY:
        st.warning("No OMDB_API_KEY found in .env — posters will show placeholders.")
    top_n = st.slider("Results", min_value=5, max_value=20, value=10)
    min_rating = st.slider("Min rating", min_value=0.0, max_value=10.0, value=6.0, step=0.5)

query = st.text_input("What are you in the mood for?", placeholder="e.g. sci-fi thriller")

if query:
    results = recommend(query, df, vectorizer, matrix, components, latent,
                        top_n=top_n, min_rating=min_rating)
    if results.empty:
        st.warning("No matches found — try different or broader keywords, or lower the min rating.")
    else:
        st.subheader(f"Top {len(results)} matches for “{query}”")
        if not omdb_key:
            st.caption("Add an OMDb API key in the sidebar to load poster images.")

        cols_per_row = 5
        rows = [results.iloc[i:i + cols_per_row] for i in range(0, len(results), cols_per_row)]

        for row in rows:
            cols = st.columns(cols_per_row)
            for col, (_, movie) in zip(cols, row.iterrows()):
                with col:
                    with st.container(border=True):
                        poster_url = fetch_poster(movie["tconst"], omdb_key)
                        st.image(poster_url, use_container_width=True)
                        st.markdown(f"**{movie['title']}** ({int(movie['year']) if pd.notna(movie['year']) else '—'})")
                        st.caption(f"⭐ {movie['rating']:.1f}  ·  {movie['genres']}")
else:
    st.info("Enter something above to get recommendations.")