# Project Reference — File Guide

This document explains what each file in the project does, so you (or anyone
reviewing the project) can understand the codebase at a glance.

The recommender is a **hybrid content-based system**: it blends keyword/genre
matching (TF-IDF), latent-factor similarity (Truncated SVD), and a damped
baseline rating. The last two are adapted from the collaborative-filtering paper
*"Machine Learning Model for Movie Recommendation System"* (IJERT, 2020) — see
`info.md` for how the paper's ideas map onto this project.

---

## `prepare_imdb_data.py`

**Purpose:** Data acquisition + cleaning. *(Unchanged. If your local copy is
named differently, e.g. `scrapper.py`, its role is the same.)*

- Downloads two official IMDb non-commercial dataset files from
  `https://datasets.imdbws.com/`:
  - `title.basics.tsv.gz` — title id, type, title name, year, genres
  - `title.ratings.tsv.gz` — title id, average rating, number of votes
- Saves the raw downloads into an `imdb_raw/` folder so they aren't
  re-downloaded on subsequent runs.
- Loads both files with pandas and merges them on the shared `tconst`
  (IMDb title ID) column.
- Cleans the data:
  - Drops rows with missing title or genre.
  - Drops titles with very few votes (`MIN_VOTES`), since their ratings
    are statistically unreliable.
  - Lowercases and strips punctuation from the title (`title_clean`).
  - Converts comma-separated genres into lowercase, space-separated
    tokens (`genres_clean`) — the format needed for TF-IDF vectorization.
- Writes the final result to **`imdb_movies_clean.csv`**, the single
  source-of-truth dataset used by the rest of the project.

**Run this first**, and only re-run it if you want a fresher pull of
IMDb's data (it's updated daily upstream).

---

## `build_model.py`

**Purpose:** Precompute everything the app needs so it doesn't have to fit
any model at startup. **This file was extended** — it now builds three
things instead of one.

- Loads `imdb_movies_clean.csv`.
- Builds a `combined_text` column per movie: `title_clean` plus
  `genres_clean` repeated `GENRE_WEIGHT` (3) times, so genre matches count
  more than incidental title words.
- **Step 1 — TF-IDF:** fits a `TfidfVectorizer` (English stop words,
  unigrams + bigrams, `min_df=2`) on `combined_text`, producing a sparse
  TF-IDF matrix (one row per movie).
- **Step 2 — Baseline rating (`compute_baseline`)**: computes
  `baseline = μ + bᵢ` for every movie, where
  - `μ` is the global mean rating, weighted by votes
  - `bᵢ = votes × (rating − μ) / (BASELINE_REG + votes)`, a per-movie bias
    that is shrunk toward zero when a movie has few votes
  (`BASELINE_REG` = 100). This is the paper's "BaselineOnly" predictor
  without the user bias, since the IMDb data has no per-user ratings.
- **Step 3 — Truncated SVD:** fits `TruncatedSVD` with `SVD_COMPONENTS`
  (100) latent factors on the TF-IDF matrix. Every movie is projected into
  this latent space and its row is normalized to unit length (so a dot
  product equals cosine similarity). Stored as `float32` to save space.
- Saves five files into the `model/` folder using `joblib`:
  - `vectorizer.joblib` — the fitted TF-IDF vectorizer
  - `matrix.joblib` — the TF-IDF matrix for every movie
  - `movies.joblib` — the DataFrame columns needed to display and rank
    results: `tconst`, `title`, `genres`, `rating`, `votes`, `year`,
    `type`, and the new `baseline`
  - `svd_components.joblib` — the `k × vocabulary` projection used to map a
    user's query into the latent space
  - `svd_latent.joblib` — every movie in latent space (`n × k`,
    unit-length rows)

**Constants you can tune here** (re-run the script after changing them):
`GENRE_WEIGHT`, `SVD_COMPONENTS`, `BASELINE_REG`.

**Run this second**, after `prepare_imdb_data.py`, and again any time the
CSV changes or you change one of the constants above.

---

## `app.py`

**Purpose:** The Streamlit front end — what the user interacts with.
**The ranking logic was extended; the UI is unchanged.**

- Loads environment variables from `.env` (via `python-dotenv`),
  specifically `OMDB_API_KEY`.
- `load_model()`: loads the five precomputed files from `model/` using
  `joblib` (cached with `st.cache_resource`) — no refitting at startup.
- `fetch_poster(tconst, api_key)`: calls the OMDb API
  (`https://www.omdbapi.com/`) with a movie's IMDb ID to retrieve its
  poster URL. Falls back to a placeholder image if no key is set, the
  request fails, or OMDb has no poster. Cached per movie per session.
- `clean_query(text)`: lowercases and strips the user's input.
- `recommend(query, df, vectorizer, matrix, components, latent, top_n, min_rating)`:
  1. Vectorizes the query with the fitted TF-IDF vectorizer. If none of
     the words exist in the vocabulary (e.g. `asdfgh`), returns no results.
  2. **Keyword/genre signal:** cosine similarity between the query vector
     and every movie's TF-IDF vector.
  3. **Latent-factor signal:** projects the query into the SVD space
     (`query_vec @ components.T`), normalizes it, and takes the dot
     product with every movie's latent vector (negatives clipped to 0).
  4. **Blend:** `relevance = TFIDF_WEIGHT × tfidf + LATENT_WEIGHT × latent`
     (0.7 / 0.3).
  5. **Quality prior:** `quality = (1 − QUALITY_WEIGHT) + QUALITY_WEIGHT ×
     baseline/10` (`QUALITY_WEIGHT` = 0.3), then
     `score = relevance × quality`. Multiplying means an irrelevant movie
     can never win on rating alone.
  6. Applies the `min_rating` filter (on the raw IMDb rating), sorts by
     `score` descending, and returns the top `top_n` rows with extra
     columns `similarity` (the blended relevance) and `score`.
- **UI layout (unchanged):**
  - Sidebar: OMDb API key field (pre-filled from `.env`), a "Results"
    slider, and a "Min rating" slider.
  - Main area: a text input for the user's query (e.g. "sci-fi thriller").
  - Results render as a card grid (5 per row) — poster, title + year,
    rating + genres.

**Constants you can tune here** (just restart the app — no rebuild
needed): `TFIDF_WEIGHT`, `LATENT_WEIGHT`, `QUALITY_WEIGHT`.

**Run this last**, after the model has been built, using
`streamlit run app.py`.

---

## `.env`

**Purpose:** Stores the OMDb API key outside of the source code.

```
OMDB_API_KEY=4671d65b
```

Read automatically by `app.py` via `load_dotenv()`. Should be listed in
`.gitignore` and never committed to a public repository.

---

## File Dependency Summary

| File                   | Depends on (must exist first)           | Produces                                                                 |
|------------------------|------------------------------------------|---------------------------------------------------------------------------|
| `prepare_imdb_data.py` | Internet access to datasets.imdbws.com   | `imdb_movies_clean.csv`                                                   |
| `build_model.py`       | `imdb_movies_clean.csv`                  | `model/vectorizer.joblib`, `matrix.joblib`, `movies.joblib`, `svd_components.joblib`, `svd_latent.joblib` |
| `app.py`               | all five `model/` files, `.env`          | Running Streamlit app (recommendations)                                   |

## Which knob needs which step?

| You change…                                        | Re-run                                   |
|-----------------------------------------------------|-------------------------------------------|
| `GENRE_WEIGHT`, `SVD_COMPONENTS`, `BASELINE_REG`    | `build_model.py`, then restart the app    |
| `TFIDF_WEIGHT`, `LATENT_WEIGHT`, `QUALITY_WEIGHT`   | Just restart the app                      |
| `MIN_VOTES` / the source data                       | `prepare_imdb_data.py`, then `build_model.py` |

See `execution.md` for the exact run order and commands.