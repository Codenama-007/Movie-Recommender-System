# Project Reference — File Guide

This document explains what each file in the project does, so you (or anyone
reviewing the project) can understand the codebase at a glance.

---

## `prepare_imdb_data.py`

**Purpose:** Data acquisition + cleaning.

- Downloads two official IMDb non-commercial dataset files from
  `https://datasets.imdbws.com/`:
  - `title.basics.tsv.gz` — title id, type, title name, year, genres
  - `title.ratings.tsv.gz` — title id, average rating, number of votes
- Saves the raw downloads into an `imdb_raw/` folder so they aren't
  re-downloaded on subsequent runs.
- Loads both files with pandas, merges them on the shared `tconst`
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

**Purpose:** Precompute the recommendation model so the app doesn't have
to rebuild it every time it starts.

- Loads `imdb_movies_clean.csv`.
- Builds a `combined_text` column per movie by concatenating
  `title_clean` with `genres_clean` (genres repeated `GENRE_WEIGHT` times
  so genre matches count more heavily than title words).
- Fits a `TfidfVectorizer` (scikit-learn) on `combined_text` across the
  entire dataset (~150k rows), producing a sparse TF-IDF matrix.
- Saves three files into a `model/` folder using `joblib`:
  - `vectorizer.joblib` — the fitted TF-IDF vectorizer
  - `matrix.joblib` — the TF-IDF matrix for every movie
  - `movies.joblib` — the trimmed DataFrame (`tconst`, title, genres,
    rating, votes, year, type) needed to display results

**Run this second**, after `prepare_imdb_data.py`, and again any time the
CSV changes or you tweak `GENRE_WEIGHT`.

---

## `app.py`

**Purpose:** The Streamlit front end — this is what the user actually
interacts with.

- Loads environment variables from `.env` (via `python-dotenv`),
  specifically `OMDB_API_KEY`.
- Loads the precomputed model files from `model/` (`vectorizer.joblib`,
  `matrix.joblib`, `movies.joblib`) using `joblib` — no refitting at
  startup.
- `fetch_poster(tconst, api_key)`: calls the OMDb API
  (`https://www.omdbapi.com/`) with a movie's IMDb ID to retrieve its
  poster image URL. Falls back to a placeholder image if no key is set,
  the request fails, or OMDb has no poster for that title. Cached per
  movie per session (`st.cache_data`) so repeat lookups don't re-hit the
  API.
- `recommend(query, df, vectorizer, matrix, top_n, min_rating)`:
  - Cleans and vectorizes the user's typed query the same way the
    dataset was vectorized.
  - Computes cosine similarity between the query vector and every
    movie's TF-IDF vector.
  - Filters out zero-similarity results and anything below the
    `min_rating` threshold.
  - Sorts by similarity, then rating, then vote count (as tiebreakers).
  - Returns the top `top_n` matches.
- **UI layout:**
  - Sidebar: OMDb API key field (pre-filled from `.env`), a "Results"
    count slider, and a "Min rating" slider.
  - Main area: a text input for the user's query (e.g. "sci-fi
    thriller").
  - Results render as a card grid (5 per row) — each card shows the
    movie's poster, title + year, and rating + genres.

**Run this last**, after the model has been built, using
`streamlit run app.py`.

---

## `.env`

**Purpose:** Stores the OMDb API key outside of the source code.

```
OMDB_API_KEY=4671d65b
```

Read automatically by `app.py` via `load_dotenv()`. Should be added to
`.gitignore` and never committed to a public repository.

---

## File Dependency Summary

| File                  | Depends on (must exist first)     | Produces                                  |
|-----------------------|------------------------------------|--------------------------------------------|
| `prepare_imdb_data.py`| Internet access to datasets.imdbws.com | `imdb_movies_clean.csv`               |
| `build_model.py`      | `imdb_movies_clean.csv`            | `model/vectorizer.joblib`, `model/matrix.joblib`, `model/movies.joblib` |
| `app.py`              | `model/` files, `.env`             | Running Streamlit app (recommendations)   |

See `execution.md` for the exact run order and commands.
