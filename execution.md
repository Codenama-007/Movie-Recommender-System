# Execution Order

Run the files in this exact order. Each step depends on the output of
the previous one.

## Prerequisites (one-time setup)

```bash
pip install pandas requests scikit-learn joblib streamlit python-dotenv
```

Create a `.env` file in the project folder (same folder as `app.py`)
containing:

```
OMDB_API_KEY=4671d65b
```

Add `.env` to `.gitignore` so it isn't committed to version control.

---

## Step 1 — `prepare_imdb_data.py`

**Downloads and cleans the raw IMDb data.**

```bash
python prepare_imdb_data.py
```

- Downloads `title.basics.tsv.gz` and `title.ratings.tsv.gz` into an
  `imdb_raw/` folder (skipped automatically if already downloaded).
- Produces **`imdb_movies_clean.csv`** in the project folder.
- Takes a minute or two depending on your internet connection (the raw
  files together are a few hundred MB).

Run this again only if you want to refresh the dataset with IMDb's
latest daily update.

---

## Step 2 — `build_model.py`

**Fits the TF-IDF model and saves it to disk.**

```bash
python build_model.py
```

- Requires `imdb_movies_clean.csv` from Step 1 to exist in the same
  folder.
- Produces a `model/` folder containing:
  - `vectorizer.joblib`
  - `matrix.joblib`
  - `movies.joblib`
- Takes anywhere from a few seconds to around a minute depending on your
  machine, since it's fitting TF-IDF across ~150,000 rows.

Run this again only if `imdb_movies_clean.csv` changes, or if you modify
`GENRE_WEIGHT` or any vectorizer settings inside the script.

---

## Step 3 — `app.py`

**Launches the Streamlit web app.**

```bash
streamlit run app.py
```

- Requires the `model/` folder from Step 2 and the `.env` file to exist
  in the same directory.
- Opens automatically in your browser (typically at
  `http://localhost:8501`).
- Loading the model files is near-instant since no fitting happens at
  this stage — everything was precomputed in Step 2.

This is the only step you repeat during normal use/demoing. Steps 1 and
2 are one-time setup (re-run only when the underlying data or model
config changes).

---

## Full Command Sequence (copy-paste)

```bash
pip install pandas requests scikit-learn joblib streamlit python-dotenv
python prepare_imdb_data.py
python build_model.py
streamlit run app.py
```

## Quick Troubleshooting

| Symptom                                   | Likely cause / fix                                      |
|--------------------------------------------|-----------------------------------------------------------|
| `FileNotFoundError: imdb_movies_clean.csv` | Run Step 1 first.                                          |
| App says "Couldn't find model files"       | Run Step 2 first.                                          |
| Posters all show placeholder images        | Check `.env` exists, contains `OMDB_API_KEY`, and the key is activated (check the verification email from OMDb). |
| DNS/connection error downloading datasets  | Double-check the URL is `datasets.imdbws.com` (not `.imdb.com`). |
