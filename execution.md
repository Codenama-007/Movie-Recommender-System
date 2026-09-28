# Execution Order

Run the files in this exact order. Each step depends on the output of
the previous one.

## Prerequisites (one-time setup)

```bash
pip install pandas numpy requests scikit-learn joblib streamlit python-dotenv
```

Create a `.env` file in the project folder (same folder as `app.py`)
containing:

```
OMDB_API_KEY=4671d65b
```

Add `.env`, `imdb_raw/`, `imdb_movies_clean.csv`, `model/`, and `venv/` to
`.gitignore` so large or private files aren't committed to version control.

---

## Step 1 — `prepare_imdb_data.py`

**Downloads and cleans the raw IMDb data.**

```bash
python prepare_imdb_data.py
```

(If your local copy of this script has a different name, such as
`scrapper.py`, run that instead — it's the same step.)

- Downloads `title.basics.tsv.gz` and `title.ratings.tsv.gz` into an
  `imdb_raw/` folder (skipped automatically if already downloaded).
- Produces **`imdb_movies_clean.csv`** in the project folder.
- Takes a minute or two depending on your internet connection (the raw
  files together are a few hundred MB).

Run this again only if you want to refresh the dataset with IMDb's
latest daily update.

---

## Step 2 — `build_model.py`

**Fits the TF-IDF model, computes the baseline ratings, fits the SVD latent
factors, and saves everything to disk.**

```bash
python build_model.py
```

- Requires `imdb_movies_clean.csv` from Step 1 in the same folder.
- Prints progress for each stage: TF-IDF → baseline (with the global mean
  rating) → SVD (with the variance explained by the latent factors).
- Produces a `model/` folder containing five files:
  - `vectorizer.joblib`
  - `matrix.joblib`
  - `movies.joblib`
  - `svd_components.joblib`
  - `svd_latent.joblib`
- Takes longer than the TF-IDF-only version because of the SVD step;
  expect from about a minute to a few minutes on ~150,000 rows,
  depending on your machine. The `model/` folder is also larger now.

Run this again if `imdb_movies_clean.csv` changes, or if you edit
`GENRE_WEIGHT`, `SVD_COMPONENTS`, or `BASELINE_REG` inside the script.
**If you are upgrading from the earlier TF-IDF-only version, you must
re-run this step** — the old `model/` folder doesn't contain the SVD files.

---

## Step 3 — `app.py`

**Launches the Streamlit web app.**

```bash
streamlit run app.py
```

- Requires all five files in `model/` from Step 2 and the `.env` file in
  the same directory.
- Opens automatically in your browser (typically at
  `http://localhost:8501`).
- Startup only loads the precomputed files — nothing is fitted at this
  stage.
- The blend weights (`TFIDF_WEIGHT`, `LATENT_WEIGHT`, `QUALITY_WEIGHT`)
  at the top of `app.py` can be changed freely; just restart the app,
  no rebuild required.

This is the only step you repeat during normal use/demoing. Steps 1 and
2 are one-time setup (re-run only when the underlying data or model
settings change).

---

## Full Command Sequence (copy-paste)

```bash
pip install pandas numpy requests scikit-learn joblib streamlit python-dotenv
python prepare_imdb_data.py
python build_model.py
streamlit run app.py
```

## Quick Troubleshooting

| Symptom                                        | Likely cause / fix                                                                 |
|-------------------------------------------------|--------------------------------------------------------------------------------------|
| `FileNotFoundError: imdb_movies_clean.csv`      | Run Step 1 first.                                                                    |
| App says "Couldn't find model files"            | Run Step 2 first — or re-run it if you just updated to the SVD version (the old `model/` folder is missing the SVD files). |
| Posters all show placeholder images             | Check `.env` exists, contains `OMDB_API_KEY`, and the key is activated (check the verification email from OMDb). |
| DNS/connection error downloading datasets       | Double-check the URL is `datasets.imdbws.com` (not `.imdb.com`).                     |
| "No matches found" for a query                  | None of the words are in the vocabulary, or the Min rating slider is too high. Try broader words (e.g. `comedy`, `sci-fi`) or lower the slider. |
| `git push` rejected: file exceeds 100 MB        | `imdb_movies_clean.csv` or `model/` was committed. Add them to `.gitignore` and remove them from git tracking (`git rm -r --cached`). |