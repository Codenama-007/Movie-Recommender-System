# Movie Recommender System — Project Report

## 1. Project Overview

This project is a **hybrid content-based movie recommendation system**
that takes free-text input from a user (e.g. "sci-fi thriller", "romantic
comedy") and returns the closest-matching movies from a real-world
dataset sourced from IMDb. It is built as an interactive web application
using Streamlit.

The recommendation logic combines three signals, all computed with
classical machine learning rather than deep learning, which keeps the
system lightweight, fast, and easy to explain:

1. **TF-IDF cosine similarity** — exact keyword and genre overlap.
2. **Latent-factor similarity (Truncated SVD)** — semantic closeness in a
   compressed feature space, so related movies can match even when they
   share no exact words with the query.
3. **A damped baseline rating** — a quality prior that prevents movies
   with only a handful of votes from outranking well-established ones.

The second and third signals are adapted from the collaborative-filtering
paper *"Machine Learning Model for Movie Recommendation System"* (see
Section 9). They are adapted, not reproduced: the paper relies on
per-user ratings that the IMDb data does not contain.

## 2. Objective

To design and implement an end-to-end machine learning pipeline that:

1. Acquires a large, real-world movie dataset.
2. Cleans and preprocesses the text data (titles and genres).
3. Converts the text into numerical representations suitable for
   similarity comparison, including a latent-factor representation.
4. Estimates a reliable quality score for every movie from aggregate
   ratings and vote counts.
5. Accepts free-text user input describing a mood or genre preference.
6. Returns a ranked list of the most relevant movies, along with
   supporting visual information (poster, rating).

## 3. Dataset

**Source:** IMDb Non-Commercial Datasets
(`https://datasets.imdbws.com/`), an official, freely available data
export provided directly by IMDb for personal and non-commercial use.

**Files used:**
- `title.basics.tsv.gz` — title ID, title type, title name, release
  year, genres
- `title.ratings.tsv.gz` — title ID, average user rating, number of
  votes

**Size:** After merging and cleaning, the dataset contains approximately
**150,000 titles** (movies, TV series, shorts, and other title types),
each with a genre classification, an average rating, and a vote count.

**What the data does *not* contain:** individual users' ratings. Each
title has a single aggregate rating and a vote count, but there is no
record of which user rated which movie. This shapes the design of the
system (see Sections 6 and 9).

**Why this source, not live web scraping:** IMDb's website is protected
by an AWS WAF (Web Application Firewall) JavaScript challenge as of
2026, which blocks non-browser HTTP clients from scraping HTML pages
directly. IMDb's Terms of Service also prohibit scraping the live site
without express consent. IMDb instead publishes the same underlying
data — title names, genres, and ratings — as structured, downloadable
TSV files specifically for non-commercial projects like this one. Using
this official dataset avoids fragile HTML scraping (which breaks
whenever IMDb changes its markup) and legal/ToS concerns, while
providing a larger and cleaner dataset than scraping would realistically
produce.

## 4. Data Preprocessing

- Rows with missing titles or genres are removed.
- Titles with very few votes are excluded, since their ratings are
  statistically unreliable (a movie with 3 votes and a 10/10 rating is
  not meaningfully "good").
- Genre strings (originally comma-separated, e.g. `Action,Sci-Fi`) are
  converted to lowercase, space-separated tokens (`action sci-fi`) to
  match the tokenization expected by scikit-learn's text vectorizers.
- Title strings are lowercased and stripped of punctuation to normalize
  them for text matching.

## 5. Feature Engineering

### 5.1 TF-IDF vectorization

Each movie is represented as a single combined text string made up of
its cleaned title and its cleaned genres. Genre tokens are deliberately
**repeated multiple times** within this string (a configurable
`GENRE_WEIGHT` of 3) so that genre similarity contributes more strongly
to the match score than incidental overlap in title wording.

This text is vectorized using **TF-IDF (Term Frequency–Inverse Document
Frequency)** via scikit-learn's `TfidfVectorizer`, with English stop-word
removal, unigrams and bigrams (`ngram_range=(1,2)`), and a minimum
document frequency filter (`min_df=2`) to discard extremely rare tokens.
TF-IDF down-weights common, less-informative terms and up-weights terms
that are distinctive to a smaller subset of movies.

### 5.2 Baseline rating

A raw average rating is misleading when the number of votes is small.
To correct for this, every movie receives a **baseline rating**, adapted
from the paper's "BaselineOnly" predictor (`μ + bu + bi`):

```
baseline_i = μ + b_i
b_i        = votes_i × (rating_i − μ) / (λ + votes_i)
```

- `μ` is the **global mean rating**, weighted by vote count (the paper's
  "average rating in the training data").
- `b_i` is the movie's **bias**: how far its rating sits from the global
  mean, shrunk toward zero when it has few votes. `λ` (`BASELINE_REG`,
  set to 100) acts as a number of "virtual votes" pulling the rating
  toward the mean.
- The user bias `bu` from the paper is omitted, because there are no
  per-user ratings.

In practice, a 9.5 rating from 12 votes is pulled strongly toward the
mean, while a 9.0 rating from two million votes is barely changed.

### 5.3 Latent factors via Truncated SVD

The paper uses **Singular Value Decomposition** as a matrix-factorization
technique that represents users and movies in a shared latent-factor
space. Here the same idea is applied to the **movie × term TF-IDF
matrix** instead of a user × movie rating matrix (a technique also known
as Latent Semantic Analysis):

- `TruncatedSVD` (scikit-learn) compresses the TF-IDF matrix into
  `SVD_COMPONENTS` = 100 latent factors.
- Every movie is projected into this 100-dimensional space and its
  vector is normalized to unit length, so a dot product equals cosine
  similarity.
- Movies that use related vocabulary end up close together in this space
  even when they do not share exact words, which lets the system surface
  semantically related results that pure keyword matching would miss.

## 6. Recommendation Logic

At query time:

1. The user's free-text input is lowercased and vectorized using the
   **same fitted TF-IDF vectorizer** used on the dataset. If none of the
   words exist in the vocabulary, no results are returned.
2. **Keyword/genre signal:** cosine similarity is computed between the
   query vector and every movie's TF-IDF vector.
3. **Latent signal:** the query vector is projected into the SVD space
   (`query_vec @ components.T`), normalized, and compared with every
   movie's latent vector; negative similarities are clipped to zero.
4. The two similarities are **blended**:

   ```
   relevance = 0.7 × sim_tfidf + 0.3 × sim_latent
   ```

5. A **quality factor** is derived from the baseline rating and applied
   multiplicatively:

   ```
   quality = (1 − 0.3) + 0.3 × clip(baseline / 10, 0, 1)
   score   = relevance × quality
   ```

   Because the factor multiplies rather than adds, an irrelevant movie
   can never outrank a relevant one purely on rating. The factor ranges
   from 0.7 to 1.0, so quality acts as a gentle nudge, not a filter.
6. Results below the user's minimum-rating threshold (applied to the raw
   IMDb rating) are removed.
7. The remaining movies are ranked by `score`, and the top N (default 10)
   are returned.

The three weights (0.7, 0.3, 0.3) are hand-chosen defaults exposed as
constants at the top of `app.py`; they have not been tuned against a
labeled evaluation set.

## 7. Precomputation & Performance

Because the dataset contains roughly 150,000 rows, fitting the TF-IDF
vectorizer and the SVD is expensive. All of this is performed **offline,
once**, by a dedicated script (`build_model.py`), and the results are
serialized to disk with `joblib`:

- the fitted vectorizer and the TF-IDF matrix,
- the movie table, including the new `baseline` column,
- the SVD projection matrix and every movie's latent vector (stored as
  `float32` to reduce size).

The Streamlit application loads these precomputed artifacts at startup
instead of refitting anything. Per query, the work is a sparse
similarity computation plus one small matrix-vector product over the
latent vectors, which stays fast at this dataset size. Only the
blend weights in `app.py` can be changed without rebuilding the model.

## 8. User Interface and Poster Images

The front end is built with **Streamlit** and presents:
- A text input field where the user types a genre, mood, or keyword
  combination.
- Sidebar controls for the number of results and a minimum acceptable
  rating.
- Results displayed as a **card grid**, five cards per row, each
  showing the movie's poster, title and release year, and average rating
  and genre list.

Since the IMDb dataset does not include poster images, artwork is
fetched at query time from the **OMDb API** (`www.omdbapi.com`), using
each movie's IMDb ID (`tconst`) as the lookup key. Lookups are cached per
session to minimize redundant API calls, and a placeholder image is shown
for any title without a poster or if no API key is configured. The API
key is stored in a `.env` file and loaded with `python-dotenv`, so it is
never hard-coded in the source.

## 9. Relationship to the Reference Paper

**Reference:** M. Chenna Keshava, P. Narendra Reddy, S. Srinivasulu,
B. Dinesh Naik, *"Machine Learning Model for Movie Recommendation
System,"* International Journal of Engineering Research & Technology
(IJERT), Vol. 9, Issue 04, April 2020 (Paper ID IJERTV9IS040741).

**What the paper does:** It uses the Netflix Prize dataset (movie ID,
customer ID, rating, date) to build a **user-item sparse matrix**,
computes **user-user** and **item-item similarity** matrices with cosine
similarity, and predicts ratings with XGBoost, Surprise's BaselineOnly
and KNNBaseline, and matrix factorization (SVD and SVD++), evaluating
with RMSE and MAPE. SVD++ achieved the lowest test RMSE (about 1.0675),
and the authors note the experiments used a subset (about 10,000 users
and 1,000 movies) because of RAM limits.

**How this project relates to it:**

| Paper concept                     | In this project                                                                 |
|------------------------------------|-----------------------------------------------------------------------------------|
| Baseline predictor (μ + bu + bi)   | **Adapted:** `μ + b_i` computed from aggregate ratings and vote counts (no `bu`). |
| Item-item cosine similarity        | **Adapted:** cosine similarity between movies' text vectors, in TF-IDF space and in latent space, rather than between rating columns. |
| Matrix factorization (SVD)         | **Adapted:** Truncated SVD on the movie × term TF-IDF matrix instead of a user × movie rating matrix. |
| User-item sparse matrix            | Not applicable: IMDb data has no per-user ratings.                               |
| User-user similarity               | Not applicable, for the same reason.                                             |
| KNNBaseline, XGBoost, SVD++        | Not applicable: they predict individual users' ratings.                          |
| Cold-start problem                 | Not applicable in the same form; the system has no user profiles, and any new query is handled through text. |
| RMSE / MAPE evaluation             | Not applicable: there are no held-out user ratings to predict.                   |

This project should therefore be described as **content-based filtering
with ideas adapted from the paper**, not as collaborative filtering.

## 10. Technology Stack

| Layer                  | Technology                                                   |
|-------------------------|---------------------------------------------------------------|
| Data source              | IMDb Non-Commercial Datasets                                 |
| Data processing          | Python, pandas, NumPy                                        |
| Vectorization / ML        | scikit-learn (`TfidfVectorizer`, `TruncatedSVD`, cosine similarity) |
| Model persistence        | joblib                                                       |
| Poster images             | OMDb API (via `requests`)                                   |
| Configuration/secrets    | python-dotenv (`.env` file)                                  |
| Frontend / UI              | Streamlit                                                   |

## 11. Project Pipeline Summary

```
IMDb Non-Commercial Datasets
        │
        ▼
prepare_imdb_data.py     (download + clean)
        │
        ▼
imdb_movies_clean.csv
        │
        ▼
build_model.py           (TF-IDF fit + baseline ratings + Truncated SVD)
        │
        ▼
model/  vectorizer, matrix, movies (+ baseline), svd_components, svd_latent
        │
        ▼
app.py (Streamlit)       (query → TF-IDF + latent similarity → blend
                          × baseline quality → ranked card grid + posters)
```

## 12. Limitations & Future Work

**Limitations**
- **Content-based, not collaborative:** recommendations come from the
  text of titles and genres plus aggregate ratings. There are no user
  profiles or histories, so nothing is personalized.
- **Vocabulary-bound queries:** query words must appear in the
  vocabulary. For example, the genre tag is "romance", so the word
  "romantic" only matches titles that literally contain it. The latent
  space softens this but does not remove it.
- **Latent factors are learned from text, not behavior:** they capture
  which words tend to co-occur, not which movies people like together.
- **Untuned blend weights:** the 0.7 / 0.3 / 0.3 weights and the
  `λ = 100` damping are reasonable defaults, not values optimized
  against labeled data. There is no offline accuracy metric, because the
  dataset has no held-out user ratings.
- **No plot text:** only title and genre are used.
- **Poster availability and rate limits:** not all titles have an OMDb
  poster, and the free tier is capped at 1,000 requests per day.

**Possible future improvements**
- Add a dataset with real user ratings, such as MovieLens, whose
  `links.csv` maps its movie IDs to IMDb IDs and so joins directly onto
  the existing `tconst` column. That would enable the paper's actual
  user-item matrix, user-user and item-item similarity, and SVD/SVD++,
  along with RMSE/MAPE evaluation.
- Incorporate plot summaries for richer semantic matching.
- Add query normalization (lemmatization or a small synonym map) so
  "romantic" maps to "romance".
- Tune the blend weights against a labeled or user-judged test set.
- Experiment with sentence-embedding models as an alternative to
  TF-IDF + SVD.
- Persist poster URLs to a local cache file to avoid re-fetching across
  sessions.

## 13. Conclusion

This project demonstrates a complete, functional machine learning
pipeline: real-world data acquisition, preprocessing, feature
engineering (TF-IDF, damped baseline ratings, latent factors),
similarity-based ranking, and an interactive user interface. It borrows
the baseline-predictor and matrix-factorization ideas from the reference
paper and applies them to the data actually available, while being
explicit about which parts of the paper's collaborative-filtering
approach require user-level ratings and are left for future work.

## 14. References

1. M. Chenna Keshava, P. Narendra Reddy, S. Srinivasulu, B. Dinesh Naik.
   *Machine Learning Model for Movie Recommendation System.*
   International Journal of Engineering Research & Technology (IJERT),
   Vol. 9, Issue 04, April 2020. Paper ID IJERTV9IS040741.
2. IMDb Non-Commercial Datasets. https://datasets.imdbws.com/
3. OMDb API. https://www.omdbapi.com/