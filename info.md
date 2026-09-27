# Movie Recommender System — Project Report

## 1. Project Overview

This project is a **content-based movie recommendation system** that
takes free-text input from a user (e.g. "sci-fi thriller", "romantic
comedy") and returns the closest-matching movies from a real-world
dataset sourced from IMDb. The system is built as an interactive web
application using Streamlit, with the underlying recommendation logic
implemented using classical NLP and machine learning techniques
(TF-IDF vectorization and cosine similarity) rather than a deep
learning model, making it lightweight, fast, and easy to explain.

## 2. Objective

To design and implement an end-to-end machine learning pipeline that:

1. Acquires a large, real-world movie dataset.
2. Cleans and preprocesses the text data (titles and genres).
3. Converts the text into a numerical representation suitable for
   similarity comparison.
4. Accepts free-text user input describing a mood or genre preference.
5. Returns a ranked list of the most relevant movies, along with
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
each with a genre classification and a community-sourced rating.

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

## 5. Feature Engineering & Vectorization

Each movie is represented as a single combined text string made up of
its cleaned title and its cleaned genres. Genre tokens are deliberately
**repeated multiple times** within this combined string (a configurable
`GENRE_WEIGHT` parameter) so that genre similarity contributes more
strongly to the final match score than incidental overlap in title
wording.

This combined text is vectorized using **TF-IDF (Term
Frequency–Inverse Document Frequency)**, implemented via scikit-learn's
`TfidfVectorizer`, with:
- English stop-word removal
- Unigrams and bigrams (`ngram_range=(1,2)`)
- A minimum document frequency filter to discard extremely rare tokens

TF-IDF was chosen over simple word counts because it down-weights
common, less-informative terms and up-weights terms that are more
distinctive to a smaller subset of movies — which is desirable when
matching specific genre/mood combinations.

## 6. Recommendation Logic

At query time:

1. The user's free-text input is cleaned using the same normalization
   applied to the dataset.
2. The cleaned query is vectorized using the **same fitted TF-IDF
   vectorizer** used on the dataset (ensuring the query and the dataset
   live in the same vector space).
3. **Cosine similarity** is computed between the query vector and every
   movie's TF-IDF vector, producing a similarity score between 0 and 1
   for each movie.
4. Results are filtered to exclude:
   - Movies with zero similarity to the query.
   - Movies below a user-configurable minimum rating threshold.
5. Remaining results are ranked by:
   1. Similarity score (primary)
   2. Average rating (tiebreaker)
   3. Number of votes (secondary tiebreaker)
6. The top N results (configurable, default 10) are returned.

## 7. Precomputation & Performance

Because the dataset contains roughly 150,000 rows, fitting the TF-IDF
vectorizer is a relatively expensive one-time operation. To keep the
application responsive, this fitting step is performed **offline**, once,
by a dedicated script (`build_model.py`), and the resulting vectorizer
and TF-IDF matrix are serialized to disk using `joblib`. The Streamlit
application loads these precomputed artifacts at startup instead of
refitting the model on every run, reducing startup time from tens of
seconds to under a second.

## 8. User Interface

The front end is built with **Streamlit** and presents:
- A text input field where the user types a genre, mood, or keyword
  combination.
- Sidebar controls for the number of results to display and a minimum
  acceptable rating.
- Results displayed as a **card grid**, five cards per row, each
  showing:
  - The movie's poster image
  - Title and release year
  - Average rating and genre list

## 9. Poster Image Integration

Since the IMDb dataset itself does not include poster images, poster
artwork is fetched at query time from the **OMDb API**
(`www.omdbapi.com`), using each movie's IMDb ID (`tconst`) — a field
already present in the IMDb dataset — as the lookup key. Poster lookups
are cached per session to minimize redundant API calls, and a
placeholder image is shown for any title without an available poster
or if no API key is configured.

## 10. Technology Stack

| Layer                  | Technology                          |
|-------------------------|-------------------------------------|
| Data source              | IMDb Non-Commercial Datasets       |
| Data processing          | Python, pandas                     |
| Vectorization / ML        | scikit-learn (`TfidfVectorizer`, cosine similarity) |
| Model persistence        | joblib                             |
| Poster images             | OMDb API (via `requests`)         |
| Configuration/secrets    | python-dotenv (`.env` file)        |
| Frontend / UI              | Streamlit                         |

## 11. Project Pipeline Summary

```
IMDb Non-Commercial Datasets
        │
        ▼
prepare_imdb_data.py   (download + clean)
        │
        ▼
imdb_movies_clean.csv
        │
        ▼
build_model.py         (TF-IDF fit + save)
        │
        ▼
model/ (vectorizer, matrix, movies)
        │
        ▼
app.py (Streamlit)     (user query → similarity ranking → card grid + posters)
```

## 12. Limitations & Future Work

- **Content-based only:** The system currently recommends based purely
  on textual similarity between the query and title/genre text. It does
  not use collaborative filtering (i.e. "users who liked X also liked
  Y"), so it cannot personalize recommendations based on user history.
- **No plot/overview text:** Recommendations are based on title and
  genre only; incorporating plot summaries could improve match quality
  for more nuanced queries (e.g. "movies about time loops").
- **Poster availability:** Not all titles have a poster available via
  OMDb's free tier, particularly older or obscure titles.
- **Rate limits:** OMDb's free tier is capped at 1,000 requests/day,
  which is sufficient for demo/personal use but would need a paid tier
  or alternative image source (e.g. TMDB) for larger-scale deployment.

**Possible future improvements:**
- Incorporate plot summaries for richer semantic matching.
- Add explicit genre filter/multi-select alongside free-text search.
- Persist poster URLs to a local cache file to avoid re-fetching across
  sessions.
- Experiment with embedding-based similarity (e.g. sentence embeddings)
  instead of, or alongside, TF-IDF for improved semantic matching.

## 13. Conclusion

This project demonstrates a complete, functional machine learning
pipeline — from real-world data acquisition through preprocessing,
feature engineering, similarity-based recommendation, and an interactive
user interface — built entirely with widely-used, well-documented Python
tools (pandas, scikit-learn, Streamlit). It provides a practical
foundation that can be extended with more advanced recommendation
techniques as needed.
