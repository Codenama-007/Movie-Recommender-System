"""
prepare_imdb_data.py

Downloads IMDb's official non-commercial datasets and builds a single
clean CSV with: tconst, title, genres, rating, votes, year, type.

Source: https://datasets.imdb.com/ (official, free, updated daily,
non-commercial use). No scraping required.

Usage:
    pip install pandas requests
    python prepare_imdb_data.py

Output:
    imdb_movies_clean.csv
"""

import gzip
import io
import os
import requests
import pandas as pd

BASICS_URL = "https://datasets.imdbws.com/title.basics.tsv.gz"
RATINGS_URL = "https://datasets.imdbws.com/title.ratings.tsv.gz"

DATA_DIR = "imdb_raw"
OUTPUT_FILE = "imdb_movies_clean.csv"

MIN_VOTES = 10          # drop titles with almost no votes (very noisy ratings)
KEEP_TYPES = None       # e.g. ["movie"] to filter; None = keep everything (per your choice)

headers = {
    # Tells the server you are using a standard Chrome browser
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    # Tells the server what types of content your client handles
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    # Tells the server the languages you prefer
    "Accept-Language": "en-US,en;q=0.5",
}

def download(url: str, dest_path: str) -> None:
    if os.path.exists(dest_path):
        print(f"[skip] {dest_path} already exists")
        return
    print(f"[download] {url}")
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    print(f"[saved] {dest_path}")


def load_tsv_gz(path: str, usecols=None) -> pd.DataFrame:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return pd.read_csv(
            f,
            sep="\t",
            na_values="\\N",
            usecols=usecols,
            low_memory=False,
        )


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    basics_path = os.path.join(DATA_DIR, "title.basics.tsv.gz")
    ratings_path = os.path.join(DATA_DIR, "title.ratings.tsv.gz")

    download(BASICS_URL, basics_path)
    download(RATINGS_URL, ratings_path)

    print("[load] title.basics ...")
    basics = load_tsv_gz(
        basics_path,
        usecols=["tconst", "titleType", "primaryTitle", "startYear", "genres"],
    )

    print("[load] title.ratings ...")
    ratings = load_tsv_gz(ratings_path)  # tconst, averageRating, numVotes

    print(f"basics rows: {len(basics):,} | ratings rows: {len(ratings):,}")

    # Merge on tconst (inner join: only keep titles that have a rating)
    df = basics.merge(ratings, on="tconst", how="inner")
    print(f"after merge: {len(df):,} rows")

    # Drop rows with no genre info at all
    df = df.dropna(subset=["genres", "primaryTitle"])

    # Optional: filter by title type
    if KEEP_TYPES:
        df = df[df["titleType"].isin(KEEP_TYPES)]
        print(f"after type filter {KEEP_TYPES}: {len(df):,} rows")

    # Drop very low-vote titles (unreliable ratings)
    df = df[df["numVotes"] >= MIN_VOTES]
    print(f"after min-votes filter ({MIN_VOTES}): {len(df):,} rows")

    # Clean genres: "Action,Sci-Fi,Thriller" -> "action sci-fi thriller"
    # (space-separated is what you want for a TF-IDF/CountVectorizer step later)
    df["genres_clean"] = (
        df["genres"]
        .str.replace(",", " ", regex=False)
        .str.lower()
        .str.strip()
    )

    # Clean title text a bit (keep original + a normalized version)
    df["title_clean"] = (
        df["primaryTitle"]
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9\s]", "", regex=True)
        .str.strip()
    )

    # Rename for clarity
    df = df.rename(
        columns={
            "primaryTitle": "title",
            "startYear": "year",
            "titleType": "type",
            "averageRating": "rating",
            "numVotes": "votes",
        }
    )

    final_cols = [
        "tconst", "title", "title_clean", "genres", "genres_clean",
        "rating", "votes", "year", "type",
    ]
    df = df[final_cols].sort_values("votes", ascending=False)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"[done] wrote {len(df):,} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()