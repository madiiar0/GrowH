# GrowH

GrowH is a Reddit data collection and clustering workflow built around the `reddit-universal-scraper` project in this repository. The current pipeline is focused on collecting Claude / Anthropic related Reddit posts and comments, combining them into shared datasets, and preparing them for topic labeling and sentiment analysis.

## Project Structure

```text
GrowH/
├── README.md
└── reddit-universal-scraper/
    ├── reddit_claude_pipeline.py
    ├── combine_data.py
    ├── prepare_clustering_data.py
    ├── cluster_posts.py
    ├── cluster_posts_clean.py
    ├── pnn_label.py
    ├── add_positivity_score.py
    └── requirements.txt
```

## Prerequisites

- Python 3.10 or newer
- `pip`
- `venv`

## Setup

```bash
cd reddit-universal-scraper
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install scikit-learn vaderSentiment
```

Notes:
- `requirements.txt` already includes `pandas`, `requests`, `aiohttp`, `aiofiles`, `streamlit`, `openpyxl`, `pyarrow`, `fastapi`, and `uvicorn`.
- `scikit-learn` is required for clustering scripts.
- `vaderSentiment` is required for comment sentiment labeling.

## 1. Run the Reddit Scraper

From `reddit-universal-scraper/`:

```bash
python3 reddit_claude_pipeline.py --limit 50 --no-media --viral-only --min-score 100 --min-comments 20
```

What this does:
- Scrapes recent posts from the configured subreddit list.
- Uses the subreddit `/new.json` feed, so it prioritizes newly submitted posts.
- Filters for Claude / Anthropic related content.
- Optionally keeps only viral posts using score and comment thresholds.
- Saves output under `data/claude_filtered/` and subreddit-specific folders under `data/`.

Useful flags:
- `--limit`: number of posts fetched per subreddit
- `--min-score`: minimum Reddit score for `--viral-only`
- `--min-comments`: minimum comment count for `--viral-only`
- `--no-media`: skip media downloads

You can change the tracked topics in:
- `DEFAULT_SUBREDDITS`
- `DIRECT_TERMS`
- `COMPARISON_TERMS`

These variables are defined in [reddit_claude_pipeline.py](/Users/madiaraskarly/Desktop/Projects/GrowH/reddit-universal-scraper/reddit_claude_pipeline.py).

## 2. Combine Scraped Data

After scraping, combine subreddit-level CSV files into shared datasets:

```bash
python3 combine_data.py
```

This generates:
- `all_posts.csv`
- `all_comments.csv`

The script reads each `data/r_*/posts.csv` and `data/r_*/comments.csv` file and merges them into one combined posts file and one combined comments file.

## 3. Prepare Data for Clustering

```bash
python3 prepare_clustering_data.py
```

This creates:
- `artifacts/posts_for_clustering_v2.csv`

The script:
- loads `all_posts.csv` and `all_comments.csv`
- keeps mostly top-level comments
- selects the highest-scoring comments per post
- builds a single `document_text` field for downstream clustering

## 4. Cluster Posts

```bash
python3 cluster_posts.py
```

This creates:
- `artifacts/posts_clustered_v2.csv`
- `artifacts/cluster_summary_v2.csv`
- `artifacts/cluster_model_selection_v2.csv`

The script uses TF-IDF, dimensionality reduction, and `MiniBatchKMeans` to assign cluster IDs to posts.

## 5. Clean Cluster Labels

```bash
python3 cluster_posts_clean.py
```

This creates:
- `artifacts/posts_labeled_v2.csv`
- `artifacts/comments_labeled_v2.csv`
- `artifacts/cluster_summary_v2_labeled.csv`

This step maps cluster IDs back onto posts and comments.

## 6. Label Comment Sentiment

```bash
python3 pnn_label.py
```

This uses VADER sentiment analysis and creates:
- `comments_labeled_v3.csv`

## 7. Add Positivity Score to Posts

```bash
python3 add_positivity_score.py
```

Default inputs and output:
- input posts: `artifacts/posts_labeled_v2.csv`
- input comments: `artifacts/comments_labeled_v3.csv`
- output posts: `artifacts/posts_labeled_v3.csv`

This adds a `positivity_score` column based on the percentage of positive labeled comments for each post.

## Typical End-to-End Workflow

```bash
cd reddit-universal-scraper
source .venv/bin/activate

python3 reddit_claude_pipeline.py --limit 50 --no-media --viral-only --min-score 100 --min-comments 20
python3 combine_data.py
python3 prepare_clustering_data.py
python3 cluster_posts.py
python3 cluster_posts_clean.py
python3 pnn_label.py
python3 add_positivity_score.py
```

## Output Locations

- Raw scraper outputs: `reddit-universal-scraper/data/`
- Filtered Claude-related outputs: `reddit-universal-scraper/data/claude_filtered/`
- Combined datasets: `reddit-universal-scraper/all_posts.csv`, `reddit-universal-scraper/all_comments.csv`
- Clustering artifacts: `reddit-universal-scraper/artifacts/`

## Main Files

- [reddit_claude_pipeline.py](/Users/madiaraskarly/Desktop/Projects/GrowH/reddit-universal-scraper/reddit_claude_pipeline.py): scrape, filter, and save Claude-related Reddit content
- [combine_data.py](/Users/madiaraskarly/Desktop/Projects/GrowH/reddit-universal-scraper/combine_data.py): merge per-subreddit CSVs into combined datasets
- [prepare_clustering_data.py](/Users/madiaraskarly/Desktop/Projects/GrowH/reddit-universal-scraper/prepare_clustering_data.py): build clustering-ready document text
- [cluster_posts.py](/Users/madiaraskarly/Desktop/Projects/GrowH/reddit-universal-scraper/cluster_posts.py): cluster posts with TF-IDF and KMeans
- [cluster_posts_clean.py](/Users/madiaraskarly/Desktop/Projects/GrowH/reddit-universal-scraper/cluster_posts_clean.py): propagate cluster labels to posts and comments
- [pnn_label.py](/Users/madiaraskarly/Desktop/Projects/GrowH/reddit-universal-scraper/pnn_label.py): assign positive / neutral / negative sentiment to comments
- [add_positivity_score.py](/Users/madiaraskarly/Desktop/Projects/GrowH/reddit-universal-scraper/add_positivity_score.py): compute post-level positivity score
