from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

VALID_SENTIMENTS = {"positive", "negative", "neutral"}


def normalize_permalink(series: pd.Series) -> pd.Series:
    """Normalize permalinks so joins are more reliable."""
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"https?://(?:www\.)?reddit\.com", "", regex=True)
        .str.rstrip("/")
    )


def pick_post_key(posts: pd.DataFrame) -> str:
    """Use post_permalink if present, otherwise fallback to permalink."""
    if "post_permalink" in posts.columns:
        return "post_permalink"
    if "permalink" in posts.columns:
        return "permalink"
    raise KeyError(
        "posts file must contain either 'post_permalink' or 'permalink' column"
    )


def compute_positivity_score(posts_path: Path, comments_path: Path, output_path: Path) -> None:
    posts = pd.read_csv(posts_path)
    comments = pd.read_csv(comments_path)

    post_key = pick_post_key(posts)
    if "post_permalink" not in comments.columns:
        raise KeyError("comments file must contain 'post_permalink' column")
    if "sentiment" not in comments.columns:
        raise KeyError("comments file must contain 'sentiment' column")

    posts["_join_permalink"] = normalize_permalink(posts[post_key])
    comments["_join_permalink"] = normalize_permalink(comments["post_permalink"])
    comments["sentiment"] = comments["sentiment"].astype(str).str.strip().str.lower()

    # Keep only labeled comments with recognized sentiment values.
    labeled_comments = comments[comments["sentiment"].isin(VALID_SENTIMENTS)].copy()

    sentiment_counts = (
        labeled_comments.groupby("_join_permalink")["sentiment"]
        .value_counts()
        .unstack(fill_value=0)
        .rename_axis(None, axis=1)
        .reset_index()
    )

    for col in ["positive", "negative", "neutral"]:
        if col not in sentiment_counts.columns:
            sentiment_counts[col] = 0

    sentiment_counts["total_labeled_comments"] = (
            sentiment_counts["positive"]
            + sentiment_counts["negative"]
            + sentiment_counts["neutral"]
    )

    # Positivity score = percentage of positive comments among all labeled comments.
    sentiment_counts["positivity_score"] = (
        sentiment_counts["positive"]
        .div(sentiment_counts["total_labeled_comments"])
        .mul(100)
        .round(2)
    )

    posts = posts.drop(columns=["positivity_score"], errors="ignore")
    merged = posts.merge(
        sentiment_counts[["_join_permalink", "positivity_score"]],
        on="_join_permalink",
        how="left",
        validate="one_to_one",
    )

    # If a post has no labeled comments, set score to 0.
    merged["positivity_score"] = merged["positivity_score"].fillna(0.0)

    merged = merged.drop(columns=["_join_permalink"])
    merged.to_csv(output_path, index=False)

    matched_posts = (merged["positivity_score"] > 0).sum()
    print(f"Saved: {output_path}")
    print(f"Posts: {len(merged)}")
    print(f"Comments with usable sentiment: {len(labeled_comments)}")
    print(f"Posts with positivity_score > 0: {matched_posts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Add a positivity_score column to posts_labeled_v2.csv using "
            "comment-level sentiment from comments_labeled_v3.csv"
        )
    )
    parser.add_argument(
        "--posts",
        default="artifacts/posts_labeled_v2.csv",
        help="Path to posts_labeled_v2.csv",
    )
    parser.add_argument(
        "--comments",
        default="artifacts/comments_labeled_v3.csv",
        help="Path to comments_labeled_v3.csv",
    )
    parser.add_argument(
        "--output",
        default="artifacts/posts_labeled_v3.csv",
        help="Output path for posts file with positivity_score",
    )
    args = parser.parse_args()

    compute_positivity_score(
        posts_path=Path(args.posts),
        comments_path=Path(args.comments),
        output_path=Path(args.output),
    )