from pathlib import Path
import pandas as pd

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

posts = pd.read_csv("all_posts.csv")
comments = pd.read_csv("all_comments.csv")

posts["title"] = posts["title"].fillna("")
posts["selftext"] = posts["selftext"].fillna("")
comments["body"] = comments["body"].fillna("")
comments["score"] = pd.to_numeric(comments["score"], errors="coerce").fillna(0)
comments["depth"] = pd.to_numeric(comments["depth"], errors="coerce").fillna(999)

# keep only top-level / near-top comments
comments = comments[comments["depth"] <= 1].copy()

# top 12 comments by score per post
comments = comments.sort_values(["post_permalink", "score"], ascending=[True, False])
top_comments = (
    comments.groupby("post_permalink", group_keys=False)
    .head(12)
    .groupby("post_permalink", as_index=False)
    .agg(
        selected_comments_text=("body", lambda x: " ".join(x.astype(str))),
        selected_comment_count=("body", "size"),
    )
)

df = posts.merge(
    top_comments,
    left_on="permalink",
    right_on="post_permalink",
    how="left"
)

df["selected_comments_text"] = df["selected_comments_text"].fillna("")
df["selected_comment_count"] = df["selected_comment_count"].fillna(0).astype(int)

df["document_text"] = (
        df["title"].astype(str).str.strip() + " "
        + df["title"].astype(str).str.strip() + " "
        + df["selftext"].astype(str).str.strip() + " "
        + df["selected_comments_text"].astype(str).str.strip()
).str.strip()

df["cluster_id"] = pd.NA
df["topic_label"] = pd.NA

df.to_csv(ARTIFACTS_DIR / "posts_for_clustering_v2.csv", index=False)
print("Saved artifacts/posts_for_clustering_v2.csv")
print("Rows:", len(df))