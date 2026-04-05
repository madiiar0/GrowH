from pathlib import Path
import pandas as pd

DATA_ROOT = Path("data")

all_posts = []
all_comments = []

for folder in DATA_ROOT.iterdir():
    if not folder.is_dir() or not folder.name.startswith("r_"):
        continue

    posts_path = folder / "posts.csv"
    comments_path = folder / "comments.csv"

    if posts_path.exists():
        posts = pd.read_csv(posts_path)
        posts["source_folder"] = folder.name
        all_posts.append(posts)

    if comments_path.exists():
        comments = pd.read_csv(comments_path)
        comments["source_folder"] = folder.name
        all_comments.append(comments)

posts_df = pd.concat(all_posts, ignore_index=True) if all_posts else pd.DataFrame()
comments_df = pd.concat(all_comments, ignore_index=True) if all_comments else pd.DataFrame()

if not posts_df.empty:
    if "cluster_id" not in posts_df.columns:
        posts_df["cluster_id"] = pd.NA
    if "topic_label" not in posts_df.columns:
        posts_df["topic_label"] = pd.NA

if not comments_df.empty:
    if "cluster_id" not in comments_df.columns:
        comments_df["cluster_id"] = pd.NA
    if "topic_label" not in comments_df.columns:
        comments_df["topic_label"] = pd.NA

posts_df.to_csv("all_posts.csv", index=False)
comments_df.to_csv("all_comments.csv", index=False)

print("all_posts.csv shape:", posts_df.shape)
print("all_comments.csv shape:", comments_df.shape)
print("Done.")