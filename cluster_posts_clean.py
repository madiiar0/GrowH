import pandas as pd

summary = pd.read_csv("artifacts/cluster_summary_v2.csv")
clustered_posts = pd.read_csv("artifacts/posts_clustered_v2.csv")
all_posts = pd.read_csv("all_posts.csv")
comments = pd.read_csv("all_comments.csv")

# build label per cluster
summary["topic_label"] = summary.apply(
    lambda r: f"c{int(r['cluster_id']):03d}__" + "__".join(
        [t.strip().replace(" ", "_") for t in str(r["top_terms"]).split(",")[:3]]
    ),
    axis=1,
)

label_map = dict(zip(summary["cluster_id"], summary["topic_label"]))

clustered_posts["cluster_id"] = pd.to_numeric(
    clustered_posts["cluster_id"], errors="coerce"
).astype("Int64")
clustered_posts["topic_label"] = clustered_posts["cluster_id"].map(label_map)

# remove old empty columns from all_posts/comments before merge
all_posts = all_posts.drop(columns=["cluster_id", "topic_label"], errors="ignore")
comments = comments.drop(columns=["cluster_id", "topic_label"], errors="ignore")

# build lookup from clustered posts
# prefer id if available, otherwise fallback to permalink
if "id" in clustered_posts.columns and "id" in all_posts.columns:
    post_lookup = clustered_posts[["id", "permalink", "cluster_id", "topic_label"]].drop_duplicates(subset=["id"])
    labeled_posts = all_posts.merge(
        post_lookup[["id", "cluster_id", "topic_label"]],
        on="id",
        how="left",
        validate="one_to_one",
    )
else:
    post_lookup = clustered_posts[["permalink", "cluster_id", "topic_label"]].drop_duplicates(subset=["permalink"])
    labeled_posts = all_posts.merge(
        post_lookup,
        on="permalink",
        how="left",
        validate="one_to_one",
    )

# label comments from parent post permalink
comments = comments.merge(
    post_lookup[["permalink", "cluster_id", "topic_label"]].drop_duplicates(subset=["permalink"]),
    left_on="post_permalink",
    right_on="permalink",
    how="left",
    validate="many_to_one",
)

comments = comments.drop(columns=["permalink"], errors="ignore")

# save
labeled_posts.to_csv("artifacts/posts_labeled_v2.csv", index=False)
comments.to_csv("artifacts/comments_labeled_v2.csv", index=False)
summary.to_csv("artifacts/cluster_summary_v2_labeled.csv", index=False)

print("unique post labels:", labeled_posts["topic_label"].nunique(dropna=True))
print("unique comment labels:", comments["topic_label"].nunique(dropna=True))
print("saved: artifacts/posts_labeled_v2.csv")
print("saved: artifacts/comments_labeled_v2.csv")