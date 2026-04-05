from pathlib import Path
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

ARTIFACTS_DIR = Path("artifacts")
INPUT_PATH = ARTIFACTS_DIR / "posts_for_clustering_v2.csv"
OUTPUT_PATH = ARTIFACTS_DIR / "posts_clustered_v2.csv"
SUMMARY_PATH = ARTIFACTS_DIR / "cluster_summary_v2.csv"
MODELSEL_PATH = ARTIFACTS_DIR / "cluster_model_selection_v2.csv"

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\br/\w+\b|\bu/\w+\b", " ", text)
    text = re.sub(r"\bpreview\b|\bredd\b|\bamp\b|\bpng\b|\bjpg\b|\bjpeg\b|\bwebp\b|\bgif\b|\bmp4\b", " ", text)
    text = re.sub(r"[`*_>#\[\]\(\)\{\}|\\/=:+-]+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

custom_stop_words = ENGLISH_STOP_WORDS.union({
    "claude", "anthropic", "ai",
    "just", "like", "really", "think", "know",
    "use", "using", "used", "com", "https", "http"
})

df = pd.read_csv(INPUT_PATH)
df["document_text"] = df["document_text"].fillna("").astype(str)
df["clean_text"] = df["document_text"].map(clean_text)

mask = df["clean_text"].str.strip() != ""
docs = df.loc[mask, "clean_text"]

vectorizer = TfidfVectorizer(
    stop_words=list(custom_stop_words),
    ngram_range=(1, 3),
    min_df=3,
    max_df=0.55,
    max_features=50000,
    sublinear_tf=True,
    lowercase=False,
    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_]{1,}\b",
)

X_tfidf = vectorizer.fit_transform(docs)

lsa = make_pipeline(
    TruncatedSVD(n_components=100, random_state=42),
    Normalizer(copy=False)
)
X = lsa.fit_transform(X_tfidf)

candidate_ks = [120, 150, 180, 220]
rows = []
best_score = -1
best_k = None
best_model = None
best_labels = None

for k in candidate_ks:
    model = MiniBatchKMeans(
        n_clusters=k,
        random_state=42,
        batch_size=1024,
        n_init=10,
    )
    labels = model.fit_predict(X)
    score = silhouette_score(X, labels, sample_size=min(5000, len(labels)), random_state=42)
    rows.append({"k": k, "silhouette_score": score})
    if score > best_score:
        best_score = score
        best_k = k
        best_model = model
        best_labels = labels

pd.DataFrame(rows).sort_values("silhouette_score", ascending=False).to_csv(MODELSEL_PATH, index=False)

df["cluster_id"] = pd.NA
df.loc[mask, "cluster_id"] = best_labels
df["cluster_id"] = df["cluster_id"].astype("Int64")

terms = vectorizer.get_feature_names_out()
svd = lsa.named_steps["truncatedsvd"]
approx_centers = svd.inverse_transform(best_model.cluster_centers_)
order_centroids = approx_centers.argsort(axis=1)[:, ::-1]

summary_rows = []
for cluster_id in range(best_k):
    cluster_mask = df["cluster_id"] == cluster_id
    top_terms = [terms[i] for i in order_centroids[cluster_id, :12]]
    sample_titles = df.loc[cluster_mask, "title"].fillna("").head(5).tolist()
    summary_rows.append({
        "cluster_id": cluster_id,
        "post_count": int(cluster_mask.sum()),
        "top_terms": ", ".join(top_terms),
        "sample_titles": " || ".join(sample_titles),
    })

summary_df = pd.DataFrame(summary_rows).sort_values("post_count", ascending=False)

df.to_csv(OUTPUT_PATH, index=False)
summary_df.to_csv(SUMMARY_PATH, index=False)

print("best_k:", best_k)
print("best_silhouette:", best_score)
print("unique cluster ids:", df["cluster_id"].nunique(dropna=True))
print("saved:", OUTPUT_PATH)
print("saved:", SUMMARY_PATH)
print("saved:", MODELSEL_PATH)