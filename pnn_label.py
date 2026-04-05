import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# 1. Load file
df = pd.read_csv("artifacts/comments_labeled_v2.csv")

# 2. Create new column for sentiment label
df["sentiment"] = ""

# 3. Initialize VADER
analyzer = SentimentIntensityAnalyzer()

# 4. Function to convert text -> negative / neutral / positive
def get_sentiment_label(text):
    if pd.isna(text) or str(text).strip() == "":
        return "neutral"

    scores = analyzer.polarity_scores(str(text))
    compound = scores["compound"]

    if compound >= 0.05:
        return "positive"
    elif compound <= -0.05:
        return "negative"
    else:
        return "neutral"

# 5. Apply labeling to each comment body
df["sentiment"] = df["body"].apply(get_sentiment_label)

# 6. Save result
df.to_csv("comments_labeled_v3.csv", index=False)

print("Saved: comments_labeled_v3.csv")
print(df[["body", "sentiment"]].head(10))