import pandas as pd
import matplotlib.pyplot as plt
import sys

df = pd.read_csv(sys.argv[1])  # your emotion_neutral_comparison.csv
plt.figure(figsize=(6,6))
plt.scatter(df["mean_jaccard"], df["mean_cosine"], s=200, c="teal", alpha=0.7)

for _, r in df.iterrows():
    plt.text(r["mean_jaccard"]+0.002, r["mean_cosine"]+0.002, r["emotion"], fontsize=10, weight="bold")

plt.axvline(df["mean_jaccard"].mean(), color="gray", linestyle="--", alpha=0.5)
plt.axhline(df["mean_cosine"].mean(), color="gray", linestyle="--", alpha=0.5)
plt.xlabel("Lexical Similarity to Neutral (Jaccard)")
plt.ylabel("Semantic Similarity to Neutral (TF-IDF Cosine)")
plt.title("Emotion Geometry: Lexical vs Semantic Drift from Neutral Baseline")
plt.tight_layout()
plt.savefig("emotion_geometry.png")
plt.show()

