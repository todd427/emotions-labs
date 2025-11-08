import json, sys
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations

# Utility: simple tokenization
def tokenize(text):
    return set(word.lower() for word in text.split() if word.isalpha())

# Load all conversation files
root = Path(sys.argv[1])
texts = {}
for file in sorted(root.glob("conversations_*.jsonl")):
    emotion = file.stem.replace("conversations_", "")
    all_text = []
    with open(file, encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                t = data.get("content") or data.get("rewritten") or ""
                all_text.append(t)
            except Exception:
                continue
    texts[emotion] = " ".join(all_text)

# --- 1. Lexical Jaccard overlap ---
def jaccard(a, b):
    ta, tb = tokenize(a), tokenize(b)
    if not ta or not tb: return 0.0
    return len(ta & tb) / len(ta | tb)

pairs = []
for (e1, t1), (e2, t2) in combinations(texts.items(), 2):
    pairs.append({"emotion1": e1, "emotion2": e2, "jaccard": jaccard(t1, t2)})

jaccard_df = pd.DataFrame(pairs)

# --- 2. TF-IDF cosine similarity ---
emotions = list(texts.keys())
corpus = [texts[e] for e in emotions]
tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
tfidf_matrix = tfidf.fit_transform(corpus)
cosine_sim = cosine_similarity(tfidf_matrix)

cos_df = pd.DataFrame(cosine_sim, index=emotions, columns=emotions)

# --- 3. Save outputs ---
outdir = root / "analysis"
outdir.mkdir(exist_ok=True)
jaccard_df.to_csv(outdir / "lexical_jaccard_pairs.csv", index=False)
cos_df.to_csv(outdir / "tfidf_cosine_matrix.csv")

print("\nLexical Jaccard (mean similarity):")
print(jaccard_df.groupby("emotion1")["jaccard"].mean())

print("\nTF-IDF Cosine Similarity Matrix:")
print(cos_df.round(3))

# --- 4. Visualize heatmaps (optional) ---
import matplotlib.pyplot as plt, seaborn as sns
plt.figure(figsize=(8,6))
sns.heatmap(cos_df, annot=True, cmap="coolwarm", vmin=0, vmax=1)
plt.title("Semantic (TF-IDF) Similarity Between Emotional Variants")
plt.tight_layout()
plt.savefig(outdir / "tfidf_similarity_heatmap.png")

plt.figure(figsize=(8,5))
sns.boxplot(data=jaccard_df, x="emotion1", y="jaccard")
plt.title("Lexical Overlap (Jaccard) Between Emotion Variants")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(outdir / "jaccard_overlap_boxplot.png")
plt.show()

