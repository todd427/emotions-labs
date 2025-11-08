import json, sys
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def tokenize(text):
    return set(w.lower() for w in text.split() if w.isalpha())

root = Path(sys.argv[1])
files = {f.stem.replace("conversations_",""): f for f in root.glob("conversations_*.jsonl")}
if "neutral" not in files:
    print("❌ Need a 'conversations_neutral.jsonl' baseline.")
    sys.exit(1)

# Load neutral corpus
neutral = [json.loads(l).get("rewritten") or json.loads(l).get("content") 
           for l in open(files["neutral"],encoding="utf-8") if l.strip()]

results = []
for emo, path in files.items():
    if emo == "neutral": continue
    emo_lines = [json.loads(l).get("rewritten") or json.loads(l).get("content") 
                 for l in open(path,encoding="utf-8") if l.strip()]
    # pair up equal counts
    n = min(len(neutral), len(emo_lines))
    j_scores, cos_scores = [], []
    for i in range(n):
        a, b = neutral[i], emo_lines[i]
        if not a or not b: continue
        # --- Jaccard
        ta, tb = tokenize(a), tokenize(b)
        if ta and tb:
            j_scores.append(len(ta & tb)/len(ta | tb))
        # --- TF-IDF cosine
        tfidf = TfidfVectorizer(stop_words="english", max_features=500)
        vecs = tfidf.fit_transform([a,b])
        sim = cosine_similarity(vecs[0], vecs[1])[0,0]
        cos_scores.append(sim)
    if j_scores:
        results.append({
            "emotion": emo,
            "mean_jaccard": sum(j_scores)/len(j_scores),
            "mean_cosine": sum(cos_scores)/len(cos_scores)
        })

df = pd.DataFrame(results)
print(df)
df.to_csv(root/"analysis"/"emotion_neutral_comparison.csv",index=False)

# visualize
import matplotlib.pyplot as plt
fig, ax1 = plt.subplots(figsize=(7,4))
ax1.bar(df["emotion"], df["mean_jaccard"], color="salmon", alpha=0.7, label="Jaccard")
ax1.plot(df["emotion"], df["mean_cosine"], color="teal", marker="o", label="TF-IDF cosine")
ax1.set_title("Lexical vs Semantic Similarity to Neutral Baseline")
ax1.set_ylabel("Mean similarity (0–1)")
ax1.legend()
plt.tight_layout()
plt.savefig(root/"analysis"/"neutral_similarity.png")
plt.show()

