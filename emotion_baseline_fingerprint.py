# emotion_baseline_fingerprint.py (fixed)
import pandas as pd, json, sys
from pathlib import Path
import matplotlib.pyplot as plt

root = Path(sys.argv[1])
records = []

for file in sorted(root.glob("conversations_*.jsonl")):
    emotion = file.stem.replace("conversations_", "")
    with open(file, encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                if "content" in data:
                    text = data["content"]
                elif "rewritten" in data:
                    text = data["rewritten"]
                else:
                    continue
                if text.strip():
                    avg_len = len(text.split())
                    records.append({"emotion": emotion, "avg_length": avg_len})
            except Exception:
                continue

if not records:
    print("❌ No records found — check JSON structure.")
    sys.exit(1)

df = pd.DataFrame(records)
summary = df.groupby("emotion")["avg_length"].mean().reset_index()
print(summary)

# Save for plotting
summary.to_csv(root / "emotion_baselines.csv", index=False)

plt.bar(summary["emotion"], summary["avg_length"], color="teal")
plt.title("Average User Utterance Length by Emotion Variant")
plt.ylabel("Mean words per message")
plt.tight_layout()
plt.savefig(root / "emotion_baseline_lengths.png")
plt.show()

