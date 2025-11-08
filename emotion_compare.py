#!/usr/bin/env python3
"""
emotion_compare.py — safer edition
Compares sentiment intensity across emotion-variant JSONLs.
"""

import sys, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

def sentiment_score(text: str) -> float:
    if not isinstance(text, str) or not text.strip():
        return 0.0
    exclaim = text.count("!") * 0.2
    upper = sum(1 for c in text if c.isupper()) / (len(text) + 1)
    length = min(len(text.split()) / 20, 1)
    return min(exclaim + upper + length, 1.0)

def load_variant(path: Path, emotion: str):
    """Load user messages from either nested or flat JSONL format."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                convo = json.loads(line)
                # Case 1: nested messages (old structure)
                if "messages" in convo:
                    for msg in convo["messages"]:
                        if msg.get("role") == "user" and msg.get("content"):
                            text = msg["content"].strip()
                            s = sentiment_score(text)
                            data.append({"emotion": emotion, "intensity": s})
                # Case 2: flat records (your current structure)
                elif convo.get("role") == "user" and convo.get("rewritten"):
                    text = convo["rewritten"].strip()
                    s = sentiment_score(text)
                    data.append({"emotion": emotion, "intensity": s})
            except Exception as e:
                print(f"⚠️ Error parsing line in {path.name}: {e}")
    return pd.DataFrame(data)

def main(base_dir: Path):
    print(f"Analyzing emotion variants in {base_dir}")
    dfs = []
    for emo in EMOTIONS:
        f = base_dir / f"conversations_{emo}.jsonl"
        if not f.exists():
            print(f"⚠️ Missing file: {f}")
            continue
        df = load_variant(f, emo)
        if not df.empty:
            dfs.append(df)
        else:
            print(f"⚠️ No user messages found in {f.name}")

    if not dfs:
        print("❌ No valid data found. Exiting.")
        return

    df = pd.concat(dfs, ignore_index=True)
    if "emotion" not in df.columns:
        print("❌ No emotion column in data. Check input JSONLs.")
        return

    summary = df.groupby("emotion", as_index=False)["intensity"].mean()
    print("\nMean intensity by emotion variant:\n", summary)

    csv_path = base_dir / "emotion_intensity_summary.csv"
    summary.to_csv(csv_path, index=False)
    print(f"✅ Saved CSV to {csv_path}")

    # Radar chart
    categories = summary["emotion"].tolist()
    values = summary["intensity"].tolist()
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(categories) + 1)

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    plt.xticks(angles[:-1], categories, color="grey", size=12)
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75, 1.0], ["0.25", "0.5", "0.75", "1.0"], color="grey", size=10)
    plt.ylim(0, 1)

    ax.plot(angles, values, linewidth=2, linestyle="solid", color="#3498db")
    ax.fill(angles, values, color="#3498db", alpha=0.25)

    plt.title("Average Sentiment Intensity by Emotion Variant", size=16, color="#2c3e50", y=1.08)
    out_path = base_dir / "emotion_intensity_radar.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"📊 Radar chart saved to {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python emotion_compare.py ./logs/emotion_variants")
        sys.exit(1)
    main(Path(sys.argv[1]))

