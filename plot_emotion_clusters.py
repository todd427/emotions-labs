#!/usr/bin/env python3
"""
plot_emotion_clusters.py
Visualize emotional composition per cluster window.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def load_worry_points(path):
    records = [json.loads(line) for line in open(path, "r", encoding="utf-8") if line.strip()]
    # Flatten emotion counts
    data = []
    for r in records:
        date = r["date"]
        for emo, count in r["emotions"].items():
            data.append({"date": date, "emotion": emo, "count": count})
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument("--path", default="logs/worry_points.jsonl")
    args = parser.parse_args()

    path = Path(args.path)
    df = load_worry_points(path)

    if args.start:
        df = df[df["date"] >= pd.to_datetime(args.start)]
    if args.end:
        df = df[df["date"] <= pd.to_datetime(args.end)]

    if df.empty:
        print("⚠️ No data in selected range.")
        return

    pivot = df.pivot_table(index="date", columns="emotion", values="count", fill_value=0).sort_index()
    pivot.plot(kind="bar", stacked=True, figsize=(14, 6))
    plt.title(f"Emotional Composition per Worry Cluster ({args.start or pivot.index.min().date()} → {args.end or pivot.index.max().date()})", fontsize=14)
    plt.xlabel("Date")
    plt.ylabel("Message Count")
    plt.legend(title="Emotion", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

