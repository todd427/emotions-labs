#!/usr/bin/env python3
"""
investigator.py
----------------
Emotion-Activity Investigator:
Analyzes message volume and emotion distribution over time,
accepts optional date filters, and computes correlation between
emotions and conversational activity.
"""

import json
import glob
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
from datasets import Dataset


# ---------------------- LOADERS ----------------------

def load_conversations(path_pattern: str) -> pd.DataFrame:
    """Load chat history from JSONL files and normalize timestamps."""
    records = []
    for file in glob.glob(path_pattern):
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    msg = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue

                content = msg.get("content") or msg.get("text") or ""
                if not isinstance(content, str) or not content.strip():
                    continue

                role = msg.get("role", "user")
                ts = msg.get("timestamp") or msg.get("created") or msg.get("create_time")

                # Convert timestamps that look like epoch seconds
                if isinstance(ts, (int, float)):
                    ts = pd.to_datetime(ts, unit="s", utc=True)
                else:
                    ts = pd.to_datetime(ts, errors="coerce", utc=True)

                records.append({"timestamp": ts, "role": role, "content": content})

    df = pd.DataFrame(records)
    df = df.dropna(subset=["timestamp"])
    if df.empty:
        print("⚠️ No valid messages found — check your file structure or timestamps.")
        return df

    df = df[df["timestamp"] > pd.Timestamp("2000-01-01", tz="UTC")]
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ---------------------- EMOTION TAGGING ----------------------

print("🚀 Loading sentiment model (distilroberta-emotion)…")
sentiment_model = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None,
    device=0
)


def tag_emotion(df: pd.DataFrame) -> pd.DataFrame:
    print("⚙️ Tagging emotions in parallel (batched pipeline)…")
    if df.empty:
        print("❌ No messages to tag.")
        return df

    # sanitize text
    df = df.copy()
    df["content"] = (
        df["content"]
        .astype(str)
        .fillna("")
        .map(lambda x: x.strip() if isinstance(x, str) else "")
    )
    df = df[df["content"].str.len() > 0]
    if df.empty:
        print("⚠️ No valid text after cleaning. Exiting.")
        return df

    # ensure we hand pure Python list of strings to the pipeline
    texts = df["content"].astype(str).tolist()
    print(f"🧩 Preparing {len(texts)} cleaned text entries for sentiment tagging…")

    results = sentiment_model(texts, batch_size=16, truncation=True)

    labels = [
        max(r, key=lambda x: x["score"])["label"]
        if isinstance(r, list) and r else "neutral"
        for r in results
    ]
    df["sentiment"] = labels
    return df


# ---------------------- ANALYSIS & PLOTS ----------------------

def plot_activity(df: pd.DataFrame, start=None, end=None):
    """Plot message volume and emotion distribution over time with timeline on X-axis."""
    df = df.copy()
    if start or end:
        mask = pd.Series(True, index=df.index)
        if start:
            mask &= df["timestamp"] >= start
        if end:
            mask &= df["timestamp"] <= end
        df = df[mask]
        print(f"🕒 Filtered to {len(df)} messages between {start} and {end}")

    if df.empty:
        print("⚠️ No data after filtering — skipping plot.")
        return

    # Daily aggregation
    df["date"] = df["timestamp"].dt.date
    daily_counts = df.groupby("date").size()
    emotion_counts = df.groupby(["date", "sentiment"]).size().unstack(fill_value=0)

    # Correlation
    merged = emotion_counts.copy()
    merged["total"] = daily_counts
    correlations = merged.corrwith(merged["total"]).sort_values(ascending=False)
    print("\n📊 Correlation between emotions and total message activity:")
    print(correlations.round(3))

    outdir = Path("logs")
    outdir.mkdir(exist_ok=True)
    merged.to_csv(outdir / "message_emotion_activity.csv", index=True)
    print(f"💾 Saved merged data to {outdir / 'message_emotion_activity.csv'}")

    # Timeline plot
    plt.figure(figsize=(14, 7))
    plt.plot(daily_counts.index, daily_counts.values, color="black", linewidth=2, label="Total Activity")

    emotion_ratios = emotion_counts.div(emotion_counts.sum(axis=1), axis=0) * 100
    for emo in emotion_ratios.columns:
        plt.plot(
            emotion_ratios.index,
            emotion_ratios[emo].rolling(3, min_periods=1).mean(),
            label=f"{emo} (%)", alpha=0.7
        )

    plt.title("Timeline of Message Activity and Emotion Distribution")
    plt.xlabel("Date (Timeline)")
    plt.ylabel("Messages / Emotion (%)")
    plt.legend()
    plt.tight_layout()
    plt.grid(alpha=0.3)
    plt.savefig(outdir / "emotion_activity_timeline.png", dpi=300)
    plt.show()

    print(f"🖼️ Saved figure: {outdir / 'emotion_activity_timeline.png'}")


# ---------------------- MAIN ----------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Investigate emotional drift and message activity over time")
    parser.add_argument("path", nargs="?", default="logs/conversations_clean.jsonl", help="Path to JSONL conversations")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    df = load_conversations(args.path)
    print(f"📘 Loaded {len(df)} messages between {df['timestamp'].min()} and {df['timestamp'].max()}")

    if df.empty:
        exit(1)

    # Optional filters
    start = pd.to_datetime(args.start, utc=True) if args.start else None
    end = pd.to_datetime(args.end, utc=True) if args.end else None
    if start or end:
        mask = pd.Series(True, index=df.index)
        if start:
            mask &= df["timestamp"] >= start
        if end:
            mask &= df["timestamp"] <= end
        df = df[mask]
        print(f"⏱️ Filtered to {len(df)} messages after applying date range.")
        if df.empty:
            print("❌ No messages match this date range. Exiting.")
            exit(1)

    df = tag_emotion(df)
    print(f"✅ Tagged {len(df)} messages → {df['sentiment'].value_counts().to_dict()}")

    plot_activity(df, start=start, end=end)

