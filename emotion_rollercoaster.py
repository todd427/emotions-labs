#!/usr/bin/env python3
# emotion_rollercoaster.py — multi-emotion, sequential timeline version
import json
import glob
import pandas as pd
from datetime import datetime
from transformers import pipeline
import matplotlib.pyplot as plt
from pathlib import Path

def load_conversations(path_pattern):
    """Load JSON or JSONL logs and normalize fields."""
    records = []
    for file in glob.glob(path_pattern):
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    msg = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue

                content = (
                    msg.get("content")
                    or msg.get("message", {}).get("content")
                    or msg.get("text")
                    or ""
                )
                role = (
                    msg.get("role")
                    or msg.get("author", {}).get("role")
                    or msg.get("message", {}).get("author", {}).get("role")
                    or "user"
                )
                timestamp = (
                    msg.get("timestamp")
                    or msg.get("created")
                    or msg.get("create_time")
                    or None
                )
                records.append({"timestamp": timestamp, "role": role, "content": content})

    df = pd.DataFrame(records)
    if not len(df):
        raise ValueError("No messages loaded. Check your path or file format.")
    return df


# ---- Multi-emotion classifier ----
sentiment_model = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None,
    device=0
)

def tag_emotion(df):
    """Run multi-emotion tagging on text content."""
    labels = []
    for text in df["content"]:
        if not text.strip():
            labels.append("neutral")
            continue
        try:
            result = sentiment_model(text[:512])[0]
            label = max(result, key=lambda x: x["score"])
            labels.append(label["label"])
        except Exception:
            labels.append("neutral")
    df = df.copy()
    df["sentiment"] = labels
    return df


def preprocess_timestamps(df):
    """If timestamps missing, generate sequential hourly timeline."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    if df["timestamp"].isna().all():
        print("⚠️ No valid timestamps found — generating sequential timeline.")
        df["timestamp"] = pd.date_range(start="2024-01-01", periods=len(df), freq="H")
    else:
        df = df.sort_values("timestamp")
        # Fill NaNs sequentially if gaps exist
        df["timestamp"] = df["timestamp"].fillna(method="ffill")
        still_missing = df["timestamp"].isna().sum()
        if still_missing:
            filler = pd.date_range(
                start=df["timestamp"].min() or "2024-01-01",
                periods=len(df),
                freq="H",
            )
            df.loc[df["timestamp"].isna(), "timestamp"] = filler[df["timestamp"].isna()]

    # If timestamps are all the same (no variation), force sequential hours
    if df["timestamp"].nunique() < 2:
        print("⚠️ Timestamps not varying — creating synthetic sequence.")
        df["timestamp"] = pd.date_range(start="2024-01-01", periods=len(df), freq="H")

    return df


def plot_emotions(df):
    """Plot normalized emotion intensity over time."""
    df = preprocess_timestamps(df)
    df = df.sort_values("timestamp")
    counts = df.groupby(["timestamp", "sentiment"]).size().unstack(fill_value=0)

    # Normalize to percentages to avoid single emotion dominating
    normalized = counts.div(counts.sum(axis=1), axis=0) * 100
    rolling = normalized.rolling(window=200, min_periods=1).mean()

    plt.figure(figsize=(14, 7))
    rolling.plot(ax=plt.gca(), title="User’s Emotional Rollercoaster (Normalized Multi-Emotion)")
    plt.ylabel("Relative intensity (%)")
    plt.xlabel("Conversation Time (synthetic)")
    plt.yscale("log")
    plt.tight_layout()
    plt.show()

    # Summary + CSV
    summary = df["sentiment"].value_counts(normalize=True) * 100
    print("\nEmotion frequency summary (%):")
    print(summary.round(2))

    csv_path = Path("logs/emotions_tagged.csv")
    Path("logs").mkdir(exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"\nSaved tagged emotions to {csv_path}")


if __name__ == "__main__":
    path = "logs/*.jsonl"  # adjust if needed
    data = load_conversations(path)
    print(f"Loaded {len(data)} messages from {len(set(data['role']))} roles.")
    tagged = tag_emotion(data)
    plot_emotions(tagged)

