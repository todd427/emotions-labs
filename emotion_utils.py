#!/usr/bin/env python3
"""
emotion_utils.py
----------------
Shared helper library for emotion-labs analytic scripts.

Provides:
    • load_conversations()  →  normalized DataFrame from JSONL logs
    • tag_emotion()         →  GPU-batched emotion inference
    • analyze_emotions()    →  daily aggregation & CSV export

Intended for import by investigator.py, detective.py, etc.
"""

import json
import glob
from pathlib import Path
import pandas as pd
from transformers import pipeline

# ============================================================
# LEXICONS & EMOTION VALENCE
# ============================================================

NEGATIVE_WORDS = {
    "can’t", "cannot", "hopeless", "tired", "fail", "worthless", "useless",
    "anxious", "panic", "alone", "stuck", "hate", "scared", "exhausted",
    "depressed", "pointless", "overwhelmed"
}

POSITIVE_WORDS = {
    "sleep", "rest", "walk", "talk", "friend", "music", "cook", "read",
    "exercise", "meditate", "breathe", "outside", "relax", "sun", "coffee"
}

VALENCE = {
    "joy": 1.0,
    "surprise": 0.5,
    "neutral": 0.0,
    "sadness": -1.0,
    "fear": -0.8,
    "anger": -0.7,
    "disgust": -0.6,
}

# ============================================================
# LOADERS
# ============================================================

def load_conversations(path_pattern: str) -> pd.DataFrame:
    """Load JSONL chat logs, normalize timestamps, and clean text."""
    records = []
    for file in glob.glob(path_pattern):
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    msg = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue

                text = msg.get("content") or msg.get("text") or ""
                if not isinstance(text, str) or not text.strip():
                    continue

                ts = msg.get("timestamp") or msg.get("created") or msg.get("create_time")
                role = msg.get("role", "user")

                # normalize timestamp
                if isinstance(ts, (int, float)):
                    ts = pd.to_datetime(ts, unit="s", utc=True)
                else:
                    ts = pd.to_datetime(ts, errors="coerce", utc=True)

                records.append({"timestamp": ts, "role": role, "content": text})

    df = pd.DataFrame(records)
    df = df.dropna(subset=["timestamp"])
    df = df[df["timestamp"] > pd.Timestamp("2000-01-01", tz="UTC")]
    df = df.sort_values("timestamp").reset_index(drop=True)

    if df.empty:
        raise ValueError("⚠️ No valid messages found in provided path.")
    return df


# ============================================================
# EMOTION TAGGING
# ============================================================

def tag_emotion(df: pd.DataFrame, batch_size: int = 16) -> pd.DataFrame:
    """
    Apply emotion tagging with DistilRoBERTa in GPU batches.
    Automatically handles text cleaning and coercion.
    """
    print("🧠 Loading emotion model (DistilRoBERTa)…")
    model = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None,
        device=0
    )

    df = df.copy()
    df["content"] = df["content"].astype(str).fillna("").map(str.strip)
    df = df[df["content"].str.len() > 0]

    if df.empty:
        print("❌ No messages to tag.")
        return df

    texts = df["content"].tolist()
    print(f"⚙️ Tagging {len(texts)} messages (batch_size={batch_size})…")

    results = model(texts, batch_size=batch_size, truncation=True)
    df["sentiment"] = [
        max(r, key=lambda x: x["score"])["label"] if isinstance(r, list) and r else "neutral"
        for r in results
    ]
    print(f"✅ Tagged → {df['sentiment'].value_counts().to_dict()}")
    return df


# ============================================================
# ANALYSIS
# ============================================================

def analyze_emotions(df: pd.DataFrame, start=None, end=None,
                     outpath: str = "logs/message_emotion_activity.csv") -> pd.DataFrame:
    """Aggregate daily message and emotion counts, filtered by date range."""
    if start or end:
        mask = pd.Series(True, index=df.index)
        if start:
            mask &= df["timestamp"] >= start
        if end:
            mask &= df["timestamp"] <= end
        df = df[mask]
        print(f"🕒 Filtered to {len(df)} messages between {start} and {end}")

    if df.empty:
        raise ValueError("No messages after filtering.")

    df["date"] = df["timestamp"].dt.date
    daily_counts = df.groupby("date").size().rename("total")
    emotion_counts = df.groupby(["date", "sentiment"]).size().unstack(fill_value=0)
    merged = pd.concat([emotion_counts, daily_counts], axis=1)

    Path(outpath).parent.mkdir(exist_ok=True)
    merged.to_csv(outpath)
    print(f"💾 Saved dataset → {outpath}")
    return merged

