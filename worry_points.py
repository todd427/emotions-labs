#!/usr/bin/env python3
"""
worry_points.py
---------------
Detect potential 'worry points' (emotional stress clusters)
in chat logs using emotion_utils helpers.

Usage:
    python worry_points.py "logs/conversations_clean.jsonl" --output logs/worry_points.jsonl
"""

import argparse
import json
from datetime import timedelta
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

import emotion_utils  # shared library


def rolling_zscore(series: pd.Series, window: int = 5) -> pd.Series:
    roll = series.rolling(window=window, min_periods=2)
    return (series - roll.mean()) / roll.std(ddof=0)


def detect_worry_dates(df: pd.DataFrame, emotions=None, z_thresh=1.5) -> set:
    if emotions is None:
        emotions = ["sadness", "fear", "anger"]
    df["date"] = df["timestamp"].dt.date
    worry_dates = set()
    for emo in emotions:
        daily_counts = (
            df[df["sentiment"] == emo]
            .groupby("date")
            .size()
            .reindex(df["date"].unique(), fill_value=0)
        )
        z = rolling_zscore(daily_counts, window=5)
        worry_dates |= set(z[z > z_thresh].index)
    return worry_dates


def keyword_counts(texts, vocab):
    words = " ".join(texts).lower().split()
    return Counter(w for w in words if w in vocab)


def summarize_window(texts):
    neg = keyword_counts(texts, emotion_utils.NEGATIVE_WORDS)
    pos = keyword_counts(texts, emotion_utils.POSITIVE_WORDS)
    return {
        "negative_keywords": dict(neg.most_common(5)),
        "positive_keywords": dict(pos.most_common(5)),
    }

def compute_recovery(df, start, end, lookahead_days=5):
    """Compute average emotional valence before/after a worry window."""
    df = df.copy()
    df["valence"] = df["sentiment"].map(emotion_utils.VALENCE).fillna(0.0)

    # Mean valence during cluster
    cluster_val = df[(df["timestamp"] >= start) & (df["timestamp"] < end)]["valence"].mean()

    # Mean valence in lookahead period
    after_start = end
    after_end = end + timedelta(days=lookahead_days)
    after_val = df[(df["timestamp"] >= after_start) & (df["timestamp"] < after_end)]["valence"].mean()

    if np.isnan(cluster_val) or np.isnan(after_val):
        return None

    return round(after_val - cluster_val, 3)


def safe_json(obj):
    """Convert NumPy / pandas scalars for safe JSON encoding."""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, (pd.Timestamp, np.datetime64)):
        return str(obj)
    if isinstance(obj, dict):
        return {k: safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [safe_json(v) for v in obj]
    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path pattern for JSONL logs")
    parser.add_argument("--output", default="logs/worry_points.jsonl")
    parser.add_argument("--window_days", type=int, default=2)
    parser.add_argument("--retag", action="store_true",
                        help="Force re-run of emotion tagging")
    args = parser.parse_args()

    df = emotion_utils.load_conversations(args.input)

    if args.retag or "sentiment" not in df.columns:
        df = emotion_utils.tag_emotion(df)

    df = df.dropna(subset=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    worry_dates = detect_worry_dates(df)
    print(f"🔍 Found {len(worry_dates)} potential worry clusters")

    results = []
    for d in sorted(worry_dates):
        start = pd.Timestamp(d).tz_localize("UTC")
        end = start + timedelta(days=args.window_days)
        window = df[(df["timestamp"] >= start) & (df["timestamp"] < end)]
        if window.empty:
            continue
        summary = summarize_window(window["content"].tolist())
        recovery_slope = compute_recovery(df, start, end)

        results.append({
            "date": str(d),
            "num_messages": int(len(window)),
            "emotions": {k: int(v) for k, v in window["sentiment"].value_counts().items()},
            "summary": safe_json(summary),
            "example_texts": window["content"].head(3).tolist(),
            "recovery_slope" : recovery_slope if recovery_slope is not None else 0.0
        })

    out_path = Path(args.output)
    out_path.parent.mkdir(exist_ok=True, parents=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(safe_json(r), ensure_ascii=False) + "\n")

    print(f"✅ Saved {len(results)} worry windows → {out_path}")


if __name__ == "__main__":
    main()

