#!/usr/bin/env python3
"""
make_worry_points.py
--------------------
Generate a 'worry point' analysis file (WP_*.jsonl) from any input chat log.

Usage:
    python make_worry_points.py --input logs/emotion_variants_local/conversations_anger.jsonl
    → produces logs/analysis/WP_anger.jsonl
"""

import argparse
import json
from pathlib import Path
from datetime import timedelta
import pandas as pd
import numpy as np
from collections import Counter

import emotion_utils  # uses your shared helpers

# ------------------------------------------
# Configurable constants
# ------------------------------------------
NEGATIVE_WORDS = emotion_utils.NEGATIVE_WORDS
POSITIVE_WORDS = emotion_utils.POSITIVE_WORDS
VALENCE = emotion_utils.VALENCE


# ------------------------------------------
# Helper functions
# ------------------------------------------
def rolling_zscore(series, window=5):
    roll = series.rolling(window=window, min_periods=2)
    return (series - roll.mean()) / roll.std(ddof=0)


def detect_worry_dates(df, emotions=None, z_thresh=1.5):
    """Identify days with emotional spikes relative to local baseline."""
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
    return sorted(worry_dates)


def keyword_counts(texts, vocab):
    words = " ".join(texts).lower().split()
    return Counter(w for w in words if w in vocab)


def summarize_window(texts):
    neg = keyword_counts(texts, NEGATIVE_WORDS)
    pos = keyword_counts(texts, POSITIVE_WORDS)
    return {
        "negative_keywords": dict(neg.most_common(5)),
        "positive_keywords": dict(pos.most_common(5)),
    }


def compute_recovery(df, start, end, lookahead_days=5):
    """Compute average emotional valence before/after a worry window."""
    df = df.copy()
    df["valence"] = df["sentiment"].map(VALENCE).fillna(0.0)
    cluster_val = df[(df["timestamp"] >= start) & (df["timestamp"] < end)]["valence"].mean()
    after_start, after_end = end, end + timedelta(days=lookahead_days)
    after_val = df[(df["timestamp"] >= after_start) & (df["timestamp"] < after_end)]["valence"].mean()
    if np.isnan(cluster_val) or np.isnan(after_val):
        return None
    return round(after_val - cluster_val, 3)


def safe_json(obj):
    """Ensure all objects are JSON serializable."""
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


# ------------------------------------------
# Main logic
# ------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input .jsonl log file")
    parser.add_argument("--output_dir", default="logs/analysis", help="Output directory for WP_*.jsonl")
    parser.add_argument("--window_days", type=int, default=2)
    parser.add_argument("--lookahead_days", type=int, default=5)
    args = parser.parse_args()

    infile = Path(args.input)
    emotion_name = infile.stem.replace("conversations_", "")
    outfile = Path(args.output_dir) / f"WP_{emotion_name}.jsonl"

    print(f"📘 Loading {infile}…")
    df = emotion_utils.load_conversations(str(infile))

    # Emotion tagging if not present
    if "sentiment" not in df.columns:
        df = emotion_utils.tag_emotion(df)

    df = df.dropna(subset=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    print("🔍 Detecting worry clusters…")
    worry_dates = detect_worry_dates(df)
    print(f"🧩 Found {len(worry_dates)} potential worry points.")

    results = []
    for d in worry_dates:
        start = pd.Timestamp(d).tz_localize("UTC")
        end = start + timedelta(days=args.window_days)
        window = df[(df["timestamp"] >= start) & (df["timestamp"] < end)]
        if window.empty:
            continue
        summary = summarize_window(window["content"].tolist())
        recovery_slope = compute_recovery(df, start, end, lookahead_days=args.lookahead_days)

        results.append({
            "date": str(d),
            "num_messages": int(len(window)),
            "emotions": {k: int(v) for k, v in window["sentiment"].value_counts().items()},
            "summary": safe_json(summary),
            "example_texts": window["content"].head(3).tolist(),
            "recovery_slope": recovery_slope,
        })

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(outfile, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(safe_json(r), ensure_ascii=False) + "\n")

    print(f"✅ Saved {len(results)} worry windows → {outfile}")


if __name__ == "__main__":
    main()

