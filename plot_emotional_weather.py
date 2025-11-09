#!/usr/bin/env python3
"""
plot_emotional_weather.py — multi-input version
------------------------------------------------
Accepts either:
  • worry_points.jsonl (precomputed analysis)
  • any conversations_*.jsonl file (raw emotion logs)

Usage examples:
    python plot_emotional_weather.py
    python plot_emotional_weather.py --input logs/emotion_variants_local/conversations_anger.jsonl
    python plot_emotional_weather.py --input logs/emotion_variants_local/conversations_sadness.jsonl --start 2025-06-01 --end 2025-09-30
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Optional smoothing
try:
    from scipy.signal import savgol_filter
    def smooth_series(y, window=5, poly=2):
        if len(y) < window:
            return y
        return savgol_filter(y, window, poly)
except ImportError:
    def smooth_series(y, window=5, poly=None):
        return pd.Series(y).rolling(window=window, min_periods=1, center=True).mean().to_numpy()


# ============================================================
# HELPERS
# ============================================================

def load_jsonl(path):
    """Safely load any JSONL file into a DataFrame."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not records:
        raise ValueError(f"⚠️ No valid JSON records in {path}")
    return pd.DataFrame(records)


def is_worry_points(df):
    """Detect whether this is a precomputed worry_points dataset."""
    return "recovery_slope" in df.columns and "emotions" in df.columns


def compute_from_raw(df):
    """Derive per-day emotion counts and pseudo-recovery from raw logs."""
    # Expect at least timestamp + sentiment columns
    if "timestamp" not in df.columns:
        raise ValueError("Expected 'timestamp' column in raw dataset.")
    if "sentiment" not in df.columns:
        raise ValueError("Expected 'sentiment' column in raw dataset.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"])
    df["date"] = df["timestamp"].dt.date
    counts = df.groupby(["date", "sentiment"]).size().unstack(fill_value=0)

    # Compute crude recovery slope: diff of (positive - negative ratio)
    valence_map = {"joy": 1, "surprise": 0.5, "neutral": 0, "sadness": -1, "fear": -0.8, "anger": -0.7, "disgust": -0.6}
    df["valence"] = df["sentiment"].map(valence_map).fillna(0)
    mean_valence = df.groupby("date")["valence"].mean().diff().fillna(0)

    out = pd.DataFrame({
        "date": pd.to_datetime(counts.index),
        "recovery_slope": mean_valence.values
    })
    out["emotions"] = counts.to_dict(orient="records")
    return out, counts


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="logs/worry_points.jsonl", help="Path to input JSONL file")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument("--save", help="Optional path to save PNG")
    args = parser.parse_args()

    df = load_jsonl(args.input)
    print(f"📘 Loaded {len(df)} records from {args.input}")

    # Distinguish between precomputed worry_points and raw logs
    if is_worry_points(df):
        print("🧩 Detected precomputed worry_points dataset.")
        df["date"] = pd.to_datetime(df["date"])
        emo_records = []
        for _, row in df.iterrows():
            for emo, count in row.get("emotions", {}).items():
                emo_records.append({"date": row["date"], "emotion": emo, "count": count})
        emo_df = pd.DataFrame(emo_records)
    else:
        print("🧠 Detected raw emotion log — computing aggregates.")
        df, emo_df = compute_from_raw(df)

    # Filter by date
    if args.start:
        start = pd.to_datetime(args.start)
        df = df[df["date"] >= start]
        emo_df = emo_df[emo_df["date"] >= start]
    if args.end:
        end = pd.to_datetime(args.end)
        df = df[df["date"] <= end]
        emo_df = emo_df[emo_df["date"] <= end]

    # Prepare pivot
    pivot = emo_df.pivot_table(index="date", columns="emotion", values="count", fill_value=0).sort_index()
    recovery = df[["date", "recovery_slope"]].dropna().sort_values("date")

    # Smooth recovery
    if len(recovery) > 3:
        recovery["smooth"] = smooth_series(recovery["recovery_slope"].to_numpy(), window=5, poly=2)
    else:
        recovery["smooth"] = recovery["recovery_slope"]

    # Align recovery to pivot
    x_pos = range(len(pivot.index))
    recovery = recovery.set_index("date").reindex(pivot.index, method="nearest").reset_index()

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(14, 7))
    pivot.plot(kind="bar", stacked=True, ax=ax1, width=0.8, alpha=0.7)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Message Count")
    ax1.set_title(f"Emotional Weather Report ({Path(args.input).stem})", fontsize=15)
    ax1.legend(title="Emotion", bbox_to_anchor=(1.05, 1), loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(x_pos, recovery["smooth"], color="black", linewidth=2.5, marker="o", label="Smoothed Recovery")
    ax2.scatter(x_pos, recovery["recovery_slope"], color="gray", alpha=0.7, s=25, label="Raw Recovery")
    ax2.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax2.set_ylabel("Recovery Slope (Δ valence)", color="black")

    # Shade recovery vs strain zones
    for i in range(len(recovery) - 1):
        x0, x1 = x_pos[i], x_pos[i + 1]
        slope = recovery.loc[i, "smooth"]
        color = "green" if slope > 0 else "red"
        ax2.axvspan(x0, x1, color=color, alpha=0.08)

    # Annotate peaks/troughs
    if not recovery.empty:
        max_idx = recovery["smooth"].idxmax()
        min_idx = recovery["smooth"].idxmin()
        max_row = recovery.loc[max_idx]
        min_row = recovery.loc[min_idx]
        ax2.annotate(f"↑ Peak Recovery\n({max_row['smooth']:.3f})", xy=(max_idx, max_row["smooth"]),
                     xytext=(max_idx, max_row["smooth"] + 0.02),
                     arrowprops=dict(arrowstyle="->", color="green", lw=1.5),
                     color="green", fontsize=10, ha="center")
        ax2.annotate(f"↓ Deepest Strain\n({min_row['smooth']:.3f})", xy=(min_idx, min_row["smooth"]),
                     xytext=(min_idx, min_row["smooth"] - 0.03),
                     arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
                     color="red", fontsize=10, ha="center")

    # Final touches
    ax2.legend(loc="upper right")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([d.strftime("%Y-%m-%d") for d in pivot.index], rotation=45, ha="right")
    plt.tight_layout()

    if args.save:
        Path(args.save).parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(args.save, dpi=300)
        print(f"💾 Saved plot → {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

