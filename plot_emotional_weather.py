#!/usr/bin/env python3
"""
plot_emotional_weather.py
-------------------------
Combined visualization of emotional composition (stacked bars)
and recovery slope (smoothed line) over time,
with shaded zones indicating emotional recovery vs. strain.

Usage examples:
    python plot_emotional_weather.py
    python plot_emotional_weather.py --start 2025-06-01 --end 2025-10-01
    python plot_emotional_weather.py --save logs/weather_report.png
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Optional smoothing (SciPy preferred)
try:
    from scipy.signal import savgol_filter
    def smooth_series(y, window=5, poly=2):
        if len(y) < window:
            return y
        return savgol_filter(y, window, poly)
except ImportError:
    def smooth_series(y, window=5, poly=None):
        return pd.Series(y).rolling(window=window, min_periods=1, center=True).mean().to_numpy()


def load_worry_points(path):
    records = [json.loads(line) for line in open(path, "r", encoding="utf-8") if line.strip()]
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])

    emolist = []
    for r in records:
        for emo, count in r.get("emotions", {}).items():
            emolist.append({"date": pd.to_datetime(r["date"]), "emotion": emo, "count": count})
    emo_df = pd.DataFrame(emolist)
    return df, emo_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="logs/worry_points.jsonl")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--save")
    args = parser.parse_args()

    df, emo_df = load_worry_points(Path(args.path))

    # Filter date range
    if args.start:
        start = pd.to_datetime(args.start)
        df = df[df["date"] >= start]
        emo_df = emo_df[emo_df["date"] >= start]
    if args.end:
        end = pd.to_datetime(args.end)
        df = df[df["date"] <= end]
        emo_df = emo_df[emo_df["date"] <= end]

    if df.empty or emo_df.empty:
        print("⚠️ No data in selected range.")
        return

    pivot = emo_df.pivot_table(index="date", columns="emotion", values="count", fill_value=0).sort_index()
    recovery = df[["date", "recovery_slope"]].dropna().sort_values("date")

    if len(recovery) > 3:
        recovery["smooth"] = smooth_series(recovery["recovery_slope"].to_numpy(), window=5, poly=2)
    else:
        recovery["smooth"] = recovery["recovery_slope"]

    # Align recovery to x positions
    x_pos = range(len(pivot.index))
    recovery = recovery.set_index("date").reindex(pivot.index, method="nearest").reset_index()

    # --- Plot setup ---
    fig, ax1 = plt.subplots(figsize=(14, 7))
    pivot.plot(kind="bar", stacked=True, ax=ax1, width=0.8, alpha=0.7)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Message Count")
    ax1.set_title("Emotional Weather Report: Composition + Recovery Trend", fontsize=15)
    ax1.legend(title="Emotion", bbox_to_anchor=(1.05, 1), loc="upper left")

    # Secondary axis for recovery line
    ax2 = ax1.twinx()
    ax2.plot(x_pos, recovery["smooth"], color="black", linewidth=2.5, marker="o", label="Smoothed Recovery")
    ax2.scatter(x_pos, recovery["recovery_slope"], color="gray", alpha=0.7, s=25, label="Raw Recovery")
    ax2.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax2.set_ylabel("Recovery Slope (Δ valence)", color="black")

    # --- Shade recovery vs strain zones ---
    for i in range(len(recovery) - 1):
        x0, x1 = x_pos[i], x_pos[i + 1]
        slope = recovery.loc[i, "smooth"]
        if slope > 0:
            ax2.axvspan(x0, x1, color="green", alpha=0.08)
        elif slope < 0:
            ax2.axvspan(x0, x1, color="red", alpha=0.08)

    # Add legend for these zones
    ax2.fill_between([], [], [], color="green", alpha=0.1, label="Recovery Zone (↑ improving)")
    ax2.fill_between([], [], [], color="red", alpha=0.1, label="Strain Zone (↓ declining)")
    ax2.legend(loc="upper right")

    # Format ticks
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

