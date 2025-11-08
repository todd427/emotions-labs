#!/usr/bin/env python3
"""
plot_recovery_timeline.py
Visualize recovery slopes from logs/worry_points.jsonl
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def load_worry_points(path):
    records = [json.loads(line) for line in open(path, "r", encoding="utf-8") if line.strip()]
    df = pd.DataFrame(records)
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

    if "recovery_slope" not in df.columns:
        print("⚠️ No recovery_slope column found.")
        return

    if args.start:
        df = df[df["date"] >= pd.to_datetime(args.start)]
    if args.end:
        df = df[df["date"] <= pd.to_datetime(args.end)]

    if df.empty:
        print("⚠️ No data in selected range.")
        return

    plt.figure(figsize=(12, 6))
    plt.plot(df["date"], df["recovery_slope"], marker="o", linewidth=1.8)
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.title(f"Emotional Recovery Over Time ({args.start or df.date.min().date()} → {args.end or df.date.max().date()})", fontsize=14)
    plt.xlabel("Date")
    plt.ylabel("Recovery Slope (Δ valence)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

