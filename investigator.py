#!/usr/bin/env python3
"""
investigator.py
---------------
Emotional Trend Investigator — timeline visualization of message activity and
emotional variance over time. Refactored to use emotion_utils.py for shared logic.
"""

import argparse
import pandas as pd
import plotly.express as px
from emotion_utils import load_conversations, tag_emotion, analyze_emotions


def plot_investigation(merged: pd.DataFrame, output="logs/investigator_timeline.html"):
    """Plot emotion counts and total messages over time."""
    merged = merged.reset_index()
    fig = px.line(
        merged,
        x="date",
        y=[c for c in merged.columns if c != "total"],
        title="📈 Investigator — Emotional Frequency Over Time",
        labels={"value": "Message Count", "variable": "Emotion"},
        template="plotly_white"
    )

    fig.add_scatter(x=merged["date"], y=merged["total"],
                    mode="lines+markers", name="Total Messages", line=dict(color="black", width=2))
    fig.write_html(output)
    print(f"✅ Investigator plot saved to {output}")
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emotion trend investigator (refactored)")
    parser.add_argument("path", nargs="?", default="logs/conversations_clean.jsonl")
    parser.add_argument("--start", type=str)
    parser.add_argument("--end", type=str)
    args = parser.parse_args()

    df = load_conversations(args.path)
    print(f"📘 Loaded {len(df)} messages from {df['timestamp'].min()} to {df['timestamp'].max()}")

    start = pd.to_datetime(args.start, utc=True) if args.start else None
    end = pd.to_datetime(args.end, utc=True) if args.end else None

    df = tag_emotion(df)
    merged = analyze_emotions(df, start=start, end=end)
    plot_investigation(merged)

