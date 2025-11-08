#!/usr/bin/env python3
"""
detective.py
------------
Detective Dashboard — correlation between emotion intensity and message activity.
Refactored to use emotion_utils.py for shared logic.
"""

import argparse
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from emotion_utils import load_conversations, tag_emotion, analyze_emotions


def plot_detective_dashboard(merged: pd.DataFrame, output="logs/detective_timeline.html"):
    """Interactive Plotly dashboard with correlation summary."""
    emotions = [col for col in merged.columns if col != "total"]
    merged = merged.reset_index()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=merged["date"], y=merged["total"], mode="lines+markers",
                   name="Total Messages", line=dict(color="black", width=2)),
        secondary_y=False,
    )

    for emo in emotions:
        fig.add_trace(
            go.Scatter(x=merged["date"], y=merged[emo],
                       mode="lines", name=emo.capitalize(), opacity=0.7),
            secondary_y=True,
        )

    corr = merged[emotions + ["total"]].corr()["total"].sort_values(ascending=False).round(3)
    corr_text = "<br>".join([f"{e}: {c}" for e, c in corr.items() if e != "total"])

    fig.update_layout(
        title="🕵️ Detective Dashboard — Emotional Activity Correlation",
        xaxis_title="Date",
        yaxis_title="Total Messages",
        yaxis2_title="Emotion Count",
        template="plotly_white",
        hovermode="x unified",
        annotations=[dict(xref="paper", yref="paper", x=1.15, y=0.5,
                          showarrow=False, align="left",
                          text=f"<b>Correlation with Activity</b><br>{corr_text}")]
    )

    fig.write_html(output)
    print(f"✅ Detective dashboard saved to {output}")
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detective emotion–activity correlation (refactored)")
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
    plot_detective_dashboard(merged)

