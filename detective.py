#!/usr/bin/env python3
"""
detective.py
------------
Interactive Emotion–Activity Dashboard

Zoomable Plotly timeline showing message activity and emotional distribution.
Correlates daily message volume with emotions, filtered by date range.

Requirements:
    pip install plotly pandas transformers
"""

import json
import glob
import argparse
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from transformers import pipeline


# ---------------------- LOADERS ----------------------

def load_conversations(path_pattern: str) -> pd.DataFrame:
    records = []
    for file in glob.glob(path_pattern):
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    msg = json.loads(line.strip())
                    ts = msg.get("timestamp") or msg.get("created") or msg.get("create_time")
                    content = msg.get("content") or msg.get("text") or ""
                    role = msg.get("role", "user")
                    records.append({"timestamp": ts, "role": role, "content": content})
                except Exception:
                    continue
    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("No messages found — check your path or JSONL format.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp")
    return df


# ---------------------- EMOTION TAGGING ----------------------

def tag_emotion(df: pd.DataFrame) -> pd.DataFrame:
    print("🧠 Loading emotion model...")
    emo_model = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None,
        device=0
    )
    labels = []
    for text in df["content"]:
        if not isinstance(text, str) or not text.strip():
            labels.append("neutral")
            continue
        try:
            result = emo_model(text[:512])[0]
            label = max(result, key=lambda x: x["score"])
            labels.append(label["label"])
        except Exception:
            labels.append("neutral")
    df = df.copy()
    df["sentiment"] = labels
    return df


# ---------------------- ANALYSIS ----------------------

def analyze_emotions(df: pd.DataFrame, start=None, end=None) -> pd.DataFrame:
    if start or end:
        mask = pd.Series(True, index=df.index)
        if start:
            mask &= df["timestamp"] >= start
        if end:
            mask &= df["timestamp"] <= end
        df = df[mask]
    df["date"] = df["timestamp"].dt.date
    daily_counts = df.groupby("date").size().rename("total")
    emotion_counts = df.groupby(["date", "sentiment"]).size().unstack(fill_value=0)
    merged = pd.concat([emotion_counts, daily_counts], axis=1)
    return merged


# ---------------------- PLOTTING ----------------------

def plot_interactive_timeline(merged: pd.DataFrame, output="logs/detective_timeline.html"):
    emotions = [col for col in merged.columns if col != "total"]
    merged = merged.reset_index()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=merged["date"],
            y=merged["total"],
            mode="lines+markers",
            name="Total Messages",
            line=dict(color="black", width=2)
        ),
        secondary_y=False,
    )

    # Add emotion traces
    for emo in emotions:
        fig.add_trace(
            go.Scatter(
                x=merged["date"],
                y=merged[emo],
                mode="lines",
                name=emo.capitalize(),
                opacity=0.7
            ),
            secondary_y=True,
        )

    # Correlation heatmap annotation
    corr = merged[emotions + ["total"]].corr()["total"].sort_values(ascending=False).round(3)
    corr_text = "<br>".join([f"{e}: {c}" for e, c in corr.items() if e != "total"])

    fig.update_layout(
        title="🕵️‍♂️ Emotional Activity Timeline (Detective Dashboard)",
        xaxis_title="Date",
        yaxis_title="Total Messages",
        yaxis2_title="Emotion Count",
        legend_title="Emotion",
        template="plotly_white",
        hovermode="x unified",
        annotations=[dict(
            xref="paper", yref="paper", x=1.15, y=0.5,
            showarrow=False, align="left",
            text=f"<b>Correlation with Activity</b><br>{corr_text}"
        )]
    )

    Path("logs").mkdir(exist_ok=True)
    fig.write_html(output)
    print(f"✅ Interactive dashboard saved to {output}")
    return fig


# ---------------------- MAIN ----------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detective Dashboard for Emotion–Activity Correlation")
    parser.add_argument("path", nargs="?", default="logs/conversations_clean.jsonl", help="Path to JSONL logs")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    df = load_conversations(args.path)
    print(f"📘 Loaded {len(df)} messages from {df['timestamp'].min()} → {df['timestamp'].max()}")

    df = tag_emotion(df)
    merged = analyze_emotions(
        df,
        start=pd.to_datetime(args.start) if args.start else None,
        end=pd.to_datetime(args.end) if args.end else None
    )

    plot_interactive_timeline(merged)

