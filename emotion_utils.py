#!/usr/bin/env python3
"""
emotion_utils.py — restored and reconciled

Includes:
- Emotion model loader
- Valence and support word dictionaries
- Emotion scoring helpers
- Log file loader with time zone handling
- Timeline plotter
"""

import json
import pandas as pd
import pytz
from pathlib import Path
from transformers import pipeline
import matplotlib.pyplot as plt
from datetime import datetime
from collections import Counter

# Load model (global singleton for speed)
_emotion_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)

def load_emotion_model():
    return _emotion_model

# Emotion valence dictionary
VALENCE = {
    "joy": 1.0,
    "surprise": 0.5,
    "neutral": 0.0,
    "sadness": -1.0,
    "fear": -0.8,
    "anger": -0.7,
    "disgust": -0.6,
}

# Support word lists
NEGATIVE_WORDS = {
    "can’t", "cannot", "hopeless", "tired", "fail", "worthless", "useless",
    "anxious", "panic", "alone", "stuck", "hate", "scared", "exhausted",
    "depressed", "pointless", "overwhelmed"
}

POSITIVE_WORDS = {
    "sleep", "rest", "walk", "talk", "friend", "music", "cook", "read",
    "exercise", "meditate", "breathe", "outside", "relax", "sun", "coffee"
}

def get_dominant_emotion(predictions):
    """Return the emotion label with the highest score from HuggingFace prediction output."""
    if not predictions or not isinstance(predictions, list):
        return "neutral"
    if isinstance(predictions[0], dict):
        return predictions[0]["label"]
    elif isinstance(predictions[0], list):
        sorted_preds = sorted(predictions[0], key=lambda x: x["score"], reverse=True)
        return sorted_preds[0]["label"]
    return "neutral"

def calculate_valence(emotion):
    """Map emotion to valence value."""
    return VALENCE.get(emotion, 0.0)

def detect_support_words(text):
    """Count the number of known support/recovery words."""
    tokens = set(text.lower().split())
    pos = len(tokens & POSITIVE_WORDS)
    neg = len(tokens & NEGATIVE_WORDS)
    return pos, neg

def load_conversations(path: str, prefer_tz: str = "UTC") -> pd.DataFrame:
    """Load and normalize conversation logs from a JSONL file."""
    rows = []
    tz = pytz.timezone(prefer_tz)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if data.get("role") != "user":
                continue
            ts = data.get("create_time") or data.get("timestamp")
            if not ts:
                continue
            try:
                timestamp = pd.to_datetime(ts, unit="s", utc=True).tz_convert(tz)
            except Exception:
                continue
            rows.append({
                "text": data.get("content", ""),
                "timestamp": timestamp,
                "local_date": timestamp.normalize(),
            })
    return pd.DataFrame(rows)

def plot_emotion_timeline(df: pd.DataFrame, column: str = "emotion", title: str = "Emotion Timeline"):
    """Plot a stacked bar chart of emotions over time."""
    plt.figure(figsize=(12, 6))
    df.groupby("local_date")[column].value_counts().unstack().fillna(0).plot(kind="bar", stacked=True, ax=plt.gca())
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
