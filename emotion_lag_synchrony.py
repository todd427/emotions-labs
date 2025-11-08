#!/usr/bin/env python3
# emotion_lag_synchrony.py — cross-correlation lag analysis for emotional entrainment
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(path="logs/emotions_tagged.csv"):
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"Loaded {len(df)} messages ({df['role'].nunique()} roles)")
    return df


def prepare_series(df):
    """Return aligned numeric emotion matrices for user and assistant."""
    df["order"] = df.groupby("role").cumcount()
    frames = []
    for role in df["role"].unique():
        sub = df[df["role"] == role]
        enc = pd.get_dummies(sub["sentiment"])
        enc["order"] = sub["order"].values
        enc["role"] = role
        frames.append(enc)

    all_df = pd.concat(frames, ignore_index=True)
    user = all_df[all_df["role"].str.lower() == "user"].drop(columns=["role"], errors="ignore")
    ai = all_df[all_df["role"].str.lower() != "user"].drop(columns=["role"], errors="ignore")

    user = user.drop_duplicates(subset=["order"]).set_index("order")
    ai = ai.drop_duplicates(subset=["order"]).set_index("order")

    max_len = int(max(user.index.max(), ai.index.max()) + 1)
    idx = pd.Index(range(max_len))
    user = user.reindex(idx, fill_value=0)
    ai = ai.reindex(idx, fill_value=0)
    return user, ai


def safe_corr(a, b):
    """Return Pearson r or 0 if invalid."""
    if len(a) < 2 or len(b) < 2:
        return 0.0
    try:
        r = np.corrcoef(a, b)[0, 1]
        if np.isnan(r):
            return 0.0
        return float(r)
    except Exception:
        return 0.0


def cross_correlation(a, b, max_lag=20):
    """Compute normalized cross-correlation for lags within ±max_lag."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = (a - np.mean(a)) / (np.std(a) + 1e-9)
    b = (b - np.mean(b)) / (np.std(b) + 1e-9)

    lags = range(-max_lag, max_lag + 1)
    corrs = []
    n = len(a)
    for lag in lags:
        if lag < 0:
            seg_a, seg_b = a[-lag:], b[:n + lag]
        elif lag > 0:
            seg_a, seg_b = a[:n - lag], b[lag:]
        else:
            seg_a, seg_b = a, b
        corrs.append(safe_corr(seg_a, seg_b))
    return np.array(list(lags)), np.array(corrs)


def lag_analysis(user, ai, max_lag=20):
    emotions = [c for c in user.columns if c in ai.columns]
    lag_scores = pd.DataFrame(index=range(-max_lag, max_lag + 1), columns=emotions)
    peaks = {}

    for emo in emotions:
        lags, corrs = cross_correlation(user[emo].values, ai[emo].values, max_lag=max_lag)
        lag_scores[emo] = corrs
        best_lag = int(lags[np.argmax(corrs)])
        best_val = float(np.max(corrs))
        peaks[emo] = (best_lag, best_val)

    print("\nPeak synchrony per emotion:")
    for emo, (lag, val) in peaks.items():
        print(f"{emo:<10} lag={lag:+d} → r={val:.3f}")
    return lag_scores, peaks


def plot_heatmap(lag_scores):
    plt.figure(figsize=(10, 6))
    sns.heatmap(lag_scores.T.astype(float), cmap="coolwarm", center=0, annot=False)
    plt.title("Emotional Lag Synchrony (User vs. Assistant)")
    plt.xlabel("Lag (messages) — positive = Assistant follows User")
    plt.ylabel("Emotion")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = load_data("logs/emotions_tagged.csv")
    user, ai = prepare_series(df)
    lag_scores, peaks = lag_analysis(user, ai, max_lag=20)
    plot_heatmap(lag_scores)

