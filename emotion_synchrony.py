#!/usr/bin/env python3
# emotion_synchrony.py — robust conversation-order synchrony (deduplicated)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_data(path="logs/emotions_tagged.csv"):
    df = pd.read_csv(path)
    if "sentiment" not in df or "role" not in df:
        raise ValueError("CSV must contain 'sentiment' and 'role' columns.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"Loaded {len(df)} messages ({df['role'].nunique()} roles)")
    if df["role"].nunique() < 2:
        print("⚠️ Only one role found — synchrony correlation may not be meaningful.")
    return df


def compute_emotion_matrix(df):
    """Convert emotions to sequential numeric series per role."""
    df["order"] = df.groupby("role").cumcount()
    roles = df["role"].unique()
    pivot_frames = []
    for role in roles:
        sub = df[df["role"] == role]
        encoded = pd.get_dummies(sub["sentiment"])
        encoded["order"] = sub["order"].values
        encoded["role"] = role
        pivot_frames.append(encoded)
    return pd.concat(pivot_frames, ignore_index=True)


def compute_synchrony(df):
    """Correlate emotion intensity across roles along message order."""
    user = df[df["role"].str.lower() == "user"].drop(columns=["role"], errors="ignore")
    ai = df[df["role"].str.lower() != "user"].drop(columns=["role"], errors="ignore")

    if user.empty or ai.empty:
        print("⚠️ Need both roles for synchrony.")
        return df, df, {}

    # Remove duplicate order indices before aligning
    user = user.drop_duplicates(subset=["order"]).set_index("order")
    ai = ai.drop_duplicates(subset=["order"]).set_index("order")

    # Align by message order (sequential)
    max_len = max(user.index.max(), ai.index.max()) + 1
    full_index = pd.Index(range(max_len))
    user = user.reindex(full_index, fill_value=0)
    ai = ai.reindex(full_index, fill_value=0)

    shared_cols = [c for c in user.columns if c in ai.columns and c not in ["timestamp", "order"]]
    correlations = {}
    for col in shared_cols:
        if user[col].std() == 0 or ai[col].std() == 0:
            correlations[col] = 0
        else:
            correlations[col] = user[col].corr(ai[col])

    mean_sync = pd.Series(correlations).mean()
    print("\nEmotion synchrony (Pearson correlation by category):")
    print(pd.Series(correlations).round(3))
    print(f"\nOverall synchrony score: {mean_sync:.3f}")
    return user, ai, correlations


def plot_synchrony(user, ai):
    """Plot user vs. AI emotion intensity curves over conversation order."""
    shared_cols = [c for c in user.columns if c in ai.columns]
    fig, axes = plt.subplots(len(shared_cols), 1, figsize=(12, 3 * len(shared_cols)), sharex=True)
    if len(shared_cols) == 1:
        axes = [axes]
    for i, col in enumerate(shared_cols):
        axes[i].plot(user.index, user[col], color="orange", label="User", alpha=0.7)
        axes[i].plot(ai.index, ai[col], color="blue", label="Assistant", alpha=0.5)
        axes[i].set_ylabel(col)
        axes[i].legend(loc="upper right")
    plt.suptitle("Emotional Synchrony: User vs. Assistant (Sequential)", fontsize=14)
    plt.xlabel("Message Order (synthetic time)")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


if __name__ == "__main__":
    csv_path = Path("logs/emotions_tagged.csv")
    df = load_data(csv_path)
    matrix = compute_emotion_matrix(df)
    user, ai, corrs = compute_synchrony(matrix)
    if corrs:
        plot_synchrony(user, ai)
