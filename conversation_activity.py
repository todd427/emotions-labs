#!/usr/bin/env python3
# conversation_activity.py — robust timestamp auto-detect

import json, sys, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path


def load_conversation(path: str):
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(f"File not found: {file}")

    records = []
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                msg = json.loads(line)
                ts = msg.get("timestamp") or msg.get("time_iso") or msg.get("meta", {}).get("time_iso")
                role = msg.get("role", "user")
                records.append({"timestamp": ts, "role": role})
            except Exception:
                continue

    df = pd.DataFrame(records)
    if df.empty:
        sys.exit("⚠️ No messages found.")

    # Show first few raw timestamps
    sample = df["timestamp"].dropna().astype(str).head(5).tolist()
    print(f"Sample timestamps: {sample}")

    # Detect format
    try:
        # If looks numeric, decide scale
        if pd.to_numeric(df["timestamp"], errors="coerce").notna().sum() > 0:
            nums = pd.to_numeric(df["timestamp"], errors="coerce").dropna()
            median = nums.median()
            if median > 1e12:
                print("Detected millisecond timestamps.")
                df["timestamp"] = pd.to_datetime(nums, unit="ms", errors="coerce")
            elif median > 1e9:
                print("Detected second timestamps.")
                df["timestamp"] = pd.to_datetime(nums, unit="s", errors="coerce")
            else:
                print("Detected small numeric values — interpreting as seconds.")
                df["timestamp"] = pd.to_datetime(nums, unit="s", errors="coerce")
        else:
            # Treat as ISO strings
            print("Detected ISO-formatted timestamps.")
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    except Exception as e:
        print("Timestamp parsing failed:", e)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    return df


def summarize(df):
    start, end = df["timestamp"].min(), df["timestamp"].max()
    span = end - start
    roles = df["role"].value_counts().to_dict()
    print("\n📊 Conversation Summary")
    print("-------------------------")
    print(f"Start date: {start}")
    print(f"End date:   {end}")
    print(f"Duration:   {span.days} days, {span.seconds//3600} hours")
    print(f"Total messages: {len(df):,}")
    print(f"Roles: {roles}\n")


def plot_activity(df):
    df = df.set_index("timestamp")
    span_days = (df.index.max() - df.index.min()).days
    freq = "D" if span_days > 7 else "h"
    activity = df.resample(freq).size()
    plt.figure(figsize=(12, 6))
    plt.plot(activity.index, activity.values, color="royalblue", label="Messages")
    plt.title("Message Activity Over Time")
    plt.xlabel("Time")
    plt.ylabel(f"Messages per {'day' if freq=='D' else 'hour'}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python conversation_activity.py <path-to-jsonl>")
        sys.exit(1)

    path = sys.argv[1]
    df = load_conversation(path)
    summarize(df)
    plot_activity(df)

