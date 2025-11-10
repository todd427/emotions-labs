import argparse
from datetime import datetime, timedelta
import pandas as pd
from emotion_utils import (
    load_conversations,
    _emotion_model,
    plot_emotion_timeline,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to JSONL chat log")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--window", type=int, default=3, help="Rolling window size (days)")
    parser.add_argument("--threshold", type=float, default=0.8, help="Emotion score threshold for alerts")
    args = parser.parse_args()

    print(f"📘 Loading {args.input} …")
    df = load_conversations(args.input, prefer_tz="Europe/Dublin")

    print("🧠 Loading emotion model (j-hartmann/emotion-english-distilroberta-base)…")
    pipe = _emotion_model

    print(f"⏳ Filtering from {args.start} (Europe/Dublin)")
    start_ts = pd.to_datetime(args.start)
    end_ts = pd.to_datetime(args.end)
    print(f"⏳ Filtering until {args.end} (Europe/Dublin)")
    df = df[(df["local_date"] >= start_ts) & (df["local_date"] <= end_ts)]

    if df.empty:
        print("⚠️ No conversations in selected range.")
        return

    print(f"📈 Analyzing {len(df)} entries…")
    emotion_scores = []

    for day in pd.date_range(start_ts, end_ts):
        window = df[df["local_date"].between(day, day + timedelta(days=args.window))]
        if window.empty:
            continue

        texts = window["text"].tolist()
        results = pipe(texts, truncation=True)
        flat_results = [x[0] for x in results]
        scores = pd.DataFrame(flat_results)
        top_emotion = scores["label"].mode().iloc[0]
        top_score = scores["score"].mean()
        emotion_scores.append({
            "date": day.strftime("%Y-%m-%d"),
            "emotion": top_emotion,
            "score": top_score
        })

    result_df = pd.DataFrame(emotion_scores)
    print(result_df.to_string(index=False))

    alerts = result_df[result_df["score"] >= args.threshold]
    if not alerts.empty:
        print("\n🚨 Escalation Triggers Detected:")
        print(alerts.to_string(index=False))
    else:
        print("\n✅ No escalation triggers found.")

    plot_emotion_timeline(df)

if __name__ == "__main__":
    main()

