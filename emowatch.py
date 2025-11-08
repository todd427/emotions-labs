
import json
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
import argparse

# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

def is_technical_noise(text):
    # Heuristic to skip terminal output, code snippets, or unhelpful logs
    return (
        any(token in text for token in ['Traceback', '.py', '.cpp', '.sh', '.so', 'File "', 'import ', 'nvcc', '/home/', 'pts/', '$', '>>>', '#include']) or
        text.count('\n') > 3 or
        len(text.strip()) < 20
    )

def load_and_filter(jsonl_path):
    rows = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            msg = json.loads(line)
            if msg.get('role') != 'user':
                continue
            content = msg.get('content', '')
            if is_technical_noise(content):
                continue
            ts = msg.get('timestamp')
            try:
                ts = datetime.fromtimestamp(float(ts)) if isinstance(ts, (int, float)) else pd.to_datetime(ts)
            except Exception:
                continue
            scores = analyzer.polarity_scores(content)
            if scores['compound'] <= -0.4:
                rows.append({
                    'timestamp': ts,
                    'conversation': msg.get('conversation', 'unknown'),
                    'content': content,
                    'compound': scores['compound'],
                    'neg': scores['neg'],
                    'neu': scores['neu'],
                    'pos': scores['pos']
                })
    return pd.DataFrame(rows)

def detect_escalation(df, window_minutes=15, spike_threshold=3):
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    spikes = []

    for idx, row in df.iterrows():
        start_time = row['timestamp']
        end_time = start_time + timedelta(minutes=window_minutes)
        window_df = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]
        if len(window_df) >= spike_threshold:
            spikes.append({
                'start': start_time,
                'end': end_time,
                'count': len(window_df),
                'sample': window_df.head(1)['content'].values[0]
            })
    return pd.DataFrame(spikes).drop_duplicates()

def main():
    parser = argparse.ArgumentParser(description="Emotional health monitor for chat logs")
    parser.add_argument("jsonl_path", help="Path to cleaned conversations JSONL")
    parser.add_argument("--output", default="flagged_vader.csv", help="CSV file to write flagged messages")
    parser.add_argument("--spikes", default="spikes.csv", help="CSV file to write escalation spikes")
    args = parser.parse_args()

    print("🔍 Analyzing", args.jsonl_path)
    df = load_and_filter(args.jsonl_path)
    df.to_csv(args.output, index=False)
    print(f"✅ Flagged {len(df)} messages saved to {args.output}")

    spikes_df = detect_escalation(df)
    if not spikes_df.empty:
        spikes_df.to_csv(args.spikes, index=False)
        print(f"🚨 Detected {len(spikes_df)} emotional spikes → saved to {args.spikes}")
    else:
        print("✅ No escalation spikes detected.")

if __name__ == "__main__":
    main()
