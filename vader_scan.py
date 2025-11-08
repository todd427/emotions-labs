# vader_scan.py

import json
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime

analyzer = SentimentIntensityAnalyzer()
neg_threshold = -0.4  # Customize as needed

input_path = "./logs/conversations_clean.jsonl"
output_path = "./logs/flagged_vader.csv"

flagged = []

with open(input_path, 'r') as f:
    for line in f:
        msg = json.loads(line)
        if msg['role'] != 'user':
            continue
        score = analyzer.polarity_scores(msg['content'])
        if score['compound'] <= neg_threshold:
            flagged.append({
                "timestamp": datetime.fromtimestamp(msg['timestamp']),
                "conversation": msg.get('conversation', 'unknown'),
                "content": msg['content'],
                "compound": score['compound'],
                "neg": score['neg'],
                "neu": score['neu'],
                "pos": score['pos']
            })

df = pd.DataFrame(flagged)
df.sort_values("timestamp").to_csv(output_path, index=False)
print(f"✅ Saved {len(df)} flagged messages to {output_path}")

