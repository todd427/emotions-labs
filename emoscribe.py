#!/usr/bin/env python3
"""
emoscribe.py — narrative summarizer (deduplicated, linked, and cleaner)
Now:
  • Removes 'neutral' clusters
  • Uses 'I sensed …'
  • Generates real /emo/{id} URLs that work with emo_view.py
  • Deduplicates by emotion+recommendation
"""

import argparse, json, os, statistics, pandas as pd
from emotion_utils import VALENCE, friendly_tag


def load_jsonl(path):
    """Read JSONL safely."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except Exception:
                continue
    return data


def mood_badge(val):
    if val > 0.2:
        return "🟢 Calm"
    if val > -0.2:
        return "🟡 Uneasy"
    return "🔴 Stressed"


def derive_output_path(input_path):
    dirname, filename = os.path.split(input_path)
    name, _ = os.path.splitext(filename)
    if name.startswith("rec"):
        name = "emo" + name[3:]
    else:
        name = f"emo{name}_{friendly_tag()}"
    out_dir = os.path.join("logs", "analysis")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{name}.md")


def main():
    parser = argparse.ArgumentParser(description="Summarize emotional clusters into Markdown.")
    parser.add_argument("--input", required=True, help="Input rec*.jsonl from emowarn.py.")
    parser.add_argument("--output", help="Optional explicit output path.")
    parser.add_argument("--baseurl", default="http://localhost:8010/emo/",
                        help="Base URL for message links.")
    args = parser.parse_args()

    data = load_jsonl(args.input)
    if not data:
        print(f"⚠️ No data in {args.input}")
        return

    df = pd.DataFrame(data)

    # Remove neutral rows
    df = df[df["emotion"] != "neutral"]

    mean_val = statistics.mean([VALENCE.get(e, 0) for e in df["emotion"]])
    badge = mood_badge(mean_val)

    lines = [
        "# Emotion-Labs Summary\n",
        f"**Source:** `{os.path.basename(args.input)}`  ",
        f"**Current Mood:** {badge}\n",
    ]

    # Deduplicate by (emotion, recommendation)
    seen = set()
    unique_records = []
    for _, row in df.iterrows():
        key = (row["emotion"], row["recommendation"])
        if key not in seen:
            seen.add(key)
            unique_records.append(row)

    # Build global message counter for consistent numbering
    msg_counter = 1
    grouped = {}
    for row in unique_records:
        grouped.setdefault(row["emotion"], []).append(row)

    for emo, group in grouped.items():
        lines.append(f"\n## {emo.title()}\n")
        for row in group:
            examples = row.get("examples", [])
            msg_ids = []
            for ex in examples:
                msg_ids.append(f"[#{msg_counter}]({args.baseurl}{msg_counter})")
                msg_counter += 1
            msg_list = ", ".join(msg_ids) if msg_ids else "(context unavailable)"
            lines.append(
                f"- In messages {msg_list}, I sensed **{emo}**.  \n"
                f"  {row['recommendation']}"
            )

    out_path = args.output or derive_output_path(args.input)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"✅ Markdown summary written → {out_path}")


if __name__ == "__main__":
    main()

