#!/usr/bin/env python3
"""
emotion_rewriter.py — safer, resumable version for flattened conversation logs.
Each JSONL line looks like:
{"timestamp": ..., "conversation": ..., "role": "user", "content": "..."}
"""

import os, sys, json, time, zipfile
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm

EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
MODEL = "gpt-4o"
OUTPUT_DIR = Path("./logs/emotion_variants")
MAX_RETRIES = 3
TEMPERATURE = 0.9
DELAY_BETWEEN_CALLS = 0.2  # seconds

def get_client():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise EnvironmentError("Missing OPENAI_API_KEY. Set it first.")
    return OpenAI(api_key=key)

client = get_client()

def rewrite_text(text: str, emotion: str) -> str:
    """Ask OpenAI to rewrite text into a strong emotional tone, with backoff for rate limits."""
    if not isinstance(text, str) or not text.strip():
        return text
    prompt = (
        f"Rewrite this message to sound as if written by someone feeling {emotion}. "
        f"Keep the meaning and phrasing natural; exaggerate only the emotional tone.\n\n"
        f"{text}"
    )

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You rewrite text in vivid human emotion."},
                    {"role": "user", "content": prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=200,
            )
            rewritten = response.choices[0].message.content.strip()
            return rewritten or text

        except Exception as e:
            msg = str(e)
            # Detect rate limit (429)
            if "rate_limit_exceeded" in msg or "429" in msg:
                wait = 60 * 10  # wait 10 minutes
                print(f"🚦 Rate limit reached for {emotion}. Waiting {wait/60} minutes...")
                time.sleep(wait)
            else:
                wait = 2 ** attempt
                print(f"⚠️ ({emotion}) API error {e} — retrying in {wait}s")
                time.sleep(wait)
    return text

def process_file(path: Path):
    """Read flattened JSONL file and rewrite all user messages."""
    lines = [json.loads(l) for l in open(path, "r", encoding="utf-8") if l.strip()]
    user_msgs = [l for l in lines if l.get("role") == "user" and l.get("content")]
    print(f"Found {len(user_msgs)} user messages to rewrite.")
    if not user_msgs:
        print("❌ No usable messages found. Check your file structure.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for emotion in EMOTIONS:
        out_path = OUTPUT_DIR / f"conversations_{emotion}.jsonl"
        zip_path = out_path.with_suffix(".zip")

        # Skip emotion if already done
        if zip_path.exists():
            print(f"⏩ {emotion.upper()} already exists, skipping.")
            continue

        print(f"\n🧠 Generating {emotion.upper()} variant...")
        with open(out_path, "w", encoding="utf-8") as out_f:
            for msg in tqdm(user_msgs, desc=f"Rewriting for {emotion}"):
                new_msg = dict(msg)
                new_msg["emotion"] = emotion
                new_msg["rewritten"] = rewrite_text(msg["content"], emotion)
                out_f.write(json.dumps(new_msg, ensure_ascii=False) + "\n")
                out_f.flush()
                time.sleep(DELAY_BETWEEN_CALLS)

        # Zip the result
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(out_path, arcname=out_path.name)

        size_kb = os.path.getsize(out_path) / 1024
        print(f"✅ Saved {zip_path} ({size_kb:.1f} KB)")

    print("\n🎉 All emotional variants generated successfully!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python emotion_rewriter.py ./logs/conversations_clean.jsonl")
        sys.exit(1)
    process_file(Path(sys.argv[1]))

