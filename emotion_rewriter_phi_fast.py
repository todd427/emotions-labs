#!/usr/bin/env python3
"""
emotion_rewriter_phi_fast.py
----------------------------
True batched Phi-3.5-mini rewriting.
No CPU offload, minimal overhead. Writes to ./logs/phi_out_fast/.
"""

import sys, os, json, zipfile, torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

EMOTIONS = ["anger","disgust","fear","joy","neutral","sadness","surprise"]
MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"
BATCH_SIZE = 6
MAX_NEW_TOKENS = 40
OUTPUT_DIR = Path("./logs/phi_out_fast")
CHECKPOINT_EVERY = 200

print(f"⚡ Loading {MODEL_NAME}")
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(device)
model.eval()

def rewrite_batch(texts, emotion):
    prompts = [
        f"Rewrite this sentence as if spoken by someone feeling {emotion}. "
        f"Keep the meaning natural and concise.\nUser: {t}\nRewrite:"
        for t in texts
    ]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.7,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    results = []
    for prompt, text in zip(prompts, decoded):
        cut = text.split("Rewrite:")[-1].strip()
        results.append(cut)
    return results

def process_file(path):
    lines = [json.loads(l) for l in open(path, "r", encoding="utf-8") if l.strip()]
    msgs = [m for m in lines if m.get("role") == "user" and m.get("content")]
    print(f"Found {len(msgs)} user messages.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for emo in EMOTIONS:
        out_path = OUTPUT_DIR / f"phi_fast_{emo}.jsonl"
        print(f"\n🧠 Generating {emo.upper()} variant...")
        with open(out_path, "w", encoding="utf-8") as f:
            for i in tqdm(range(0, len(msgs), BATCH_SIZE), desc=f"{emo[:5]}"):
                batch = msgs[i:i + BATCH_SIZE]
                texts = [m["content"] for m in batch]
                rewrites = rewrite_batch(texts, emo)
                for m, rew in zip(batch, rewrites):
                    record = dict(m)
                    record["emotion"] = emo
                    record["rewritten"] = rew
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                if i % (CHECKPOINT_EVERY * BATCH_SIZE) == 0:
                    f.flush()
                    os.fsync(f.fileno())

        with zipfile.ZipFile(out_path.with_suffix(".zip"), "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(out_path, arcname=out_path.name)
        print(f"✅ Saved {out_path.with_suffix('.zip')}")

    print("\n🎉 Finished Phi-3.5-mini fast rewriting!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python emotion_rewriter_phi_fast.py ./logs/conversations_clean.jsonl")
        sys.exit(1)
    inp = Path(sys.argv[1])
    if not inp.exists():
        print(f"❌ File not found: {inp}")
        sys.exit(1)
    process_file(inp)

