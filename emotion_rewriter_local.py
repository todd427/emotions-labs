#!/usr/bin/env python3
"""
emotion_rewriter_local_batched.py
---------------------------------
Batched + resumable local rewriting using a quantized Llama model.
Optimized for 8 GB GPUs.
"""

import sys, os, json, time, zipfile
from pathlib import Path
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)

# ------------------ CONFIG ------------------
EMOTIONS = ["anger","disgust","fear","joy","neutral","sadness","surprise"]
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
BATCH_SIZE = 2          # adjust for VRAM
MAX_NEW_TOKENS = 60     # shorter generations = faster
OUTPUT_DIR = Path("./logs/emotion_variants_local")
CHECKPOINT_EVERY = 200  # save partial progress
# --------------------------------------------

print(f"🚀 Loading local model: {MODEL_NAME}")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    llm_int8_enable_fp32_cpu_offload=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    quantization_config=bnb_config,
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
)

def rewrite_batch(texts, emotion):
    """Run the model on a batch of user messages."""
    prompts = [
        f"Rewrite this message as if spoken by someone feeling {emotion}. "
        f"Keep meaning natural:\n\nUser: {t}\n\nRewrite:"
        for t in texts
    ]
    try:
        outs = pipe(
            prompts,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=0.7,
            pad_token_id=pipe.tokenizer.eos_token_id,
        )
        results = []
        for o in outs:
            text = o[0]["generated_text"] if isinstance(o,list) else o["generated_text"]
            results.append(text.split("Rewrite:")[-1].strip())
        return results
    except Exception as e:
        print(f"⚠️ Batch error ({emotion}): {e}")
        return texts

def process_file(path):
    lines = [json.loads(l) for l in open(path,"r",encoding="utf-8") if l.strip()]
    msgs = [m for m in lines if m.get("role")=="user" and m.get("content")]
    print(f"Found {len(msgs)} user messages to rewrite.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for emo in EMOTIONS:
        out_path = OUTPUT_DIR / f"conversations_{emo}.jsonl"
        done_ids = set()
        if out_path.exists():
            # resume mode
            with open(out_path,"r",encoding="utf-8") as f:
                for line in f:
                    try: done_ids.add(json.loads(line).get("id"))
                    except: continue
            print(f"↩️ Resuming {emo} — already {len(done_ids)} processed.")

        print(f"\n🧠 Generating {emo.upper()} variant...")
        with open(out_path,"a",encoding="utf-8") as f:
            for i in tqdm(range(0,len(msgs),BATCH_SIZE), desc=f"{emo[:5]}"):
                batch = msgs[i:i+BATCH_SIZE]
                batch_ids = [hash(m["content"]) for m in batch]
                if any(bid in done_ids for bid in batch_ids):
                    continue
                texts = [m["content"] for m in batch]
                rewrites = rewrite_batch(texts,emo)
                for m,rew in zip(batch,rewrites):
                    record = dict(m)
                    record["emotion"]=emo
                    record["rewritten"]=rew
                    record["id"]=hash(m["content"])
                    f.write(json.dumps(record,ensure_ascii=False)+"\n")
                if i % (CHECKPOINT_EVERY*BATCH_SIZE) == 0:
                    f.flush(); os.fsync(f.fileno())

        with zipfile.ZipFile(out_path.with_suffix(".zip"),"w",zipfile.ZIP_DEFLATED) as zf:
            zf.write(out_path, arcname=out_path.name)
        print(f"✅ Saved {out_path.with_suffix('.zip')}")

    print("\n🎉 All emotional variants generated locally!")

if __name__=="__main__":
    if len(sys.argv)<2:
        print("Usage: python emotion_rewriter_local_batched.py ./logs/conversations_clean.jsonl")
        sys.exit(1)
    inp=Path(sys.argv[1])
    if not inp.exists():
        print(f"❌ File not found: {inp}")
        sys.exit(1)
    process_file(inp)

