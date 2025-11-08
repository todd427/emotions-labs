#!/usr/bin/env python3
import json
import zipfile
from pathlib import Path
from datetime import datetime


def extract_from_zip(zip_path: str, output_dir: str = "./logs"):
    """
    Flatten any ChatGPT export ZIP into logs/conversations_clean.jsonl
    Supports 2025 export schema (mapping-based format).
    """
    zip_path = Path(zip_path).expanduser().resolve()
    if not zip_path.exists():
        raise FileNotFoundError(f"{zip_path} not found")

    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "conversations_clean.jsonl"
    records = []

    with zipfile.ZipFile(zip_path, "r") as z:
        print(f"Scanning {len(z.namelist())} files inside {zip_path.name} ...")
        for name in z.namelist():
            if not name.lower().endswith(".json"):
                continue
            if "conversations" not in name:
                continue
            print(f"Processing: {name}")
            try:
                data = json.loads(z.read(name).decode("utf-8"))
            except Exception as e:
                print(f"  ⚠️ Skipping {name}: {e}")
                continue

            if isinstance(data, list):
                for convo in data:
                    parse_conversation(convo, records)
            elif isinstance(data, dict):
                parse_conversation(data, records)
            else:
                print(f"  ⚠️ Unrecognized format in {name} ({type(data)})")

    # Write output JSONL
    with open(output_path, "w", encoding="utf-8") as out:
        for rec in records:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n✅ Extracted {len(records)} messages from {zip_path.name}")
    print(f"Saved to {output_path}")
    return output_path


def parse_conversation(convo, records):
    """Handle both 'messages' (old) and 'mapping' (new) conversation formats."""
    if not isinstance(convo, dict):
        return

    title = convo.get("title") or convo.get("meta", {}).get("conversation_title", "Untitled")
    time = convo.get("create_time") or convo.get("meta", {}).get("time_iso") or datetime.now().isoformat()

    # 1. Newer (mapping) format
    mapping = convo.get("mapping")
    if mapping:
        for node in mapping.values():
            msg = node.get("message")
            if not msg:
                continue
            role = msg.get("author", {}).get("role", "user")
            content_obj = msg.get("content")
            if not content_obj:
                continue
            parts = content_obj.get("parts", [])
            for part in parts:
                # Some parts are dicts (e.g., images, citations)
                if isinstance(part, dict):
                    text = json.dumps(part, ensure_ascii=False)
                else:
                    text = str(part).strip()
                if not text:
                    continue
                records.append({
                    "timestamp": time,
                    "conversation": title,
                    "role": role,
                    "content": text
                })

    # 2. Older (messages) format
    elif "messages" in convo:
        for msg in convo["messages"]:
            if not isinstance(msg, dict):
                continue
            content = (msg.get("content") or "").strip()
            if not content:
                continue
            records.append({
                "timestamp": time,
                "conversation": title,
                "role": msg.get("role", "user"),
                "content": content
            })


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python extractor_chatgpt_export.py /path/to/chatGPT-download.zip [output_dir]")
        sys.exit(1)

    zip_file = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "./logs"
    extract_from_zip(zip_file, out_dir)

