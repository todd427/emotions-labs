#!/usr/bin/env python3
"""
emo_view.py — FastAPI viewer for Emotion-Labs analysis output
Usage:
    python emo_view.py --port 8010
"""

import argparse, glob, json, os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI(title="Emotion-Labs Viewer")

# --- Static files ---
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# --- Data helpers ---
def find_latest_jsonl():
    analysis_dir = os.path.join("logs", "analysis")
    if not os.path.isdir(analysis_dir):
        return None
    files = glob.glob(os.path.join(analysis_dir, "rec*.jsonl"))
    return max(files, key=os.path.getmtime) if files else None


def load_data():
    """Load data from the latest or default rec*.jsonl file."""
    path = "logs/recommendations.jsonl" if os.path.exists("logs/recommendations.jsonl") else find_latest_jsonl()
    if not path:
        return [], None
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]
    return data, path


# --- Routes ---
@app.get("/emo/", response_class=HTMLResponse)
def list_emotions():
    data, path = load_data()
    if not data:
        return HTMLResponse("""
        <html><head><title>Emotion-Labs Viewer</title>
        <link rel="stylesheet" href="/static/style.css"></head>
        <body><h1>Emotion-Labs Viewer</h1>
        <p><b>Source:</b> none</p>
        <p>No Emotion-Labs data found. Run <code>emowarn.py</code> to generate results.</p>
        </body></html>
        """)

    cards = []
    for i, d in enumerate(data, 1):
        examples = " ".join(d.get("examples", []))
        cards.append(f"""
        <div class="card">
          <h3><a href="/emo/{i}">{i}. {d['emotion'].title()} — {d['date']}</a></h3>
          <p>{examples}</p>
          <b>{d['recommendation']}</b>
        </div>""")

    html = f"""
    <html>
      <head>
        <title>Emotion-Labs Viewer</title>
        <link rel="stylesheet" href="/static/style.css">
      </head>
      <body>
        <h1>Emotion-Labs Viewer</h1>
        <p><b>Source:</b> {os.path.basename(path)}</p>
        <div class="grid">
          {''.join(cards)}
        </div>
      </body>
    </html>
    """
    return HTMLResponse(html)


@app.get("/emo/{idx}", response_class=HTMLResponse)
def show_emotion(idx: int):
    """Return a detail view for a specific message."""
    data, path = load_data()
    if not data or idx < 1 or idx > len(data):
        return HTMLResponse(
            f"<html><body><h1>404 Not Found</h1><p>No entry #{idx} found.</p></body></html>",
            status_code=404,
        )

    d = data[idx - 1]
    examples = "<br>".join(d.get("examples", []))
    html = f"""
    <html>
      <head>
        <title>Emotion {idx}</title>
        <link rel="stylesheet" href="/static/style.css">
      </head>
      <body>
        <a href="/emo/">← Back to list</a>
        <div class="card">
          <h3>{idx}. {d['emotion'].title()} — {d['date']}</h3>
          <p>{examples}</p>
          <b>{d['recommendation']}</b>
        </div>
      </body>
    </html>
    """
    return HTMLResponse(html)


# --- Entrypoint ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", 8010)))
    args = parser.parse_args()
    print(f"🚀 Emotion-Labs Viewer → http://localhost:{args.port}/emo/")
    uvicorn.run("emo_view:app", host="0.0.0.0", port=args.port, reload=False)


if __name__ == "__main__":
    main()

