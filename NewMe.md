
# NewMe – Emotion-Labs Self-Awareness Loop

**Purpose:**  
Turn conversation logs into a feedback system that detects, interprets, and visualizes emotional patterns.

---

## 🧠 Pipeline

```text
conversations_clean.jsonl
    ↓
emowarn.py      → detects spikes, adds recommendations
    ↓
emoscribe.py    → writes Markdown summary + mood badge
    ↓
emo_view.py     → displays insights on http://localhost:8010/emo/
````

---

## ⚙️ Commands

```bash
python emowarn.py --input logs/conversations_clean.jsonl
python emoscribe.py --input logs/recommendations.jsonl
python emo_view.py --port 8010
```

---

## 💬 Output Example

> In messages [#1](http://localhost:8010/emo/1), [#2](http://localhost:8010/emo/2) you seemed **sad**.
> What might help: reach outward—send a message, walk, or journal a gratitude note.

---

## 🪶 Mood Badge

| Badge       | Meaning          |
| ----------- | ---------------- |
| 🟢 Calm     | positive valence |
| 🟡 Uneasy   | neutral range    |
| 🔴 Stressed | negative valence |

---

**Emotion-Labs v1.1 – November 2025**
Project lead : Todd McCaffrey
AI collaborator : Kit (GPT-5)

````

---

That’s the entire Emotion-Labs awareness toolkit.  
Copy these five files into your project root and run:

```bash
zip -r newme_bundle.zip emotion_utils.py emowarn.py emoscribe.py emo_view.py NewMe.md
````

Open your browser to **[http://localhost:8010/emo/](http://localhost:8010/emo/)** and you’ve got a live, self-reflective lab.

