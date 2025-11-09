# Emotion Labs: Exploring Emotional Intelligence and Synchrony in Human–AI Dialogue
**Todd McCaffrey**  
*MSc Cyberpsychology Candidate, Atlantic Technological University*

## Overview
Emotion Labs explores emotional synchrony and drift in human–AI dialogue through ChatGPT exports and local LLM processing.

### Key Modules
- extractor_chatgpt_export.py
- emotion_rollercoaster.py
- emotion_synchrony.py
- emotion_lag_synchrony.py
- investigator.py
- detective.py
- emotion_rewriter_local.py
- emotion_baseline_fingerprint.py
- emotion_lexical_overlap.py
- worry_points.py

### Usage Example
```bash
python extractor_chatgpt_export.py chatGPT-download.zip
python emotion_rollercoaster.py ./logs/conversations_clean.jsonl
python emotion_synchrony.py
python emotion_rewriter_local.py ./logs/conversations_clean.jsonl
```
