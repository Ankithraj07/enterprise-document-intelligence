import json
import os
from datetime import datetime

FEEDBACK_FILE = "feedback_log.jsonl"

def log_feedback(question: str, answer: str, sources: list, rating: str, comment: str = ""):
    """
    Log a feedback entry to the JSON file.
    
    rating: "positive" or "negative"
    comment: optional text from user explaining what was wrong
    """
    entry = {
        "timestamp": datetime.now().isoformat(),
        "rating": rating,
        "question": question,
        "answer": answer,
        "sources": sources,
        "comment": comment
    }
    with open(FEEDBACK_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

def get_all_feedback():
    """Read all feedback entries. Returns a list of dicts."""
    if not os.path.exists(FEEDBACK_FILE):
        return []
    entries = []
    with open(FEEDBACK_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries

def get_negative_feedback():
    """Return only the bad answers — useful for debugging retrieval quality."""
    return [e for e in get_all_feedback() if e["rating"] == "negative"]

def get_feedback_summary():
    """Return a quick summary dict for the Streamlit sidebar."""
    all_entries = get_all_feedback()
    if not all_entries:
        return {"total": 0, "positive": 0, "negative": 0, "rate": "N/A"}
    
    positive = sum(1 for e in all_entries if e["rating"] == "positive")
    negative = len(all_entries) - positive
    rate = round((positive / len(all_entries)) * 100, 1)
    
    return {
        "total": len(all_entries),
        "positive": positive,
        "negative": negative,
        "rate": f"{rate}%"
    }