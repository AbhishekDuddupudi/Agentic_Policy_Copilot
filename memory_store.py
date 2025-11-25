# memory_store.py
"""
File-based long-term memory for the Agentic Policy Copilot.

- user_profiles.json: per-user semantic profile
- episodes.jsonl: episodic summaries of interactions
"""

import json
from pathlib import Path
from typing import Dict, Any

# Paths under the repo's data/ folder
PROFILE_PATH = Path("data/user_profiles.json")
EPISODES_PATH = Path("data/episodes.jsonl")


def load_user_profile(user_id: str) -> Dict[str, Any]:
    """
    Load a user's profile (long-term semantic memory) from JSON.

    If the file or user_id doesn't exist, return an empty dict.
    """
    if PROFILE_PATH.exists():
        try:
            profiles = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            profiles = {}
    else:
        profiles = {}

    return profiles.get(user_id, {})


def save_user_profile(user_id: str, profile: Dict[str, Any]) -> None:
    """
    Upsert the user's profile and write all profiles back to disk.
    """
    if PROFILE_PATH.exists():
        try:
            profiles = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            profiles = {}
    else:
        profiles = {}

    profiles[user_id] = profile

    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROFILE_PATH.write_text(json.dumps(profiles, indent=2), encoding="utf-8")


def append_episode(user_id: str, summary: str) -> None:
    """
    Append an episodic summary of this interaction to episodes.jsonl.

    Each line is a JSON object: {"user_id": ..., "summary": ...}
    """
    EPISODES_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {"user_id": user_id, "summary": summary}

    with EPISODES_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")