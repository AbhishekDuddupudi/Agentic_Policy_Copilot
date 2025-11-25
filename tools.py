# tools.py
"""
Agent tools for the Agentic Policy Copilot.

- search_policies(query): naive keyword-based 'RAG' over policies/*.txt
- get_user_profile_tool(user_id): fetch long-term user profile
- update_user_profile_tool(user_id, new_fields): update + persist user profile
"""

from pathlib import Path
from typing import List, Dict, Any

from langchain_core.documents import Document

from memory_store import load_user_profile, save_user_profile

# Folder where plain-text policy files live
POLICY_DIR = Path("policies")


def search_policies(query: str, top_k: int = 3) -> List[Document]:
    """
    More robust keyword-based 'RAG' tool.

    - Split the query into words.
    - Use non-trivial words (length > 3) as keywords.
    - A file matches if ANY keyword appears in its text.
    """
    query = query.strip().lower()
    if not query:
        return []

    raw_words = query.split()
    keywords = [w.strip(".,?!") for w in raw_words if len(w) > 3]

    if not keywords:
        keywords = [query]

    docs: List[Document] = []

    for path in POLICY_DIR.glob("*.txt"):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue

        text_lower = text.lower()

        if any(kw in text_lower for kw in keywords):
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": str(path)},
                )
            )

        if len(docs) >= top_k:
            break

    return docs


def get_user_profile_tool(user_id: str) -> Dict[str, Any]:
    """
    Tool wrapper: fetch the long-term user profile for this user_id.
    """
    return load_user_profile(user_id)


def update_user_profile_tool(user_id: str, new_fields: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tool wrapper: update the user's profile in long-term memory.

    - Loads existing profile
    - Merges new_fields into it
    - Saves back to disk
    - Returns the updated profile
    """
    profile = load_user_profile(user_id)
    profile.update(new_fields)
    save_user_profile(user_id, profile)
    return profile