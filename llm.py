# llm.py
"""
LLM wrapper for the Agentic Policy Copilot.

We expose a single helper:
    call_llm(messages: List[BaseMessage]) -> AIMessage

All LangGraph nodes will use this instead of creating models directly.
"""

from typing import List
import os

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, AIMessage
from langchain_openai import ChatOpenAI

# Load environment variables from .env (OPENAI_API_KEY, etc.)
load_dotenv()

# Optional: you can assert the key is present to catch config issues early
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env file.")

# Create a single shared chat model instance.
_chat = ChatOpenAI(
    model="gpt-4o-mini",  # you can change this if needed
    temperature=0.2,
)


def call_llm(messages: List[BaseMessage]) -> AIMessage:
    """
    Call the chat model with a list of messages (System/Human/AI)
    and return the AIMessage response.

    Parameters
    ----------
    messages : List[BaseMessage]
        The conversation so far (system + user + assistant messages).

    Returns
    -------
    AIMessage
        The model's response message.
    """
    response = _chat.invoke(messages)
    return response