# graph.py
from typing import TypedDict, List, Dict, Any, Optional

"""
LangGraph StateGraph for the Agentic Policy & Ticket Copilot.

High-level flow:
- ingest_user: state comes in with latest HumanMessage already in messages.
- planner: LLM decides next_action: "answer", "search_policies", "update_profile", "ask_clarification".
- run_tools: executes tools based on next_action (policy search, profile update).
- answer: LLM generates final reply using conversation, user_profile, and retrieved policies.
- log_episode: append a short episodic summary.

Short-term memory:
- Managed by LangGraph's checkpointer (MemorySaver) keyed by thread_id.

Long-term memory:
- user_profiles.json via memory_store
- policies/*.txt via search_policies tool
- episodes.jsonl via append_episode
"""

from typing import TypedDict, List, Dict, Any, Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)

from llm import call_llm
from tools import search_policies, update_user_profile_tool, get_user_profile_tool
from memory_store import append_episode


class AgentState(TypedDict, total=False):
    """
    Shared state flowing between nodes in the graph.

    Fields
    ------
    messages : List[BaseMessage]
        Conversation history (short-term memory for this thread).
    user_id : str
        Identifier for the end user.

    user_profile : Dict[str, Any]
        Long-term profile loaded from disk (semantic memory).
    profile_updated : bool
        Flag to show if profile was updated in this run.

    policy_query : Optional[str]
        The last policy search query.
    retrieved_policies : List[Any]
        Documents returned by search_policies.
    tool_results : Dict[str, Any]
        Miscellaneous outputs from tools (if needed).

    next_action : str
        Planner's decision: "answer", "search_policies", "update_profile", "ask_clarification".
    done : bool
        Whether this turn is complete.
    """

    messages: List[BaseMessage]
    user_id: str

    user_profile: Dict[str, Any]
    profile_updated: bool

    policy_query: Optional[str]
    retrieved_policies: List[Any]
    tool_results: Dict[str, Any]

    next_action: str
    done: bool


def ingest_user(state: AgentState) -> AgentState:
    """
    Entry node.

    Right now, we assume the latest HumanMessage has already been
    appended to state["messages"] in app.py.

    This is also a good place to ensure user_profile is loaded.
    """
    user_id = state.get("user_id")
    if user_id and "user_profile" not in state:
        # Load existing long-term profile on first turn for this thread.
        profile = get_user_profile_tool(user_id)
        state["user_profile"] = profile
        state["profile_updated"] = False

    # Ensure some defaults exist so later nodes don't crash.
    state.setdefault("retrieved_policies", [])
    state.setdefault("tool_results", {})
    state.setdefault("done", False)

    return state


def planner(state: AgentState) -> AgentState:
    """
    Planner node.

    Uses a simple rule-based step first:
    - If the last user message clearly talks about policies (refund, loan, privacy, tickets),
      and we have NOT already searched for this exact message,
      we force next_action = "search_policies".

    Otherwise, uses the LLM to decide the next_action:
    - "answer"
    - "search_policies"
    - "update_profile"
    - "ask_clarification"
    """
    messages = state.get("messages", [])

    # 1) Cheap keyword router: if user mentions policy-ish words,
    # and we haven't already run search_policies for this exact message,
    # then force a policy search.
    last_human: Optional[HumanMessage] = None
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            last_human = m
            break

    if last_human:
        text = last_human.content.lower()
        policy_keywords = ["refund", "loan", "privacy", "ticket", "escalation", "policy"]

        # Have we already searched using this exact question?
        already_searched = state.get("policy_query") == last_human.content

        if any(kw in text for kw in policy_keywords) and not already_searched:
            state["next_action"] = "search_policies"
            return state

    # 2) Otherwise, ask the LLM to decide.
    system = SystemMessage(
        content=(
            "You are a support copilot for internal policies.\n"
            "You can decide one of these next actions:\n"
            "- search_policies: when you need to look up a policy document.\n"
            "- update_profile: when the user shares stable preferences or profile info (e.g., favorite color).\n"
            "- ask_clarification: when the question is unclear and you need more info.\n"
            "- answer: when you can answer directly from the current context.\n"
            "In your reply, clearly include the chosen action name exactly once in this format:\n"
            "Action: search_policies\n"
            "or\n"
            "Action: update_profile\n"
            "or\n"
            "Action: ask_clarification\n"
            "or\n"
            "Action: answer"
        )
    )

    messages_for_llm = [system] + messages
    ai_msg = call_llm(messages_for_llm)

    # Append planner's reasoning to the conversation for traceability.
    state.setdefault("messages", []).append(ai_msg)

    content_lower = ai_msg.content.lower()

    # Naive parsing: look for "action: ..."
    if "action: search_policies" in content_lower:
        next_action = "search_policies"
    elif "action: update_profile" in content_lower:
        next_action = "update_profile"
    elif "action: ask_clarification" in content_lower:
        next_action = "ask_clarification"
    elif "action: answer" in content_lower:
        next_action = "answer"
    else:
        # Fallback: just answer.
        next_action = "answer"

    state["next_action"] = next_action
    return state

def run_tools(state: AgentState) -> AgentState:
    """
    Tool execution node.

    Executes tools based on state["next_action"]:
    - search_policies: calls search_policies using last user message as query.
    - update_profile: updates user profile based on last user message.
    """
    action = state.get("next_action", "answer")
    user_id = state["user_id"]

    # Make sure we have a messages list
    messages = state.get("messages", [])

    # Find the last HumanMessage content to use as a query or source of info.
    last_human: Optional[HumanMessage] = None
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            last_human = m
            break

    if action == "search_policies":
        query = last_human.content if last_human else ""
        hits = search_policies(query)
        state["policy_query"] = query
        state["retrieved_policies"] = hits

    elif action == "update_profile":
        new_fields: Dict[str, Any] = {}

        if last_human:
            text = last_human.content.strip()

            # Toy rule: if user says "my favorite color is X", capture X.
            lower = text.lower()
            if "favorite color" in lower:
                # Very naive extraction: last word in the sentence.
                # Example: "My favorite color is blue" -> "blue"
                parts = text.replace(".", "").split()
                if parts:
                    new_fields["favorite_color"] = parts[-1]

            # You could add more rules here later (name, location, etc).

        if new_fields:
            updated_profile = update_user_profile_tool(user_id, new_fields)
            state["user_profile"] = updated_profile
            state["profile_updated"] = True

    # Ensure tool_results exists even if we don't use it yet.
    state.setdefault("tool_results", {})
    return state


def answer(state: AgentState) -> AgentState:
    """
    Answer node.

    Generates the final user-facing reply using:
    - conversation history (short-term memory)
    - user_profile (long-term semantic memory)
    - retrieved_policies (RAG-style context)
    """
    system = SystemMessage(
        content=(
            "You are a helpful, concise support copilot.\n"
            "Use the user profile, any retrieved policy snippets, and the "
            "conversation to answer the user's latest question.\n"
            "If you looked up policies, briefly mention that you checked the policy documents.\n"
            "If you updated the user's profile, you may briefly acknowledge it."
        )
    )

    # Build context from long-term memories
    profile_context = f"[USER_PROFILE]\n{state.get('user_profile', {})}\n"

    policy_context = ""
    for doc in state.get("retrieved_policies", []):
        src = doc.metadata.get("source", "unknown")
        policy_context += f"\n[POLICY from {src}]\n{doc.page_content}\n"

    context_msg = SystemMessage(
        content=f"Context for this conversation:\n{profile_context}\n{policy_context}"
    )

    messages = [system, context_msg] + state.get("messages", [])
    ai_msg = call_llm(messages)

    state.setdefault("messages", []).append(ai_msg)
    state["done"] = True
    return state


def log_episode_node(state: AgentState) -> AgentState:
    """
    Episodic logging node.

    Takes the last AIMessage as a summary and appends it to episodes.jsonl.
    """
    messages = state.get("messages", [])
    last_ai: Optional[AIMessage] = None
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            last_ai = m
            break

    if last_ai is not None:
        user_id = state.get("user_id", "unknown-user")
        append_episode(user_id, last_ai.content)

    return state


def build_graph():
    """
    Build and compile the LangGraph StateGraph for this agent.

    Returns
    -------
    app : Compiled graph application
        You can call app.invoke(state, config={"configurable": {"thread_id": ...}})
        from app.py to run one turn of the agent.
    """
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("ingest_user", ingest_user)
    graph.add_node("planner", planner)
    graph.add_node("run_tools", run_tools)
    graph.add_node("answer", answer)
    graph.add_node("log_episode", log_episode_node)

    # Entry point
    graph.set_entry_point("ingest_user")

    # ingest_user -> planner
    graph.add_edge("ingest_user", "planner")

    # planner decides where to go next based on state["next_action"]
    graph.add_conditional_edges(
        "planner",
        lambda state: state["next_action"],
        {
            "answer": "answer",
            "search_policies": "run_tools",
            "update_profile": "run_tools",
            "ask_clarification": "answer",  # we treat this as just answering with a clarification question
        },
    )

    # After tools, go back to planner (agentic loop: think -> act -> observe)
    graph.add_edge("run_tools", "planner")

    # After answer, log episode then END
    graph.add_edge("answer", "log_episode")
    graph.add_edge("log_episode", END)

    # Checkpointer gives us short-term memory across turns per thread_id
    checkpointer = MemorySaver()
    app = graph.compile(checkpointer=checkpointer)
    return app