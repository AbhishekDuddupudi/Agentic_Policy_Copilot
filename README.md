# Agentic Policy & Ticket Copilot (LangGraph Demo)

A small **agentic AI** demo that acts as an internal **Policy & Ticket Copilot**.  
It uses **LangGraph** + **LangChain** to:

- Decide when to **answer directly** vs **call tools** (policy search, profile updates).
- Maintain **short-term conversational memory** per thread.
- Maintain **long-term memory** of user preferences and interaction history.
- Ground answers in **policy documents** stored as local `.txt` files.

This project is a compact example of **agentic orchestration + layered memory design**.

---

## Quick project summary

- Built with a **LangGraph StateGraph** and a shared `AgentState`.
- Uses a **planner node + tool node** pattern (agent decides when to search policies vs update profile vs answer).
- Implements **multiple memory types**:
  - Short-term conversational memory (checkpointer + `thread_id`).
  - Long-term user profile memory (JSON store).
  - Long-term policy knowledge via retrieval (RAG-style over `.txt` files).
  - Episodic memory via JSONL interaction logs.
- Exposes a simple **CLI chat interface** to interact with the copilot.

---

## Concepts used (high-level)

This project demonstrates the following concepts:

- **Agentic AI patterns**
  - Planner → Tool → Answer flow.
  - Rule-based routing combined with LLM-based planning.
  - Tool calling for policy search and profile updates.
- **Memory**
  - Short-term / working memory (conversation history per `thread_id`).
  - Long-term semantic memory (user profile store).
  - Long-term domain memory (policy docs + retrieval).
  - Episodic memory (episodes log).
- **LangGraph**
  - `StateGraph` with typed state (`AgentState`).
  - Node-based orchestration and conditional edges.
  - `MemorySaver` checkpointer for per-thread state.
- **RAG-style retrieval**
  - Simple keyword-based retrieval over `.txt` policies.
  - Injecting retrieved documents into the LLM context.
- **Persistent storage**
  - JSON (`user_profiles.json`) for profile data.
  - JSONL (`episodes.jsonl`) for interaction histories.

---

## What it does

From a simple CLI, you can:

- Chat with the agent as if you’re a support advisor.
- Tell it profile info like:  
  > “My name is Abhishek. My favorite color is blue.”
- Ask policy questions like:  
  > “What is your refund policy?”  
  > “What is the refund eligibility criteria?”  
  > “How do you escalate tickets?”

Under the hood, the graph:

1. **Plans** the next action (answer vs use tools).
2. **Runs tools** (policy search, profile update) when needed.
3. **Answers** using conversation + profile + policy snippets.
4. **Logs the episode** so the interaction history can be analyzed later.

---

## Agentic design & LangGraph architecture

The application is built as a **LangGraph `StateGraph`** with a shared `AgentState` and the following nodes.

### State (`AgentState` in `graph.py`)

The shared state flowing between nodes includes:

- `messages: List[BaseMessage]` — chat history (short-term memory).
- `user_id: str` — user identifier.
- `user_profile: Dict[str, Any]` — long-term profile (e.g., `favorite_color`).
- `profile_updated: bool` — flag when we write to profile.
- `policy_query: Optional[str]` — last policy search query.
- `retrieved_policies: List[Document]` — policy snippets from `search_policies`.
- `tool_results: Dict[str, Any]` — placeholder for other tool outputs.
- `next_action: str` — planner’s decision:
  - `"answer" | "search_policies" | "update_profile" | "ask_clarification"`.
- `done: bool` — marks the end of a turn.

### Nodes

- **`ingest_user`**  
  - Entry node.  
  - Ensures `user_profile` is loaded from disk for this `user_id`.  
  - Initializes defaults (`retrieved_policies`, `tool_results`, `done`).

- **`planner`**  
  - Planner node (agent “brain”).  
  - First does a **simple rule-based route**:  
    - If the last user message contains policy-ish keywords (`refund`, `loan`, `privacy`, `ticket`, `escalation`, `policy`) and the agent hasn’t already searched for that exact question, it sets `next_action = "search_policies"`.  
  - Otherwise, it calls the LLM with a system prompt that asks it to choose exactly one action:
    - `search_policies`, `update_profile`, `ask_clarification`, or `answer` (using an `Action: ...` format).
  - Writes `next_action` into state and appends the planner’s AIMessage to `messages` for traceability.

- **`run_tools`**  
  - Executes tools based on `next_action`:
    - `"search_policies"`  
      - Finds the last `HumanMessage` and uses its content as the query.  
      - Calls `search_policies(query)` and stores hits in `retrieved_policies`.  
    - `"update_profile"`  
      - Looks at the last `HumanMessage`.  
      - Simple rule: if it contains `"favorite color"`, captures the last word as the color (e.g., `"blue"`).  
      - Calls `update_user_profile_tool(user_id, {"favorite_color": ...})`.  
      - Updates `user_profile` and sets `profile_updated=True`.

- **`answer`**  
  - Builds a system prompt telling the LLM to:
    - Use `[USER_PROFILE]` and any `[POLICY ...]` blocks from `retrieved_policies`.
    - Ground answers in policy text when available.
    - Mention that it checked policy documents if it used them.
  - Creates a `SystemMessage` for instructions and another `SystemMessage` with:
    - `[USER_PROFILE]\n{user_profile}`  
    - `[POLICY from path]\n<policy text>` blocks  
  - Calls `call_llm([system, context_msg] + messages)` and appends the AIMessage to `messages`.
  - Sets `done=True`.

- **`log_episode_node`**  
  - Takes the last AIMessage as a summary.  
  - Appends a JSONL record to `data/episodes.jsonl`:
    - `{"user_id": ..., "summary": "..."}`  

### Graph wiring

The graph edges are:

```text
ingest_user → planner → answer → log_episode → END
                       ↘ run_tools ↗