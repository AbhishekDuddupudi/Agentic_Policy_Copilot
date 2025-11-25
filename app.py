# app.py
"""
Simple CLI interface for the Agentic Policy & Ticket Copilot.

- Builds the LangGraph StateGraph.
- Uses a fixed thread_id so the checkpointer gives us short-term memory.
- Reads user input, appends it as a HumanMessage, runs one graph turn,
  and prints the agent's reply.
"""

from typing import Dict, Any

from langchain_core.messages import HumanMessage
from graph import build_graph


def main() -> None:
    # Build the compiled graph application
    app = build_graph()

    # For demo, we use a fixed user/thread id
    thread_id = "demo-user-123"
    user_id = "demo-user-123"

    # Initial state for this conversation
    state: Dict[str, Any] = {
        "messages": [],
        "user_id": user_id,
    }

    print("=== Agentic Policy & Ticket Copilot ===")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if user_input.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break

        if not user_input:
            continue

        # Append the latest HumanMessage to the conversation
        state.setdefault("messages", []).append(HumanMessage(content=user_input))

        # Invoke one turn of the graph.
        # thread_id tells the checkpointer which short-term memory to load/save.
        state = app.invoke(
            state,
            config={"configurable": {"thread_id": thread_id}},
        )

        # The last message in state["messages"] should be the AI's reply
        messages = state.get("messages", [])
        if not messages:
            print("Agent: (no response, something went wrong)\n")
            continue

        last_msg = messages[-1]
        print(f"Agent: {last_msg.content}\n")


if __name__ == "__main__":
    main()