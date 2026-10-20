"""Conversation identity belongs to the Streamlit session, never an MCP connection."""

from uuid import uuid4


def chat_context(state):
    if "conversation_id" not in state:
        state["conversation_id"] = str(uuid4())
    return {
        "conversation_id": state["conversation_id"],
        "allow_web_search": bool(state.get("allow_web_search", False)),
    }


def reset_chat_context(state):
    state["conversation_id"] = str(uuid4())
    state["allow_web_search"] = False
