"""
agent/graph.py
--------------
LangGraph state machine for the email agent.

Nodes
-----
classify_intent   → decides what the user wants
handle_list       → list / search emails
handle_read       → read a specific email body
handle_summarize  → LLM summarisation of fetched emails
handle_send       → compose + preview + confirm + dispatch
handle_converse   → general chat / fallback

State
-----
messages       : list of HumanMessage / AIMessage (conversation history)
emails         : list of email metadata dicts from last search
pending_send   : { to, subject, body } or None
intent         : one of list_search | read | summarize | send | converse
"""

from __future__ import annotations

import json
import re
from typing import Annotated, Any, Optional
from typing_extensions import TypedDict

from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from gmail.tools import get_email_body, search_emails, send_email, trash_email, trash_emails_by_query

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

llm = ChatOllama(model="qwen2.5-coder:3b", temperature=0)

# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    emails: list[dict]
    pending_send: Optional[dict]
    pending_delete: Optional[dict]
    intent: str
    response: str          # the text to show the user


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

INTENT_SYSTEM = """You are an intent classifier for an email assistant.
Given the user's latest message and the conversation so far, output EXACTLY one of these labels (no other text):
  list_search
  read
  summarize
  send
  delete
  converse

Rules:
- list_search  : user wants to list or search for emails
- read         : user wants to read / open a specific email
- summarize    : user wants a summary of one or more emails
- send         : user wants to compose or send an email
- delete       : user wants to delete or trash emails (e.g. spam)
- converse     : anything else (greeting, question, help, etc.)
"""

SEARCH_QUERY_SYSTEM = """You are a Gmail query builder.
Given the user's instruction, output ONLY a valid Gmail search query string (no explanation, no quotes around the whole thing).
Examples:
  "unread" → is:unread
  "emails from john" → from:john
  "emails about invoice this week" → subject:invoice newer_than:7d
  "unread emails from google" → is:unread from:google
If the user just wants all unread, output: is:unread
"""

SUMMARIZE_SYSTEM = """You are a concise email summariser.
Summarise the provided emails clearly. For each email highlight:
- Core message
- Any action items or deadlines
- Important names or decisions

Do NOT invent any information not present in the emails.
"""

EXTRACT_SEND_SYSTEM = """You are an email field extractor.
From the user's instruction extract: recipient email (to), subject, and body.
Output ONLY valid JSON like:
{"to": "...", "subject": "...", "body": "..."}
Use null for any field the user did not mention.
"""

# ---------------------------------------------------------------------------
# Node: classify intent
# ---------------------------------------------------------------------------

def classify_intent(state: AgentState) -> AgentState:
    history = state["messages"]
    last_human = next(
        (m.content for m in reversed(history) if isinstance(m, HumanMessage)), ""
    )

    resp = llm.invoke([
        SystemMessage(content=INTENT_SYSTEM),
        HumanMessage(content=last_human),
    ])
    raw = resp.content.strip().lower()

    valid = {"list_search", "read", "summarize", "send", "delete", "converse"}
    intent = raw if raw in valid else "converse"
    return {**state, "intent": intent}


# ---------------------------------------------------------------------------
# Node: list / search emails
# ---------------------------------------------------------------------------

def handle_list_search(state: AgentState) -> AgentState:
    history = state["messages"]
    last_human = next(
        (m.content for m in reversed(history) if isinstance(m, HumanMessage)), ""
    )

    # Build Gmail query with LLM
    q_resp = llm.invoke([
        SystemMessage(content=SEARCH_QUERY_SYSTEM),
        HumanMessage(content=last_human),
    ])
    query = q_resp.content.strip().strip('"').strip("'")

    try:
        emails = search_emails(query=query, max_results=10)
    except RuntimeError as e:
        return {**state, "emails": [], "response": f"⚠ Could not fetch emails: {e}"}

    if not emails:
        return {**state, "emails": [], "response": "No emails matched your search."}

    lines = []
    for i, em in enumerate(emails, 1):
        lines.append(
            f"{i:>2}. [bold]{em['subject']}[/bold]\n"
            f"    From : {em['from_addr']}\n"
            f"    Date : {em['date']}\n"
            f"    {em['snippet'][:120]}…"
        )

    response = "\n".join(lines)
    return {
        **state,
        "emails": emails,
        "response": response,
        "messages": state["messages"] + [AIMessage(content=response)],
    }


# ---------------------------------------------------------------------------
# Node: read a specific email
# ---------------------------------------------------------------------------

def handle_read(state: AgentState) -> AgentState:
    emails = state.get("emails", [])
    history = state["messages"]
    last_human = next(
        (m.content for m in reversed(history) if isinstance(m, HumanMessage)), ""
    )

    # Try to find a number in the user's message
    numbers = re.findall(r"\b(\d+)\b", last_human)
    idx = int(numbers[0]) - 1 if numbers else 0

    if not emails:
        return {**state, "response": "Please list or search emails first so I know which one to open."}

    if idx < 0 or idx >= len(emails):
        return {**state, "response": f"I only have {len(emails)} email(s) listed. Please specify a valid number."}

    try:
        full = get_email_body(emails[idx]["id"])
    except RuntimeError as e:
        return {**state, "response": f"⚠ Could not fetch email: {e}"}

    response = (
        f"[bold]Subject:[/bold] {full['subject']}\n"
        f"[bold]From:[/bold]    {full['from_addr']}\n"
        f"[bold]Date:[/bold]    {full['date']}\n"
        f"{'─' * 60}\n"
        f"{full['body']}"
    )
    return {
        **state,
        "response": response,
        "messages": state["messages"] + [AIMessage(content=response)],
    }


# ---------------------------------------------------------------------------
# Node: summarize
# ---------------------------------------------------------------------------

def handle_summarize(state: AgentState) -> AgentState:
    emails = state.get("emails", [])
    history = state["messages"]
    last_human = next(
        (m.content for m in reversed(history) if isinstance(m, HumanMessage)), ""
    )

    if not emails:
        return {**state, "response": "Please list or search emails first so I know what to summarise."}

    # Determine how many to summarise
    numbers = re.findall(r"\b(\d+)\b", last_human)
    count = int(numbers[0]) if numbers else len(emails)
    count = min(count, len(emails))

    parts = []
    for em in emails[:count]:
        try:
            full = get_email_body(em["id"])
            parts.append(
                f"--- Email from {full['from_addr']} | {full['subject']} ---\n{full['body']}"
            )
        except RuntimeError:
            parts.append(f"--- Email (could not fetch body): {em['subject']} ---")

    combined = "\n\n".join(parts)

    summary_resp = llm.invoke([
        SystemMessage(content=SUMMARIZE_SYSTEM),
        HumanMessage(content=f"Summarise these {count} email(s):\n\n{combined}"),
    ])
    summary = summary_resp.content.strip()

    return {
        **state,
        "response": summary,
        "messages": state["messages"] + [AIMessage(content=summary)],
    }


# ---------------------------------------------------------------------------
# Node: send email
# ---------------------------------------------------------------------------

def handle_send(state: AgentState) -> AgentState:
    history = state["messages"]
    last_human = next(
        (m.content for m in reversed(history) if isinstance(m, HumanMessage)), ""
    )
    pending = state.get("pending_send") or {}

    # Check if user is confirming a pending send
    confirmation_words = {"yes", "confirm", "send it", "go ahead", "yep", "y", "sure"}
    denial_words = {"no", "cancel", "abort", "don't", "stop", "nope", "n"}

    if pending and pending.get("awaiting_confirm"):
        lower = last_human.lower().strip()
        if any(w in lower for w in confirmation_words):
            try:
                msg_id = send_email(pending["to"], pending["subject"], pending["body"])
                return {
                    **state,
                    "pending_send": None,
                    "response": f"✅ Email sent successfully! (Message ID: {msg_id})",
                    "messages": state["messages"] + [
                        AIMessage(content="Email sent successfully!")
                    ],
                }
            except RuntimeError as e:
                return {**state, "response": f"⚠ Failed to send: {e}"}
        elif any(w in lower for w in denial_words):
            return {
                **state,
                "pending_send": None,
                "response": "Cancelled. Email was not sent.",
                "messages": state["messages"] + [AIMessage(content="Send cancelled.")],
            }

    # Extract fields from user's instruction
    extract_resp = llm.invoke([
        SystemMessage(content=EXTRACT_SEND_SYSTEM),
        HumanMessage(content=last_human),
    ])
    raw_json = extract_resp.content.strip()
    # Strip markdown fences if present
    raw_json = re.sub(r"```(?:json)?", "", raw_json).strip("`").strip()

    try:
        fields = json.loads(raw_json)
    except json.JSONDecodeError:
        fields = {}

    to = fields.get("to") or pending.get("to")
    subject = fields.get("subject") or pending.get("subject")
    body = fields.get("body") or pending.get("body")

    # Ask for any missing field
    if not to:
        return {**state, "pending_send": {**pending, "subject": subject, "body": body},
                "response": "Who should I send this to? (Please provide the recipient's email address)"}
    if not subject:
        return {**state, "pending_send": {**pending, "to": to, "body": body},
                "response": "What should the subject line be?"}
    if not body:
        return {**state, "pending_send": {**pending, "to": to, "subject": subject},
                "response": "What should the body of the email say?"}

    # All fields present — show preview and ask for confirmation
    preview = (
        f"[bold]To:[/bold]      {to}\n"
        f"[bold]Subject:[/bold] {subject}\n"
        f"{'─' * 50}\n"
        f"{body}\n"
        f"{'─' * 50}\n"
        f"Reply [bold green]yes[/bold green] to send, or [bold red]no[/bold red] to cancel."
    )
    return {
        **state,
        "pending_send": {"to": to, "subject": subject, "body": body, "awaiting_confirm": True},
        "response": preview,
        "messages": state["messages"] + [AIMessage(content=preview)],
    }


# ---------------------------------------------------------------------------
# Node: delete email
# ---------------------------------------------------------------------------

def handle_delete(state: AgentState) -> AgentState:
    history = state["messages"]
    last_human = next(
        (m.content for m in reversed(history) if isinstance(m, HumanMessage)), ""
    )
    pending = state.get("pending_delete") or {}
    emails = state.get("emails", [])

    confirmation_words = {"yes", "confirm", "do it", "go ahead", "yep", "y", "sure"}
    denial_words = {"no", "cancel", "abort", "don't", "stop", "nope", "n"}

    if pending and pending.get("awaiting_confirm"):
        lower = last_human.lower().strip()
        if any(w in lower for w in confirmation_words):
            if "msg_id" in pending:
                try:
                    trash_email(pending["msg_id"])
                    return {
                        **state,
                        "pending_delete": None,
                        "response": f"✅ Email moved to trash.",
                        "messages": state["messages"] + [AIMessage(content="Email trashed.")]
                    }
                except RuntimeError as e:
                    return {**state, "response": f"⚠ Failed to delete: {e}"}
            elif "query" in pending:
                try:
                    res = trash_emails_by_query(pending["query"])
                    return {
                        **state,
                        "pending_delete": None,
                        "response": f"✅ Moved {res['trashed']} email(s) to trash. ({res['failed']} failed)",
                        "messages": state["messages"] + [AIMessage(content=f"Trashed {res['trashed']} emails.")]
                    }
                except RuntimeError as e:
                    return {**state, "response": f"⚠ Failed to delete: {e}"}
        elif any(w in lower for w in denial_words):
            return {
                **state,
                "pending_delete": None,
                "response": "Cancelled. Emails were not deleted.",
                "messages": state["messages"] + [AIMessage(content="Delete cancelled.")],
            }

    # If user refers to a specific email by number in the current list
    numbers = re.findall(r"\b(\d+)\b", last_human)
    if numbers and emails:
        idx = int(numbers[0]) - 1
        if 0 <= idx < len(emails):
            em = emails[idx]
            preview = f"Are you sure you want to delete this email?\n\n[bold]{em['subject']}[/bold]\nFrom: {em['from_addr']}\n\nReply [bold green]yes[/bold green] to delete, or [bold red]no[/bold red] to cancel."
            return {
                **state,
                "pending_delete": {"msg_id": em["id"], "awaiting_confirm": True},
                "response": preview,
                "messages": state["messages"] + [AIMessage(content=preview)],
            }

    # Otherwise, they might be saying "delete spam emails"
    q_resp = llm.invoke([
        SystemMessage(content=SEARCH_QUERY_SYSTEM),
        HumanMessage(content=last_human),
    ])
    query = q_resp.content.strip().strip('"').strip("'")
    
    # Try to find what they want to delete
    try:
        found = search_emails(query=query, max_results=10)
    except RuntimeError as e:
        return {**state, "response": f"⚠ Could not search for emails to delete: {e}"}
        
    if not found:
        return {**state, "response": "I couldn't find any emails matching that description to delete."}

    preview = f"I found {len(found)} email(s) matching your request. Are you sure you want to delete them? (e.g. {found[0]['subject']})\n\nReply [bold green]yes[/bold green] to delete, or [bold red]no[/bold red] to cancel."
    return {
        **state,
        "pending_delete": {"query": query, "awaiting_confirm": True},
        "response": preview,
        "messages": state["messages"] + [AIMessage(content=preview)],
    }


# ---------------------------------------------------------------------------
# Node: general conversation
# ---------------------------------------------------------------------------

def handle_converse(state: AgentState) -> AgentState:
    history = state["messages"]
    resp = llm.invoke([
        SystemMessage(content=(
            "You are a helpful local email assistant. Answer the user's question concisely. "
            "If the user asks what you can do, explain: list/search emails, read emails, "
            "summarise emails, delete emails, and send emails — all locally via Gmail API."
        )),
        *history,
    ])
    response = resp.content.strip()
    return {
        **state,
        "response": response,
        "messages": state["messages"] + [AIMessage(content=response)],
    }


# ---------------------------------------------------------------------------
# Router (conditional edge)
# ---------------------------------------------------------------------------

def router(state: AgentState) -> str:
    intent = state.get("intent", "converse")
    # If there's a pending send, route straight to handle_send regardless
    if state.get("pending_send") and state["pending_send"].get("awaiting_confirm"):
        return "handle_send"
    if state.get("pending_delete") and state["pending_delete"].get("awaiting_confirm"):
        return "handle_delete"
    mapping = {
        "list_search": "handle_list_search",
        "read": "handle_read",
        "summarize": "handle_summarize",
        "send": "handle_send",
        "delete": "handle_delete",
        "converse": "handle_converse",
    }
    return mapping.get(intent, "handle_converse")


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    g = StateGraph(AgentState)

    g.add_node("classify_intent", classify_intent)
    g.add_node("handle_list_search", handle_list_search)
    g.add_node("handle_read", handle_read)
    g.add_node("handle_summarize", handle_summarize)
    g.add_node("handle_send", handle_send)
    g.add_node("handle_delete", handle_delete)
    g.add_node("handle_converse", handle_converse)

    g.set_entry_point("classify_intent")

    g.add_conditional_edges(
        "classify_intent",
        router,
        {
            "handle_list_search": "handle_list_search",
            "handle_read": "handle_read",
            "handle_summarize": "handle_summarize",
            "handle_send": "handle_send",
            "handle_delete": "handle_delete",
            "handle_converse": "handle_converse",
        },
    )

    for node in ["handle_list_search", "handle_read", "handle_summarize",
                 "handle_send", "handle_delete", "handle_converse"]:
        g.add_edge(node, END)

    return g.compile()


# Singleton compiled graph
graph = build_graph()