"""
agent/graph.py
--------------
LangGraph state machine for the email agent.
"""

from __future__ import annotations

import json
import re
import datetime
from typing import Annotated, Any, Optional
from typing_extensions import TypedDict

from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from gmail.tools import get_email_body, search_emails, send_email, trash_email, trash_emails_by_query, modify_message_labels
from gmail.client import get_authenticated_email
from tasks.tools import add_multiple_google_tasks

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------



# Upgraded to Llama 3.1!
llm = ChatOllama(model="llama3.1", temperature=0, num_predict=2048)
import os
from jinja2 import Environment, FileSystemLoader, select_autoescape

current_dir = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(os.path.dirname(current_dir), "templates")
jinja_env = Environment(
    loader=FileSystemLoader(TEMPLATE_DIR),
    autoescape=select_autoescape(["html"]),  # escapes <, &, etc. in email subjects automatically
)

# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    emails: list[dict]
    pending_send: Optional[dict]
    pending_delete: Optional[dict]
    pending_todo: Optional[dict]
    triage_config: Optional[dict]  # <-- ADDED: Allows programmatic bypass of LLM configuration
    intent: str
    response: str


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

INTENT_SYSTEM = """You are an intent classifier for an email assistant.
Given the user's latest message and the conversation so far, output EXACTLY one of these labels (no other text, no punctuation):
  list_search
  read
  summarize
  send
  delete
  label
  todo
  triage
  converse

Rules:
- list_search  : user wants to list or search for emails
- read         : user wants to read / open a specific email
- summarize    : user wants a summary of one or more emails
- send         : user wants to compose or send an email
- delete       : user wants to delete or trash emails
- label        : user wants to archive, star, or mark emails as read/unread
- todo         : user wants to create a reminder, task, or to-do item
- triage       : user wants to run a daily sorting/filtering batch job based on keywords
- converse     : anything else (greeting, question, help, etc.)
"""

TRIAGE_CONFIG_SYSTEM = """You extract configuration for a daily email triage batch job.
Read the user's instructions and output a strict JSON object.

FORMAT REQUIREMENT:
{
  "keywords": ["keyword1", "keyword2"],
  "add_tasks": true or false,
  "label_to_add": "IMPORTANT" or "STARRED" or null,
  "timeframe": "1d" (use "7d" for this week, "1m" for this month, default "1d"),
  "max_results": 40 (increase up to 50 if user asks for a lot of emails),
  "use_semantic": false (true if they specifically ask for semantic, conceptual, or AI search)
}

RULES:
- "keywords" must be a list of strings the user wants to search for.
- "add_tasks" is true if they mention adding tasks, todos, or deadlines.
- "label_to_add" is "IMPORTANT" if they say "mark as imp/important", "STARRED" if they want it starred, else null.
DO NOT wrap the output in markdown code blocks. Output ONLY raw JSON.
"""

EXTRACT_TODO_SYSTEM = """You are a strict data extractor.
You must extract task details from the user's instruction and the conversational context.

CRITICAL INSTRUCTIONS:
- Output ONLY a raw, valid JSON array.
- DO NOT wrap the output in markdown code blocks (```json).
- DO NOT add any conversational text before or after the JSON.
- If you cannot extract a task, output an empty array: []

FORMAT REQUIREMENT:
[
  {
    "title": "Short summary of the task",
    "notes": "Relevant details from the email",
    "due_date": "YYYY-MM-DDTHH:MM:SS.000Z"
  }
]
"""

SEARCH_QUERY_SYSTEM = """You are a Gmail query builder.
Given the user's instruction, output ONLY a valid Gmail search query string (no explanation, no quotes around the whole thing).
"""

SUMMARIZE_SYSTEM = """You are a concise email summariser.
Summarise the provided emails clearly. For each email highlight:
- Core message
- Any action items or deadlines
- Important names or decisions
Do NOT invent any information not present in the emails.
"""

EXTRACT_SEND_SYSTEM = """You are an email field extractor.
From the user's instruction extract: recipient email (to), subject, body, and an array of absolute file paths for attachments.
Output ONLY valid JSON like:
{"to": "...", "subject": "...", "body": "...", "attachments": ["/path/to/file1.pdf"]}
"""

# ---------------------------------------------------------------------------
# Node: classify intent
# ---------------------------------------------------------------------------

def classify_intent(state: AgentState) -> AgentState:
    # --- PROGRAMMATIC OVERRIDE ---
    # If this is an automated run with a predefined config, skip LLM classification entirely
    if state.get("triage_config"):
        return {**state, "intent": "triage"}

    history = state["messages"]
    last_human = next(
        (m.content for m in reversed(history) if isinstance(m, HumanMessage)), ""
    )

    resp = llm.invoke([
        SystemMessage(content=INTENT_SYSTEM),
        HumanMessage(content=last_human),
    ])
    raw = resp.content.strip().lower()

    # Priority hybrid routing
    valid_intents = ["list_search", "read", "summarize", "send", "delete", "label", "todo", "triage", "converse"]
    
    intent = "converse"
    for v in valid_intents:
        if v in raw:
            intent = v
            break

    # Hard-coded quick intercepts for reliability
    lower_human = last_human.lower()
    if "triage" in lower_human or "daily" in lower_human or "sort out" in lower_human:
        intent = "triage"
    elif "task" in lower_human or "todo" in lower_human or "remind" in lower_human:
        intent = "todo"
    
    return {**state, "intent": intent}


# ---------------------------------------------------------------------------
# Node: Daily Triage (Autonomous Batch Job)
# ---------------------------------------------------------------------------

def handle_triage(state: AgentState) -> AgentState:
    history = state["messages"]
    last_human = next(
        (m.content for m in reversed(history) if isinstance(m, HumanMessage)), ""
    )
    
    # --- PROGRAMMATIC CONFIG CHECK ---
    config = state.get("triage_config")
    
    if not config:
        # Fallback to LLM extraction if running from the interactive chat CLI
        config_resp = llm.invoke([
            SystemMessage(content=TRIAGE_CONFIG_SYSTEM),
            HumanMessage(content=last_human),
        ])
        
        raw_json = config_resp.content.strip()
        raw_json = re.sub(r"```json\s*", "", raw_json)
        raw_json = re.sub(r"```\s*", "", raw_json).strip()
        
        try:
            config = json.loads(raw_json)
        except json.JSONDecodeError:
            return {**state, "response": f"⚠ Could not parse triage instructions. Please try rephrasing your keywords.\n[dim]Model Output:\n{raw_json}[/dim]"}
            
    keywords = config.get("keywords", [])
    if not keywords:
        return {**state, "response": "I couldn't detect any keywords to search for. Please specify keywords like 'keywords: AI, jobs, offers'."}

    timeframe = str(config.get("timeframe", "1d"))
    max_results = int(config.get("max_results", 20))
    use_semantic = config.get("use_semantic", False)

    # FIX: Define user_category OUTSIDE the if/else blocks so the summarizer can use it!
    user_category = ", ".join(keywords)

    if use_semantic:
        # --- SEMANTIC SEARCH FLOW (RAG) ---
        print(f"DEBUG: Using Semantic Search (RAG) for concepts: {keywords}")
        from vector.tools import semantic_search_emails
        concept = " ".join(keywords)
        rag_results = semantic_search_emails(query=concept, n_results=max_results)
        
        # Format the RAG results back into the dictionary structure the downstream code expects
        matched_emails = []
        for res in rag_results:
            matched_emails.append({
                "id": res["id"],
                "subject": res["subject"],
                "from_addr": res["from"],
                "date": res["date"]
            })
    else:
        # --- ORIGINAL GMAIL KEYWORD FLOW ---
        expansion_prompt = (
            f"You are a search query expansion tool. The user is interested in the category: '{user_category}'.\n"
            f"Generate 10 highly relevant terms to search for. "
            f"Include synonyms AND well-known brands, companies, or platforms associated with this category "
            f"(e.g., if the category is 'courses', you might include 'udemy, coursera, edx, class').\n"
            f"Output ONLY a comma-separated list of single words, nothing else."
        )
        
        expansion_resp = llm.invoke([HumanMessage(content=expansion_prompt)])
        expanded_words = [w.strip() for w in expansion_resp.content.replace('"', '').split(',') if w.strip()]
        
        all_search_terms = list(set([k.lower() for k in keywords] + [w.lower() for w in expanded_words]))
        keyword_str = " OR ".join([f'"{k}"' for k in all_search_terms])
        
        query = f"({keyword_str}) in:inbox newer_than:{timeframe}"
        print(f"DEBUG: Expanded Categorical & Brand Query: {query}")
        
        try:
            matched_emails = search_emails(query=query, max_results=max_results)
        except RuntimeError as e:
            return {**state, "response": f"⚠ Gmail Search Failed: {e}"}
            
    if not matched_emails:
        return {**state, "response": f"No emails found matching your criteria."}

    print(f"DEBUG: Found {len(matched_emails)} emails for triage.")
    
    # Fetch full bodies and combine
    parts = []
    for i, em in enumerate(matched_emails, 1):
        try:
            full = get_email_body(em["id"])
            body_text = full['body'][:500] # Truncate to save context window
            parts.append(
                f"<email_item index='{i}'>\n"
                f"From: {full['from_addr']}\n"
                f"Subject: {full['subject']}\n"
                f"Date: {full['date']}\n"
                f"Body: {body_text}\n"
                f"</email_item>"
            )
        except RuntimeError:
            continue

   
            
    combined_context = "\n\n".join(parts)
    
    # --- 3. SEMANTIC FILTERING & HTML SUMMARY ---
#     sem_prompt_instruction = (
#     f"STRICT RULES:\n"
#     f"- YOU ARE A DATA EXTRACTOR. Do NOT summarize newsletters.\n"
#     f"- ONLY output valid HTML.\n"
#     f"- DO NOT use Markdown, lists, or bolding outside of HTML tags.\n"
#     f"- STRUCTURE: <h2>Daily Digest</h2> followed by <h3>Subject</h3> then <ul><li>Point</li></ul>.\n"
#     f"- If the input contains no professional items, return ONLY: <h2>No results found.</h2>"
# )
    # --- 3. SEMANTIC FILTERING & JSON SUMMARY ---
   # --- 3. SEMANTIC FILTERING & JSON SUMMARY ---
    DIGEST_SUMMARY_SYSTEM = (
        "You are a strict JSON data extraction API. You do not converse. You ONLY output valid JSON.\n\n"
        "TASK:\n"
        "1. Extract uniquely relevant emails based on the Category.\n"
        "2. Write a 1-sentence summary for each email.\n"
        "3. DO NOT duplicate emails. List each unique email only once.\n\n"
        "REQUIRED JSON SCHEMA:\n"
        "{\n"
        '  "emails": [\n'
        '    {"id": "...", "subject": "...", "sender": "...", "summary": "..."}\n'
        "  ]\n"
        "}"
    )

    summary_resp = llm.bind(format="json").invoke([
        SystemMessage(content=DIGEST_SUMMARY_SYSTEM),
        HumanMessage(content=f"Category: '{user_category}'\n\nEMAILS TO PROCESS:\n{combined_context}\n\nRETURN ONLY JSON:"),
    ])

    # Clean the raw string
    raw_summary_json = summary_resp.content.strip()
    raw_summary_json = re.sub(r"^```(?:json)?\s*", "", raw_summary_json)
    raw_summary_json = re.sub(r"\s*```$", "", raw_summary_json)

    # Fallback to extract just the curly braces
    match = re.search(r'\{.*\}', raw_summary_json, re.DOTALL)
    if match:
        raw_summary_json = match.group(0)

    try:
        raw_emails = json.loads(raw_summary_json).get("emails", [])
        
        # --- DATA CLEANING & DEDUPLICATION ---
        seen_subjects = set()
        relevant_emails = []
        
        # Build a lookup from matched_emails by subject+sender for Gmail ID mapping
        matched_lookup = {}
        for me in matched_emails:
            sub_key = me["subject"].lower().strip()
            from_key = me["from_addr"].lower().strip()
            matched_lookup[(sub_key, from_key)] = me["id"]
            # Subject-only fallback in case sender format differs
            if sub_key not in matched_lookup:
                matched_lookup[sub_key] = me["id"]
        
        for em in raw_emails:
            # Handle model hallucinating wrong keys
            subject = em.get("subject") or em.get("title") or "No Subject"
            sender = em.get("sender") or em.get("from") or "Unknown Sender"
            summary = em.get("summary") or em.get("insight") or "No insight provided."
            
            clean_sub = subject.lower().strip()
            clean_sender = sender.lower().strip()
            # Deduplicate based on lowercase subject
            if clean_sub not in seen_subjects and clean_sub != "no subject":
                seen_subjects.add(clean_sub)
                email_id = matched_lookup.get((clean_sub, clean_sender)) or matched_lookup.get(clean_sub) or ""
                relevant_emails.append({
                    "subject": subject,
                    "sender": sender,
                    "summary": summary,
                    "id": email_id
                })
                
    except json.JSONDecodeError:
        print(f"DEBUG: Failed to parse LLM JSON output. Raw output:\n{summary_resp.content}")
        relevant_emails = []

    def render_digest(user_category: str, relevant_emails: list[dict], execution_log: list[str]) -> str:
        template = jinja_env.get_template("digest.html")
        return template.render(
            date=datetime.datetime.now().strftime("%B %d, %Y"),
            execution_log=execution_log,
            categories=[{"category_name": user_category, "emails": relevant_emails}],
        )
    
    # If the output doesn't contain at least one <h3>, it's likely a failure.
    
    # Add this snippet before calling send_email
    # if not digest_body.strip().startswith("<h"):
    #     # If the LLM went rogue and gave you plain text, force a wrapper
    #     digest_body = f"<h2>Summary</h2><p>{digest_body}</p>"


# Automate Execution (Labels, Tasks, Emailing self)
    execution_log_items = []
    execution_log_cli = f"✅ Triage complete! Processed {len(matched_emails)} emails.\n"

    # --- SMART LABEL LOGIC ---
    label_to_add = config.get("label_to_add")
    if label_to_add:
        lbl_lower = label_to_add.lower()
        add_labels = []
        remove_labels = []

        if "star" in lbl_lower: add_labels.append("STARRED")
        elif "imp" in lbl_lower: add_labels.append("IMPORTANT")
        elif "unread" in lbl_lower: add_labels.append("UNREAD")
        elif "read" in lbl_lower: remove_labels.append("UNREAD")
        else: add_labels.append(label_to_add.upper())

        success_count = 0
        if add_labels or remove_labels:
            for em in matched_emails:
                try:
                    modify_message_labels(em["id"], add_labels=add_labels if add_labels else None, remove_labels=remove_labels if remove_labels else None)
                    success_count += 1
                except Exception as e:
                    print(f"DEBUG: Failed to label email {em['id']}: {e}")

            label_action_str = f"Applied '{label_to_add}' logic to {success_count} emails"
            execution_log_items.append(f"🏷️ {label_action_str}")
            execution_log_cli += f"- 🏷️ {label_action_str}.\n"

    # Tasks
    if config.get("add_tasks"):
        current_time = datetime.datetime.now().isoformat() + "Z"
        task_prompt = f"Current time: {current_time}\n\nExtract tasks from these emails:\n\n{combined_context}"
        task_resp = llm.invoke([
            SystemMessage(content=EXTRACT_TODO_SYSTEM),
            HumanMessage(content=task_prompt),
        ])

        raw_task_json = task_resp.content.strip()
        raw_task_json = re.sub(r"```json\s*", "", raw_task_json)
        raw_task_json = re.sub(r"```\s*", "", raw_task_json).strip()

        try:
            tasks_list = json.loads(raw_task_json)
            if isinstance(tasks_list, list) and tasks_list:
                results = add_multiple_google_tasks(tasks_list)
                execution_log_items.append(f"📝 Added {len(results)} tasks to Google Tasks.")
                execution_log_cli += f"- 📝 Added {len(results)} tasks to Google Tasks.\n"
            else:
                execution_log_items.append("📝 No deadlines/tasks found to add.")
                execution_log_cli += "- 📝 No deadlines/tasks found to add.\n"
        except json.JSONDecodeError:
            execution_log_items.append("⚠ Failed to parse tasks from the LLM output.")
            execution_log_cli += "- ⚠ Failed to parse tasks from the LLM output.\n"

        


    email_body_html = render_digest(user_category, relevant_emails, execution_log_items)

    try:
        user_email = get_authenticated_email()
        if not user_email:
            raise Exception("Could not determine your authenticated email address.")
        subject_line = f"Daily Agent Digest: {', '.join(keywords)}"
        send_email(to=user_email, subject=subject_line, body=email_body_html, is_html=True)
        execution_log_cli += f"- 📧 Daily HTML Digest sent to {user_email}.\n"
    except Exception as e:
        execution_log_cli += f"- ⚠ Could not send digest email: {e}\n"

    final_response = f"{execution_log_cli}\n{'─' * 50}\n[bold]Digest Preview (HTML Source):[/bold]\n\n{email_body_html}"
    
    return {
        **state,
        "response": final_response,
        "messages": state["messages"] + [AIMessage(content=final_response)],
    }


# ---------------------------------------------------------------------------
# Other Nodes
# ---------------------------------------------------------------------------

def handle_list_search(state: AgentState) -> AgentState:
    history = state["messages"]
    last_human = next((m.content for m in reversed(history) if isinstance(m, HumanMessage)), "")
    q_resp = llm.invoke([SystemMessage(content=SEARCH_QUERY_SYSTEM), HumanMessage(content=last_human)])
    query = q_resp.content.strip().strip('"').strip("'")
    try:
        emails = search_emails(query=query, max_results=10)
    except RuntimeError as e:
        return {**state, "emails": [], "response": f"⚠ Could not fetch emails: {e}"}
    if not emails:
        return {**state, "emails": [], "response": "No emails matched your search."}
    lines = []
    for i, em in enumerate(emails, 1):
        lines.append(f"{i:>2}. [bold]{em['subject']}[/bold]\n    From : {em['from_addr']}\n    Date : {em['date']}\n    {em['snippet'][:120]}…")
    response = "\n".join(lines)
    return {**state, "emails": emails, "response": response, "messages": state["messages"] + [AIMessage(content=response)]}

def handle_read(state: AgentState) -> AgentState:
    emails = state.get("emails", [])
    history = state["messages"]
    last_human = next((m.content for m in reversed(history) if isinstance(m, HumanMessage)), "")
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
    
    att_text = ""
    if full.get("attachments"):
        att_text = "\n\n[bold]Attachments:[/bold]\n"
        for att in full["attachments"]:
            att_text += f"📎 {att['filename']} ({att['size']} bytes)\n"
            if "download" in last_human.lower():
                try:
                    from gmail.tools import download_attachment
                    path = download_attachment(emails[idx]["id"], att["attachmentId"], att["filename"], "./downloads")
                    att_text += f"   ✅ Downloaded to: {path}\n"
                except Exception as e:
                    att_text += f"   ⚠ Failed to download: {e}\n"

    response = f"[bold]Subject:[/bold] {full['subject']}\n[bold]From:[/bold]    {full['from_addr']}\n[bold]Date:[/bold]    {full['date']}\n{'─' * 60}\n{full['body']}{att_text}"
    return {**state, "response": response, "messages": state["messages"] + [AIMessage(content=response)]}

def handle_summarize(state: AgentState) -> AgentState:
    emails = state.get("emails", [])
    history = state["messages"]
    last_human = next((m.content for m in reversed(history) if isinstance(m, HumanMessage)), "")
    if not emails:
        return {**state, "response": "Please list or search emails first so I know what to summarise."}
    numbers = re.findall(r"\b(\d+)\b", last_human)
    count = int(numbers[0]) if numbers else len(emails)
    count = min(count, len(emails))
    parts = []
    for i, em in enumerate(emails[:count], 1):
        try:
            full = get_email_body(em["id"])
            body_text = full['body']
            if len(body_text) > 1500: body_text = body_text[:1500] + "\n... [Content Truncated] ..."
            parts.append(f"<email_item index='{i}'>\nFrom: {full['from_addr']}\nSubject: {full['subject']}\nDate: {full['date']}\nBody:\n{body_text}\n</email_item>")
        except RuntimeError:
            parts.append(f"<email_item index='{i}'>\nCould not fetch body for: {em['subject']}\n</email_item>")
    combined = "\n\n".join(parts)
    prompt_instruction = f"You are given {count} distinct emails wrapped in `<email_item>` tags. Provide a numbered list summarizing EACH email individually."
    summary_resp = llm.invoke([SystemMessage(content=SUMMARIZE_SYSTEM), HumanMessage(content=f"{prompt_instruction}\n\n{combined}")])
    summary = summary_resp.content.strip()
    return {**state, "response": summary, "messages": state["messages"] + [AIMessage(content=summary)]}


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
                msg_id = send_email(pending["to"], pending["subject"], pending["body"], pending.get("attachments", []))
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
    attachments = fields.get("attachments") or pending.get("attachments") or []

    # Ask for any missing field
    if not to:
        return {**state, "pending_send": {**pending, "subject": subject, "body": body, "attachments": attachments},
                "response": "Who should I send this to? (Please provide the recipient's email address)"}
    if not subject:
        return {**state, "pending_send": {**pending, "to": to, "body": body, "attachments": attachments},
                "response": "What should the subject line be?"}
    if not body:
        return {**state, "pending_send": {**pending, "to": to, "subject": subject, "attachments": attachments},
                "response": "What should the body of the email say?"}

    # All fields present — show preview and ask for confirmation
    att_preview = ""
    if attachments:
        att_preview = f"[bold]Attachments:[/bold] {', '.join(attachments)}\n"

    preview = (
        f"[bold]To:[/bold]      {to}\n"
        f"[bold]Subject:[/bold] {subject}\n"
        f"{att_preview}{'─' * 50}\n"
        f"{body}\n"
        f"{'─' * 50}\n"
        f"Reply [bold green]yes[/bold green] to send, or [bold red]no[/bold red] to cancel."
    )
    return {
        **state,
        "pending_send": {"to": to, "subject": subject, "body": body, "attachments": attachments, "awaiting_confirm": True},
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
# Node:     label email
# ---------------------------------------------------------------------------

def handle_label(state: AgentState) -> AgentState:
    history = state["messages"]
    last_human = next(
        (m.content for m in reversed(history) if isinstance(m, HumanMessage)), ""
    ).lower()
    emails = state.get("emails", [])

    if not emails:
        return {**state, "response": "Please list or search emails first so I know which one to label."}

    numbers = re.findall(r"\b(\d+)\b", last_human)
    idx = int(numbers[0]) - 1 if numbers else 0

    if idx < 0 or idx >= len(emails):
        return {**state, "response": f"Please specify a valid email number (1-{len(emails)})."}

    msg_id = emails[idx]["id"]
    
    add_labels = []
    remove_labels = []
    
    if "read" in last_human and ("mark" in last_human or "as" in last_human):
        remove_labels.append("UNREAD")
    if "unread" in last_human:
        add_labels.append("UNREAD")
    if "archive" in last_human:
        remove_labels.append("INBOX")
    if "star" in last_human and "unstar" not in last_human:
        add_labels.append("STARRED")
    if "unstar" in last_human:
        remove_labels.append("STARRED")
        
    if not add_labels and not remove_labels:
        return {**state, "response": "I couldn't understand what label action to take. (Try 'archive email 1' or 'mark email 2 as read')"}
        
    try:
        from gmail.tools import modify_message_labels
        modify_message_labels(msg_id, add_labels, remove_labels)
        
        # update the local emails state snippet if they changed UNREAD
        new_emails = list(emails)
        return {**state, "emails": new_emails, "response": "✅ Labels updated successfully."}
    except Exception as e:
        return {**state, "response": f"⚠ Failed to update labels: {e}"}

# ---------------------------------------------------------------------------
# Node: TO DO List
# ---------------------------------------------------------------------------


# 2. Update the handle_todo node to cleanly catch JSON failures:
def handle_todo(state: AgentState) -> AgentState:
    history = state["messages"]
    last_human = next(
        (m.content for m in reversed(history) if isinstance(m, HumanMessage)), ""
    )
    pending = state.get("pending_todo") or {}

    # --- STEP 1: Check if user is confirming a pending task ---
    confirmation_words = {"yes", "confirm", "do it", "go ahead", "yep", "y", "sure"}
    denial_words = {"no", "cancel", "abort", "don't", "stop", "nope", "n"}

    if pending and pending.get("awaiting_confirm"):
        lower = last_human.lower().strip()
        if any(w in lower for w in confirmation_words):
            try:
                # Add multiple tasks from the pending state
                from tasks.tools import add_multiple_google_tasks
                results = add_multiple_google_tasks(pending["tasks"])
                
                response_msg = f"✅ Added {len(results)} task(s) to Google Tasks:\n"
                for t in results:
                    response_msg += f"- [bold]{t.get('title')}[/bold]\n"
                
                return {
                    **state,
                    "pending_todo": None,
                    "response": response_msg,
                    "messages": state["messages"] + [AIMessage(content=response_msg)],
                }
            except RuntimeError as e:
                return {**state, "response": f"⚠ Failed to create task: {e}"}
        
        elif any(w in lower for w in denial_words):
            return {
                **state,
                "pending_todo": None,
                "response": "Cancelled. Task was not created.",
                "messages": state["messages"] + [AIMessage(content="Task creation cancelled.")],
            }

    # --- STEP 2: Extract task details ---
    last_ai = next(
        (m.content for m in reversed(history) if isinstance(m, AIMessage)), ""
    )

    import datetime
    current_time = datetime.datetime.now().isoformat() + "Z"
    context_prompt = (
        f"Current time: {current_time}\n\n"
        f"User Instruction: {last_human}\n\n"
        f"Email Context:\n{last_ai}"
    )

    extract_resp = llm.invoke([
        SystemMessage(content=EXTRACT_TODO_SYSTEM),
        HumanMessage(content=context_prompt),
    ])
    
    raw_json = extract_resp.content.strip()
    
    # Aggressively clean up common LLM formatting mistakes
    import re
    raw_json = re.sub(r"```json\s*", "", raw_json)
    raw_json = re.sub(r"```\s*", "", raw_json)
    raw_json = raw_json.strip()

    import json
    try:
        tasks_list = json.loads(raw_json)
        if not isinstance(tasks_list, list):
            tasks_list = [tasks_list] # Fallback if it outputs a single dict
    except json.JSONDecodeError as e:
        # Give visibility into what the model actually outputted to help debugging
        return {
            **state, 
            "response": f"⚠ Could not parse task details. The model failed to output raw JSON.\n\n[dim]Model Output:\n{raw_json}[/dim]"
        }

    if not tasks_list or not tasks_list[0].get("title"):
        return {**state, "response": "I couldn't figure out what the task should be. Please provide more details."}

    # --- STEP 3: Show preview and ask for confirmation ---
    preview = "[bold]Tasks to add:[/bold]\n\n"
    for t in tasks_list:
        title = t.get("title", "Untitled Task")
        notes = t.get("notes", "")
        due = t.get("due_date", "")
        due_str = f"Due: {due[:10]}" if due else "No due date"
        
        preview += f"🔹 [bold]{title}[/bold] ({due_str})\n"
        if notes:
            preview += f"   Notes: {notes}\n"
    
    preview += f"\n{'─' * 50}\nReply [bold green]yes[/bold green] to add these tasks, or [bold red]no[/bold red] to cancel."
    
    return {
        **state,
        "pending_todo": {
            "tasks": tasks_list,
            "awaiting_confirm": True
        },
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
            "summarise emails, delete emails, and send emails and create todo lists — all locally via Gmail API."
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
    if state.get("pending_send") and state["pending_send"].get("awaiting_confirm"):
        return "handle_send"
    if state.get("pending_delete") and state["pending_delete"].get("awaiting_confirm"):
        return "handle_delete"
        
    # <-- ADD THIS BLOCK -->
    if state.get("pending_todo") and state["pending_todo"].get("awaiting_confirm"):
        return "handle_todo"
        
    intent = state.get("intent", "converse")
    mapping = {
        "list_search": "handle_list_search",
        "read": "handle_read",
        "summarize": "handle_summarize",
        "send": "handle_send",
        "delete": "handle_delete",
        "label": "handle_label",
        "todo": "handle_todo",
        "triage": "handle_triage",  # <-- ADDED triage mapping
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
    g.add_node("handle_label", handle_label)
    g.add_node("handle_todo", handle_todo)
    g.add_node("handle_triage", handle_triage)  # <-- ADDED triage node
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
            "handle_label": "handle_label",
            "handle_todo": "handle_todo",
            "handle_triage": "handle_triage",  # <-- ADDED triage edge
            "handle_converse": "handle_converse",
        },
    )

    for node in ["handle_list_search", "handle_read", "handle_summarize",
                 "handle_send", "handle_delete", "handle_label", "handle_todo", "handle_triage", "handle_converse"]:  # <-- ADDED triage to END
        g.add_edge(node, END)

    return g.compile()


# Singleton compiled graph
graph = build_graph()