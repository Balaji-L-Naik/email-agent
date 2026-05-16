"""
gmail/tools.py
--------------
Clean Gmail API wrappers used by the LangGraph agent.
All functions return plain Python dicts / strings — no LLM calls here.
"""

import base64
import re
from email.mime.text import MIMEText

from gmail.client import get_gmail_service
from utils.parser import extract_email_body, extract_headers

# Initialise once at import time
_service = get_gmail_service()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_service():
    return _service


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search_emails(query: str = "", max_results: int = 10) -> list[dict]:
    """
    Search / list emails using a Gmail query string.

    Returns a list of dicts:
      { id, subject, from_addr, date, snippet }
    """
    service = _get_service()
    params = {"userId": "me", "maxResults": max_results}
    if query:
        params["q"] = query
    else:
        params["labelIds"] = ["INBOX"]

    try:
        results = service.users().messages().list(**params).execute()
    except Exception as exc:
        raise RuntimeError(f"Gmail API error while listing messages: {exc}") from exc

    messages = results.get("messages", [])
    output = []

    for msg in messages:
        try:
            full = service.users().messages().get(
                userId="me", id=msg["id"], format="metadata",
                metadataHeaders=["Subject", "From", "Date"]
            ).execute()
            headers = extract_headers(full)
            output.append({
                "id": msg["id"],
                "subject": headers["subject"],
                "from_addr": headers["from_addr"],
                "date": headers["date"],
                "snippet": full.get("snippet", ""),
            })
        except Exception:
            # Skip malformed messages silently
            continue

    return output


def get_email_body(msg_id: str) -> dict:
    """
    Fetch the full plain-text body of a single email by ID.

    Returns { id, subject, from_addr, date, body }
    """
    service = _get_service()
    try:
        full = service.users().messages().get(
            userId="me", id=msg_id, format="full"
        ).execute()
    except Exception as exc:
        raise RuntimeError(f"Gmail API error fetching message {msg_id}: {exc}") from exc

    headers = extract_headers(full)
    body = extract_email_body(full.get("payload", {}))

    return {
        "id": msg_id,
        "subject": headers["subject"],
        "from_addr": headers["from_addr"],
        "date": headers["date"],
        "body": body,
    }


def send_email(to: str, subject: str, body: str) -> str:
    """
    Send an email via the authenticated Gmail account.

    Returns the sent message ID on success.
    Raises RuntimeError on failure.
    """
    service = _get_service()
    mime = MIMEText(body)
    mime["to"] = to
    mime["subject"] = subject
    raw = base64.urlsafe_b64encode(mime.as_bytes()).decode()

    try:
        sent = service.users().messages().send(
            userId="me", body={"raw": raw}
        ).execute()
    except Exception as exc:
        raise RuntimeError(f"Gmail API error while sending: {exc}") from exc

    return sent["id"]


def trash_email(msg_id: str) -> None:
    """
    Move a single email to Trash by message ID.
    Requires gmail.modify scope.
    Raises RuntimeError on failure.
    """
    service = _get_service()
    try:
        service.users().messages().trash(userId="me", id=msg_id).execute()
    except Exception as exc:
        raise RuntimeError(f"Gmail API error while trashing {msg_id}: {exc}") from exc


def trash_emails_by_query(query: str, max_results: int = 50) -> dict:
    """
    Trash all emails matching a Gmail search query.

    Returns { trashed: int, failed: int, subjects: list[str] }
    """
    emails = search_emails(query=query, max_results=max_results)
    trashed, failed, subjects = 0, 0, []

    for em in emails:
        try:
            trash_email(em["id"])
            trashed += 1
            subjects.append(em["subject"])
        except RuntimeError:
            failed += 1

    return {"trashed": trashed, "failed": failed, "subjects": subjects}