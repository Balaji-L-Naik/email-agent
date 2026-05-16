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

    from utils.parser import extract_attachments
    headers = extract_headers(full)
    payload = full.get("payload", {})
    body = extract_email_body(payload)
    attachments = extract_attachments(payload)

    return {
        "id": msg_id,
        "subject": headers["subject"],
        "from_addr": headers["from_addr"],
        "date": headers["date"],
        "body": body,
        "attachments": attachments
    }


import os
import mimetypes
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

def send_email(to: str, subject: str, body: str, attachments: list[str] = None) -> str:
    """
    Send an email via the authenticated Gmail account, with optional attachments.
    """
    service = _get_service()
    
    if not attachments:
        mime = MIMEText(body)
    else:
        mime = MIMEMultipart()
        mime.attach(MIMEText(body))
        
        for file_path in attachments:
            if not os.path.exists(file_path):
                continue
            ctype, encoding = mimetypes.guess_type(file_path)
            if ctype is None or encoding is not None:
                ctype = "application/octet-stream"
            maintype, subtype = ctype.split("/", 1)
            
            with open(file_path, "rb") as f:
                part = MIMEBase(maintype, subtype)
                part.set_payload(f.read())
            
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f'attachment; filename="{os.path.basename(file_path)}"')
            mime.attach(part)
            
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


def modify_message_labels(msg_id: str, add_labels: list[str] = None, remove_labels: list[str] = None) -> None:
    """
    Modify labels for a specific message (e.g. mark read, archive, star).
    """
    service = _get_service()
    body = {}
    if add_labels:
        body["addLabelIds"] = add_labels
    if remove_labels:
        body["removeLabelIds"] = remove_labels
    
    try:
        service.users().messages().modify(userId="me", id=msg_id, body=body).execute()
    except Exception as exc:
        raise RuntimeError(f"Gmail API error while modifying labels for {msg_id}: {exc}") from exc


def download_attachment(msg_id: str, attachment_id: str, filename: str, download_dir: str = ".") -> str:
    """
    Download an attachment from an email and save it to the specified directory.
    Returns the absolute path to the saved file.
    """
    service = _get_service()
    try:
        att = service.users().messages().attachments().get(
            userId="me", messageId=msg_id, id=attachment_id
        ).execute()
        
        file_data = base64.urlsafe_b64decode(att["data"])
        
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)
            
        filepath = os.path.join(download_dir, filename)
        with open(filepath, "wb") as f:
            f.write(file_data)
            
        return os.path.abspath(filepath)
    except Exception as exc:
        raise RuntimeError(f"Gmail API error while downloading attachment {filename}: {exc}") from exc


def read_text_attachment(msg_id: str, attachment_id: str, mime_type: str) -> str:
    """
    Extract text content from a text/csv/pdf attachment for summarisation without saving to disk permanently.
    """
    service = _get_service()
    try:
        att = service.users().messages().attachments().get(
            userId="me", messageId=msg_id, id=attachment_id
        ).execute()
        
        file_data = base64.urlsafe_b64decode(att["data"])
        
        if mime_type == "application/pdf":
            import io
            from pypdf import PdfReader
            pdf = PdfReader(io.BytesIO(file_data))
            text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
            return text
        else:
            # Assume plain text or CSV
            return file_data.decode("utf-8", errors="replace")
            
    except Exception as exc:
        raise RuntimeError(f"Gmail API error while reading attachment: {exc}") from exc