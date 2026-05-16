import base64


def extract_email_body(payload: dict) -> str:
    """Recursively extract plain-text body from a Gmail message payload."""
    if "parts" in payload:
        for part in payload["parts"]:
            mime_type = part.get("mimeType", "")
            if mime_type == "text/plain":
                data = part["body"].get("data")
                if data:
                    return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
            # Recurse into nested multipart
            if mime_type.startswith("multipart"):
                result = extract_email_body(part)
                if result:
                    return result

    body_data = payload.get("body", {}).get("data")
    if body_data:
        return base64.urlsafe_b64decode(body_data).decode("utf-8", errors="replace")

    return "(No readable content)"


def extract_headers(message: dict) -> dict:
    """Return a dict with subject, from_addr, and date from a Gmail message."""
    headers = message.get("payload", {}).get("headers", [])
    result = {"subject": "(No Subject)", "from_addr": "Unknown", "date": "Unknown"}
    for h in headers:
        name = h.get("name", "").lower()
        if name == "subject":
            result["subject"] = h["value"]
        elif name == "from":
            result["from_addr"] = h["value"]
        elif name == "date":
            result["date"] = h["value"]
    return result