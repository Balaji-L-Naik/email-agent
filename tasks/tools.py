"""
tasks/tools.py
--------------
Google Tasks API wrappers.
"""

from googleapiclient.discovery import build
from gmail.auth import authenticate_gmail

_tasks_service = None

def get_tasks_service():
    global _tasks_service
    if not _tasks_service:
        creds = authenticate_gmail()
        _tasks_service = build('tasks', 'v1', credentials=creds)
    return _tasks_service

def add_google_task(title: str, notes: str = "", due_date: str = None) -> dict:
    """
    Creates a single task in the user's default task list (@default).
    due_date must be an RFC 3339 timestamp (e.g., '2026-06-25T00:00:00.000Z')
    """
    service = get_tasks_service()
    
    task_body = {
        "title": title,
        "notes": notes
    }
    
    if due_date:
        task_body["due"] = due_date

    try:
        result = service.tasks().insert(tasklist='@default', body=task_body).execute()
        return result
    except Exception as exc:
        raise RuntimeError(f"Tasks API error while creating task: {exc}") from exc

def add_multiple_google_tasks(tasks: list[dict]) -> list[dict]:
    """
    Creates multiple tasks in the user's default task list (@default).
    'tasks' should be a list of dictionaries, each containing 'title', 'notes', and optionally 'due_date'.
    """
    service = get_tasks_service()
    results = []
    
    for task_data in tasks:
        task_body = {
            "title": task_data.get("title"),
            "notes": task_data.get("notes", "")
        }
        if task_data.get("due_date"):
            task_body["due"] = task_data.get("due_date")

        try:
            result = service.tasks().insert(tasklist='@default', body=task_body).execute()
            results.append(result)
        except Exception as exc:
            print(f"Tasks API error while creating task '{task_data.get('title')}': {exc}")
            
    return results