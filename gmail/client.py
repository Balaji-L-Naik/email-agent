from googleapiclient.discovery import build
from gmail.auth import authenticate_gmail


def get_gmail_service():
    creds = authenticate_gmail()

    service = build(
        "gmail",
        "v1",
        credentials=creds
    )

    return service



def get_authenticated_email() -> str:
    """
    Fetch the authenticated user's email address dynamically.
    """
    try:
        service = get_gmail_service()
        profile = service.users().getProfile(userId='me').execute()
        return profile.get('emailAddress', '')
    except Exception as e:
        print(f"DEBUG: Could not fetch authenticated user email: {e}")
        return ""