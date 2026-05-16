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