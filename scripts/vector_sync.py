"""
scripts/vector_sync.py
----------------------
Run this standalone script to securely download your recent emails 
and embed them into a local vector database. 
"""

import os
import sys
import chromadb
from chromadb.utils import embedding_functions

# Ensure we can import from the parent directory (root)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your existing, working Gmail tools
from gmail.tools import search_emails, get_email_body

def sync_recent_emails(max_emails=50):
    print("🚀 Initializing Local Vector Database (ChromaDB)...")
    
    # Creates the 'email_vector_db' in your project root
    db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "email_vector_db")
    chroma_client = chromadb.PersistentClient(path=db_path)
    
    # We use a lightweight, fast, local embedding model
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    
    collection = chroma_client.get_or_create_collection(
        name="inbox",
        embedding_function=sentence_transformer_ef
    )
    
    print(f"📥 Fetching up to {max_emails} recent emails to sync...")
    recent_emails = search_emails(query="newer_than:7d", max_results=max_emails)
    
    if not recent_emails:
        print("No recent emails found to sync.")
        return

    documents = []
    metadatas = []
    ids = []

    print("🧠 Embedding emails (this may take a moment the first time)...")
    for em in recent_emails:
        msg_id = em["id"]
        
        # Skip if already in database (makes syncing fast on subsequent runs!)
        existing = collection.get(ids=[msg_id])
        if existing and existing['ids']:
            continue
            
        try:
            full = get_email_body(msg_id)
            # Combine subject and body for rich semantic meaning
            content = f"Subject: {full['subject']}\n\n{full['body'][:2000]}" 
            
            documents.append(content)
            metadatas.append({
                "from": full["from_addr"],
                "subject": full["subject"],
                "date": full["date"]
            })
            ids.append(msg_id)
            print(f"  -> Prepared: {full['subject'][:40]}...")
        except Exception as e:
            print(f"  -> ⚠ Skipped {msg_id}: {e}")

    if documents:
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"✅ Successfully embedded and saved {len(documents)} new emails!")
    else:
        print("✅ Database is already up to date. No new emails to sync.")

if __name__ == "__main__":
    sync_recent_emails(50)