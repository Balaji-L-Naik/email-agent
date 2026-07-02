"""
vector/tools.py
---------------
This module handles querying the local ChromaDB vector database.
It is safely isolated from the LangGraph agent for testing.
"""

import os
import chromadb
from chromadb.utils import embedding_functions

def get_chroma_collection():
    """Initializes and returns the ChromaDB inbox collection."""
    # This ensures the DB is always created in the root of your project
    db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "email_vector_db")
    
    if not os.path.exists(db_path):
        raise RuntimeError(f"Database folder not found at {db_path}. Run scripts/vector_sync.py first!")
        
    chroma_client = chromadb.PersistentClient(path=db_path)
    
    # Must use the exact same embedding model used during sync!
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    return chroma_client.get_collection(
        name="inbox",
        embedding_function=sentence_transformer_ef
    )

def semantic_search_emails(query: str, n_results: int = 5):
    """
    Takes a natural language query, converts it to a vector, 
    and mathematically finds the closest matching emails.
    """
    try:
        collection = get_chroma_collection()
        
        # ChromaDB handles the embedding of the query string automatically
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        formatted_results = []
        
        if results and results['ids'] and results['ids'][0]:
            # Chroma returns lists of lists, so we access index 0
            ids = results['ids'][0]
            metadatas = results['metadatas'][0]
            distances = results['distances'][0] # Lower distance = closer meaning
            documents = results['documents'][0]
            
            for i in range(len(ids)):
                formatted_results.append({
                    "id": ids[i],
                    "subject": metadatas[i].get("subject", "No Subject"),
                    "from": metadatas[i].get("from", "Unknown Sender"),
                    "date": metadatas[i].get("date", "Unknown Date"),
                    "snippet": documents[i][:200].replace('\n', ' ') + "...",
                    "similarity_score": round(1.0 - distances[i], 4) # Rough conversion to a score
                })
                
        return formatted_results
        
    except Exception as e:
        print(f"Error during semantic search: {e}")
        return []