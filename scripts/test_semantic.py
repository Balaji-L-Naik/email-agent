"""
scripts/test_semantic.py
------------------------
A safe playground to test your new RAG (Semantic Search) tool.
Run this from the terminal to see what your Vector DB finds!
"""

import sys
import os

# Ensure we can import from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector.tools import semantic_search_emails

def main():
    print("🧠 Semantic Email Search Playground 🧠")
    print("Type 'exit' or 'quit' to stop.\n")
    
    while True:
        try:
            query = input("\n🔍 Enter a concept to search for (e.g., 'learning', 'finances'): ")
            
            if query.lower() in ['exit', 'quit', 'q']:
                print("Exiting playground...")
                break
                
            if not query.strip():
                continue
                
            print(f"Embedding query and searching Vector DB for top 5 matches...")
            results = semantic_search_emails(query=query, n_results=5)
            
            if not results:
                print("No results found. (Is your database populated?)")
                continue
                
            print(f"\n✅ Top {len(results)} Concept Matches:\n" + "-"*50)
            for i, res in enumerate(results, 1):
                print(f"{i}. [Score: {res['similarity_score']}] {res['subject']}")
                print(f"   From: {res['from']}")
                print(f"   Snippet: {res['snippet']}")
                print("-" * 50)
                
        except KeyboardInterrupt:
            print("\nExiting playground...")
            break

if __name__ == "__main__":
    main()