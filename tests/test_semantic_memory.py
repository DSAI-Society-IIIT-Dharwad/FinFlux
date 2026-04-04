import chromadb
import os
from pathlib import Path

# Paths - Ensure absolute paths for Chroma
CHROMA_PATH = Path(r"c:\Users\Radhakrishna\Desktop\finflux\data\chroma_db")

def test_memory():
    if not CHROMA_PATH.exists():
        print(f"ERROR: ChromaDB directory not found at {CHROMA_PATH}")
        return

    print("--- 1. Testing Connections & Isolation ---")
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection(name="financial_memories")
    count = collection.count()
    print(f"Total global memories: {count}")

    # Test Users
    users = ["guest_001", "guest_002"]
    
    for user in users:
        print(f"\n--- Testing Isolation for: {user} ---")
        
        # Intent query
        query = "I have debt pressure"
        results = collection.query(
            query_texts=[query],
            n_results=5,
            where={"user_id": user} # This is the multi-tenant guardrail
        )
        
        user_matches = results['documents'][0]
        print(f"Results for query '{query}': {len(user_matches)}")
        
        for i, doc in enumerate(user_matches):
            dist = results['distances'][0][i]
            score = round(1 - dist, 2)
            meta = results['metadatas'][0][i]
            
            # CRITICAL CHECK: Verify user_id matches the query filter
            if meta['user_id'] != user:
                print(f"  [SECURITY FAILURE] Data bleed from {meta['user_id']} into {user}!")
            else:
                print(f"  [{i+1}] Relevance: {score} | Snapshot: {doc[:60]}...")

if __name__ == "__main__":
    test_memory()
