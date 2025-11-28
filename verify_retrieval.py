from rag_system.legal_rag import ProductionLegalRAGSystem
import os
from dotenv import load_dotenv

load_dotenv()

def verify():
    print("🔄 Initializing RAG System...")
    rag = ProductionLegalRAGSystem()
    
    print("\n🧪 Test 1: Search by Section (302 IPC)")
    results = rag.search_by_section("302", "IPC", k=5)
    for i, res in enumerate(results):
        print(f"\nResult {i+1}:")
        print(res[:500] + "..." if len(res) > 500 else res)

    print("\n🧪 Test 2: Natural Language Query ('punishment for murder')")
    docs = rag.retrieve_documents("What is the punishment for murder under IPC?", k=10)
    for i, doc in enumerate(docs):
        meta = doc.metadata
        print(f"\nRank {i+1}: {meta.get('case_name', 'Unknown')}")
        print(f"  Source: {meta.get('source', 'Unknown')}")
        print(f"  Score: {meta.get('smart_score', 0):.4f} (Base: {meta.get('original_score', 0):.4f})")
        print(f"  Court: {meta.get('court', 'Unknown')}")
        print(f"  Date: {meta.get('date', 'Unknown')}")

if __name__ == "__main__":
    verify()
