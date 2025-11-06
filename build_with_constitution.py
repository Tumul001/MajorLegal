"""
Build FAISS Index with Constitution, Acts, and Scraped Cases
Combines all legal data sources into one comprehensive RAG system
"""
import json
import os
from pathlib import Path
from rag_system.vector_store import IndianLegalVectorStore
from langchain_core.documents import Document
from tqdm import tqdm

def load_constitution_and_acts():
    """Load Constitution and Acts database"""
    
    # Try complete Constitution first, then fallback to comprehensive
    constitution_files = [
        "data/raw/constitution_complete_395_articles.json",
        "data/raw/constitution_and_acts_comprehensive.json"
    ]
    
    constitution_file = None
    for file_path in constitution_files:
        if os.path.exists(file_path):
            constitution_file = file_path
            break
    
    if not constitution_file:
        print(f"❌ Constitution database not found")
        return []
    
    print(f"📖 Loading from: {constitution_file}")
    
    with open(constitution_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    documents = []
    
    # Check if it's the complete Constitution file (395 articles)
    if 'articles' in data and len(data.get('articles', {})) > 100:
        print("✅ Using Complete Constitution (395 articles)")
        
        # Add all 395 articles
        for article_num, article_text in data['articles'].items():
            doc = Document(
                page_content=f"Article {article_num} of Constitution of India: {article_text}",
                metadata={
                    'source': 'Constitution of India',
                    'type': 'constitutional_article',
                    'article_number': article_num,
                    'document_type': 'constitution'
                }
            )
            documents.append(doc)
        
        print(f"   Added {len(data['articles'])} constitutional articles")
    
    # Also load Acts if available
    if 'acts' in data:
        for act in data['acts']:
            # Add act overview
            doc = Document(
                page_content=f"{act['name']}: {act['description']} (Total Sections: {act['total_sections']})",
                metadata={
                    'source': act['name'],
                    'type': 'act_overview',
                    'act_name': act['name'],
                    'total_sections': act['total_sections'],
                    'document_type': 'act'
                }
            )
            documents.append(doc)
            
            # Add key sections
            for section_num, section_text in act['key_sections'].items():
                doc = Document(
                    page_content=f"Section {section_num} of {act['name']}: {section_text}",
                    metadata={
                        'source': act['name'],
                        'type': 'act_section',
                        'act_name': act['name'],
                        'section_number': section_num,
                        'document_type': 'act'
                    }
                )
                documents.append(doc)
        
        print(f"   Added {len(data['acts'])} major Acts")
    
    # Fallback: Try old comprehensive format
    elif 'constitution' in data:
        constitution = data['constitution']
        
        # Add key articles
        for article_num, article_text in constitution['key_articles'].items():
            doc = Document(
                page_content=f"Article {article_num} of Constitution of India: {article_text}",
                metadata={
                    'source': 'Constitution of India',
                    'type': 'constitutional_article',
                    'article_number': article_num,
                    'document_type': 'constitution'
                }
            )
            documents.append(doc)
        
        # Add Constitution parts
        for part in constitution['parts']:
            doc = Document(
                page_content=f"Part {part['part']}: {part['title']} (Articles {part['articles']})",
                metadata={
                    'source': 'Constitution of India',
                    'type': 'constitutional_part',
                    'part_number': part['part'],
                    'document_type': 'constitution'
                }
            )
            documents.append(doc)
        
        print(f"   Using comprehensive Constitution database")
    
    print(f"✅ Loaded {len(documents)} Constitution & Acts documents")
    return documents

def load_scraped_cases():
    """Load scraped cases from bulk scraping"""
    # Try different file paths
    possible_files = [
        "data/raw/indiankanoon_massive_cases.json",
        "data/raw/indiankanoon_bulk_cases.json",
        "data/raw/indiankanoon_cases.json"
    ]
    
    cases_file = None
    for file_path in possible_files:
        if os.path.exists(file_path):
            cases_file = file_path
            break
    
    if not cases_file:
        print("⚠️ No scraped cases found")
        return []
    
    print(f"\n📂 Loading scraped cases from {cases_file}...")
    with open(cases_file, 'r', encoding='utf-8') as f:
        cases = json.load(f)
    
    print(f"✅ Loaded {len(cases)} cases")
    
    # Chunk each case
    documents = []
    chunk_size = 500  # words
    overlap = 50
    
    for case in tqdm(cases, desc="Chunking cases"):
        case_text = case.get('full_text', case.get('text', ''))
        if not case_text:
            continue
        
        # Split into word chunks
        words = case_text.split()
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            if len(chunk_text) < 50:  # Skip very small chunks
                continue
            
            doc = Document(
                page_content=chunk_text,
                metadata={
                    'case_name': case.get('case_name', 'Unknown'),
                    'citation': case.get('citation', 'N/A'),
                    'court': case.get('court', 'Unknown'),
                    'date': case.get('date', 'N/A'),
                    'url': case.get('url', ''),
                    'source': 'Indian Kanoon Case Law',
                    'document_type': 'case',
                    'type': 'case_chunk'
                }
            )
            documents.append(doc)
    
    print(f"✅ Created {len(documents)} case chunks")
    return documents

def main():
    print("="*80)
    print("BUILDING COMPREHENSIVE RAG SYSTEM")
    print("Constitution + Acts + Case Law")
    print("="*80)
    
    # Load all data sources
    print("\n📖 Loading Constitution and Acts...")
    constitution_docs = load_constitution_and_acts()
    
    print("\n⚖️ Loading scraped cases...")
    case_docs = load_scraped_cases()
    
    # Combine all documents
    all_documents = constitution_docs + case_docs
    
    print(f"\n📊 Total documents: {len(all_documents):,}")
    print(f"   - Constitution & Acts: {len(constitution_docs)}")
    print(f"   - Case Law chunks: {len(case_docs)}")
    
    if len(all_documents) == 0:
        print("\n❌ No documents to index!")
        return
    
    # Build FAISS index
    print("\n🧠 Building FAISS index...")
    vector_store = IndianLegalVectorStore()  # Will use settings from .env
    
    # Build index in batches
    initial_batch = 500
    batch_size = 200
    
    print(f"🔨 Creating initial index with {initial_batch} documents...")
    from langchain_community.vectorstores import FAISS
    
    vector_store.vectorstore = FAISS.from_documents(
        documents=all_documents[:initial_batch],
        embedding=vector_store.embeddings
    )
    
    # Add remaining documents
    if len(all_documents) > initial_batch:
        remaining = all_documents[initial_batch:]
        print(f"➕ Adding remaining {len(remaining):,} documents...")
        
        for i in tqdm(range(0, len(remaining), batch_size), desc="Building index"):
            batch = remaining[i:i+batch_size]
            vector_store.vectorstore.add_documents(batch)
    
    # Save index
    print("\n💾 Saving FAISS index...")
    vector_store.save_index()
    
    print("\n✅ INDEX BUILT SUCCESSFULLY!")
    print(f"📁 Saved to: {vector_store.index_path}")
    print(f"📊 Total vectors: {len(all_documents):,}")
    
    # Test the system
    print("\n🔍 Testing retrieval...")
    test_queries = [
        "Article 21 right to life",
        "Section 302 IPC murder",
        "bail provisions CrPC",
        "fundamental rights equality"
    ]
    
    for query in test_queries:
        results = vector_store.similarity_search(query, k=2)
        print(f"\n   Query: '{query}'")
        for idx, doc in enumerate(results, 1):
            source = doc.metadata.get('source', 'Unknown')
            doc_type = doc.metadata.get('document_type', 'unknown')
            preview = doc.page_content[:80] + "..."
            print(f"   {idx}. [{doc_type}] {source}")
            print(f"      {preview}")
    
    print("\n" + "="*80)
    print("✅ COMPREHENSIVE RAG SYSTEM READY!")
    print("="*80)
    print(f"\n🎯 Your system now includes:")
    print(f"   ✅ Constitution of India (395 articles)")
    print(f"   ✅ Major Acts (IPC, CrPC, Evidence, etc.)")
    print(f"   ✅ Indian case law ({len(case_docs):,} chunks)")
    print(f"\n🚀 Launch app: streamlit run app.py")

if __name__ == "__main__":
    main()
