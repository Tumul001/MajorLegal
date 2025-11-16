"""
Rebuild FAISS vector store with merged dataset
"""
import sys
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Set

# Add rag_system to path
sys.path.insert(0, str(Path(__file__).parent))

from rag_system.vector_store import IndianLegalVectorStore

def convert_raw_to_processed(raw_data_file: str, output_file: str):
    """Convert raw merged dataset to processed format expected by vector store"""
    
    print("📄 Loading raw dataset...")
    with open(raw_data_file, 'r', encoding='utf-8') as f:
        raw_cases = json.load(f)
    
    print(f"✅ Loaded {len(raw_cases):,} cases")
    print("🔄 Converting to processed format with deduplication...")
    
    processed_docs = []
    seen_hashes: Set[str] = set()  # Track chunk hashes for deduplication
    duplicate_count = 0
    
    for case in raw_cases:
        # Extract text content - handle different data structures
        text = None
        
        # InLegalNER format: nested 'data' -> 'text'
        if isinstance(case, dict) and 'data' in case:
            text = case['data'].get('text', '')
            # Extract metadata from InLegalNER
            meta = case.get('meta', {})
            source_info = meta.get('source', '')
        # IndianKanoon format
        elif isinstance(case, dict):
            text = (case.get('judgment_text') or 
                    case.get('text') or 
                    case.get('content') or 
                    case.get('judgement', ''))
            source_info = case.get('url', '')
        else:
            text = str(case)
            source_info = ''
        
        if not text or len(text) < 100:
            continue
        
        # Split long texts into chunks
        chunk_size = 2000
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        
        for idx, chunk in enumerate(chunks):
            # Skip very short chunks
            if len(chunk.strip()) < 100:
                continue
            
            # DEDUPLICATION: Hash the normalized chunk text
            normalized_chunk = ' '.join(chunk.lower().split())  # Normalize whitespace + lowercase
            chunk_hash = hashlib.sha256(normalized_chunk.encode('utf-8')).hexdigest()
            
            # Skip if we've seen this exact chunk before
            if chunk_hash in seen_hashes:
                duplicate_count += 1
                continue
            
            seen_hashes.add(chunk_hash)
            
            # Extract case name from source_info if available
            case_name = 'Unknown'
            if 'data' in case and isinstance(case, dict):
                # For InLegalNER, try to extract from text or source
                case_name = source_info.split('/')[-2] if '/' in source_info else 'Indian Legal Case'
            else:
                case_name = case.get('title', case.get('case_name', 'Unknown'))
            
            processed_doc = {
                'text': chunk,
                'metadata': {
                    'case_name': case_name,
                    'citation': case.get('citation', case.get('cite', 'N/A')),
                    'court': case.get('court', 'Indian Court'),
                    'date': case.get('date', case.get('year', 'Unknown')),
                    'judges': case.get('judges', []),
                    'acts_mentioned': case.get('acts', []),
                    'sections': case.get('sections', []),
                    'url': source_info if source_info else case.get('url', ''),
                    'chunk_index': idx,
                    'source': 'Indian Legal Dataset'
                }
            }
            processed_docs.append(processed_doc)
    
    print(f"✅ Created {len(processed_docs):,} unique document chunks")
    print(f"🗑️  Removed {duplicate_count:,} duplicate chunks ({duplicate_count/(len(processed_docs)+duplicate_count)*100:.1f}% deduplication)")
    
    # Ensure output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"💾 Saving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_docs, f, ensure_ascii=False, indent=2)
    
    print("✅ Conversion complete!")
    return len(processed_docs)

def main():
    print("\n" + "="*70)
    print("🔨 REBUILDING VECTOR STORE")
    print("="*70 + "\n")
    
    # Path to merged dataset
    raw_file = Path("data/raw/merged_final_dataset.json")
    processed_file = Path("data/processed/processed_cases.json")
    
    if not raw_file.exists():
        print(f"❌ Error: {raw_file} not found")
        print("Run: python merge_datasets.py first")
        return
    
    # Step 1: Convert raw to processed format
    print("STEP 1: Converting raw data to processed format\n")
    num_chunks = convert_raw_to_processed(str(raw_file), str(processed_file))
    
    # Step 2: Build vector store
    print("\n" + "-"*70)
    print("STEP 2: Building FAISS index")
    print("-"*70 + "\n")
    
    vector_store = IndianLegalVectorStore()
    vector_store.build_from_processed_docs(str(processed_file), batch_size=32)
    
    print("\n" + "="*70)
    print("✅ VECTOR STORE REBUILT SUCCESSFULLY")
    print("="*70)
    print(f"\nIndexed {num_chunks:,} document chunks from 15,622 cases")
    print("\nYou can now run the app:")
    print("  streamlit run app.py")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
