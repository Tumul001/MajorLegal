import os
import json
import re
import zipfile
from typing import List, Dict, Optional
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset, Dataset
from huggingface_hub import hf_hub_download

class ILDCDataIngestion:
    """Ingest and process ILDC dataset for legal RAG system"""
    
    def __init__(self, output_dir: str = "data/processed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = self.output_dir / "ildc_cases.json"
    
    def load_ildc_dataset(self, split: str = "train[:1000]", cache_dir: str = None, dataset_name: str = "nyaya-anumana"):
        """
        Load Indian legal dataset from HuggingFace
        
        Args:
            split: Dataset split to load (e.g., "train[:1000]" for first 1000 cases)
            cache_dir: Optional cache directory for dataset
            dataset_name: Dataset to use ('nyaya-anumana', 'nyayarag', 'ildc-expert')
        
        Returns:
            Dataset object or List[Dict]
        """
        print(f"📚 Loading {dataset_name} dataset: {split}")
        
        # Special handling for NyayaRAG (Custom Zip + JSON structure)
        if dataset_name == "nyayarag":
            try:
                print("ℹ️  Downloading NyayaRAG zip file (this may take a while)...")
                zip_path = hf_hub_download(
                    repo_id="L-NLProc/NyayaRAG", 
                    filename="1.Base Dataset.zip", 
                    repo_type="dataset",
                    cache_dir=cache_dir
                )
                
                print("📦 Extracting JSON from zip...")
                with zipfile.ZipFile(zip_path, 'r') as z:
                    with z.open('Base Dataset/SCI_judgements_56k.json') as f:
                        data = json.load(f)
                
                print(f"✅ Loaded {len(data)} cases from NyayaRAG")
                
                # Handle split (simple slicing for list)
                if "[:" in split:
                    limit = int(split.split("[:")[1].replace("]", ""))
                    data = data[:limit]
                    print(f"✂️  Applied split '{split}': keeping {len(data)} cases")
                
                return data
            except Exception as e:
                print(f"❌ Error loading NyayaRAG: {e}")
                raise

        # Map other dataset names to HuggingFace paths
        dataset_map = {
            "nyaya-anumana": "NyayaAnumana/nyaya-anumana",
            "ildc-expert": "anuragiiser/ILDC_expert"
        }
        
        dataset_path = dataset_map.get(dataset_name, dataset_name)
        
        try:
            dataset = load_dataset(
                dataset_path,
                split=split,
                cache_dir=cache_dir
            )
            print(f"✅ Loaded {len(dataset)} cases from {dataset_name}")
            return dataset
        except Exception as e:
            print(f"❌ Error loading {dataset_name} dataset: {e}")
            print(f"💡 Tip: Ensure you have internet connection and datasets library is installed")
            raise
    
    def extract_metadata(self, case: Dict) -> Dict:
        """
        Extract metadata from case dictionary
        
        Handles multiple schemas:
        1. NyayaRAG: document_id, full_text
        2. ILDC_expert: Case ID, Case Description
        3. Generic: text, case_no, etc.
        """
        # 1. NyayaRAG Schema
        if 'document_id' in case:
            case_id = str(case.get('document_id', 'Unknown'))
            case_text = case.get('full_text', '')
            source = 'NyayaRAG'
        # 2. ILDC Expert Schema
        elif 'Case ID' in case:
            case_id = str(case.get('Case ID', 'Unknown'))
            case_text = case.get('Case Description', '')
            source = 'ILDC (HuggingFace)'
        # 3. Generic/Fallback
        else:
            case_id = str(case.get('case_no', case.get('id', 'Unknown')))
            case_text = case.get('text', '')
            source = 'Unknown'

        # Generate case name
        # Try to extract from first line of text
        first_line = case_text.split('\n')[0] if case_text else f"Case {case_id}"
        case_name = first_line[:100].strip() if len(first_line) > 100 else first_line.strip()
        if not case_name:
            case_name = f"Case {case_id}"

        metadata = {
            'case_name': case_name,
            'citation': f"{source}-{case_id}",
            'court': 'Supreme Court of India',
            'date': 'Unknown',
            'judges': '',
            'appellant': '',
            'respondent': '',
            'case_status': '',
            'source': source,
            'url': '',
            'acts_mentioned': '',
            'sections': '',
            'case_id': case_id
        }
        
        return metadata
    
    def process_dataset(self, dataset, chunk_size: int = 500) -> List[Dict]:
        """
        Process ILDC dataset into LangChain-compatible format
        
        Returns:
            List of processed document dictionaries
        """
        print(f"🔄 Processing {len(dataset)} cases...")
        processed_docs = []
        
        for case in tqdm(dataset, desc="Processing cases"):
            # Extract case text (handle multiple possible field names)
            case_text = case.get('full_text') or case.get('Case Description') or case.get('text') or ''
            
            # Skip empty cases
            if not case_text or len(case_text.strip()) < 100:
                continue
            
            # Extract metadata
            metadata = self.extract_metadata(case)
            
            # Chunk the text
            chunks = self.chunk_text(case_text, chunk_size=chunk_size)
            
            # Create document for each chunk
            for idx, chunk in enumerate(chunks):
                doc = {
                    'text': chunk,
                    'metadata': {
                        **metadata,
                        'chunk_index': idx,
                        'total_chunks': len(chunks)
                    }
                }
                processed_docs.append(doc)
        
        print(f"✅ Processed into {len(processed_docs)} document chunks")
        return processed_docs
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Split case text into overlapping chunks for better retrieval
        
        Args:
            text: Full case text
            chunk_size: Target words per chunk
            overlap: Overlapping words between chunks
        
        Returns:
            List of text chunks
        """
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if len(chunk.split()) >= 50:  # Minimum chunk size
                chunks.append(chunk)
        
        return chunks if chunks else [text]  # Fallback to full text if chunking fails
    
    def save_processed_data(self, processed_docs: List[Dict]):
        """Save processed documents to JSON"""
        print(f"💾 Saving to {self.output_file}...")
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_docs, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Saved {len(processed_docs)} documents")
        print(f"📊 File size: {self.output_file.stat().st_size / (1024*1024):.2f} MB")
    
    def run_pipeline(self, split: str = "train[:1000]", chunk_size: int = 500, dataset_name: str = "nyaya-anumana"):
        """
        Execute full ingestion pipeline
        
        Args:
            split: Dataset split (default: first 1000 training cases)
            chunk_size: Words per chunk
            dataset_name: Dataset to use ('nyaya-anumana' or 'ildc-expert')
        """
        # Update output filename based on dataset
        dataset_filename = dataset_name.replace("-", "_")
        self.output_file = self.output_dir / f"{dataset_filename}_cases.json"
        
        print("=" * 60)
        print(f"🚀 {dataset_name.upper()} Data Ingestion Pipeline")
        print("=" * 60)
        
        # Load dataset
        dataset = self.load_ildc_dataset(split=split, dataset_name=dataset_name)
        
        # Process dataset
        processed_docs = self.process_dataset(dataset, chunk_size=chunk_size)
        
        # Save to file
        self.save_processed_data(processed_docs)
        
        print("\n" + "=" * 60)
        print("✅ Pipeline Complete!")
        print("=" * 60)
        print(f"📁 Output: {self.output_file}")
        print(f"📊 Total documents: {len(processed_docs)}")
        print("\n💡 Next Steps:")
        print("   1. Run: python rebuild_vector_store.py")
        print("   2. This will rebuild FAISS index with InLegalBERT embeddings")
        print("=" * 60)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest ILDC dataset for legal RAG")
    parser.add_argument(
        '--split',
        type=str,
        default='train[:1000]',
        help='Dataset split to load (e.g., train[:1000], train[:5000], train)'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=500,
        help='Words per chunk (default: 500)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed',
        help='Output directory for processed data'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='nyayarag',
        choices=['nyayarag', 'nyaya-anumana', 'ildc-expert'],
        help='Dataset to use: nyayarag (56k SC cases), nyaya-anumana (702k), or ildc-expert (54)'
    )
    
    args = parser.parse_args()
    
    # Run ingestion
    ingestion = ILDCDataIngestion(output_dir=args.output_dir)
    ingestion.run_pipeline(split=args.split, chunk_size=args.chunk_size, dataset_name=args.dataset)


if __name__ == "__main__":
    main()
