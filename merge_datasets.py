"""
Simple script to merge downloaded HuggingFace datasets with existing data
"""
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def merge_datasets():
    """Merge all datasets"""
    
    data_dir = Path('data/raw')
    all_cases = []
    seen_urls = set()
    seen_titles = set()
    duplicates = 0
    
    print("\n" + "="*70)
    print("🔗 MERGING DATASETS")
    print("="*70 + "\n")
    
    # Files to merge
    files_to_merge = [
        'indiankanoon_massive_cases.json',  # Existing data (7,247 cases)
        'inlegal_train.json',                # HuggingFace training (10,995 cases)
        'inlegal_test.json',                 # HuggingFace test (4,501 cases)
    ]
    
    for filename in files_to_merge:
        file_path = data_dir / filename
        
        if not file_path.exists():
            logger.warning(f"File not found: {filename}")
            continue
        
        logger.info(f"Loading: {filename}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"  Found {len(data):,} records")
        
        # Process each case
        for case in data:
            # Generate dedup key
            url = case.get('url', '')
            title = case.get('title', case.get('text', ''))[:200].lower().strip()
            
            # Check for duplicates
            is_duplicate = False
            if url and url in seen_urls:
                is_duplicate = True
            elif title and title in seen_titles:
                is_duplicate = True
            
            if is_duplicate:
                duplicates += 1
                continue
            
            # Add to collection
            if url:
                seen_urls.add(url)
            if title:
                seen_titles.add(title)
            
            all_cases.append(case)
    
    # Save merged dataset
    output_file = data_dir / 'merged_final_dataset.json'
    
    logger.info(f"\nSaving merged dataset to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_cases, f, ensure_ascii=False, indent=2)
    
    # Print summary
    print("\n" + "="*70)
    print("📊 MERGE SUMMARY")
    print("="*70)
    print(f"Total unique cases: {len(all_cases):,}")
    print(f"Duplicates removed: {duplicates:,}")
    print(f"Output file: {output_file}")
    print("="*70 + "\n")
    
    return {
        'total_cases': len(all_cases),
        'duplicates_removed': duplicates,
        'output_file': str(output_file)
    }

if __name__ == "__main__":
    results = merge_datasets()
    
    print("✅ Merge complete!")
    print(f"\nYou now have {results['total_cases']:,} unique Indian legal cases")
    print("\nNext steps:")
    print("1. Run parallel scraper to add recent cases: python scrape_parallel.py")
    print("2. Rebuild FAISS index: python build_system.py")
