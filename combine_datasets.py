# combine_datasets.py
"""
Combine various processed JSON sources into a single file for vector store rebuilding.
Sources:
- data/processed/nyayarag_cases.json (56k Supreme Court cases)
- data/processed/processed_scraped_cases.json (other scraped cases)
- data/raw/constitution_complete_395_articles.json (full Indian Constitution)
- data/processed/statutes.json (IPC, CrPC, IEA, HMA, ICA)
The resulting file is written to data/processed/combined_cases.json.

OPTIMIZED: Uses streaming writes to avoid MemoryError.
"""
import json
import random
import gc
from pathlib import Path

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    base_dir = Path(__file__).parent
    processed_dir = base_dir / "data" / "processed"
    raw_dir = base_dir / "data" / "raw"
    output_path = processed_dir / "combined_cases.json"

    LANDMARK_KEYWORDS = [
        "Mohori Bibee", "Lalita Kumari", "Gurbaksh Singh Sibbia", 
        "Kesavananda Bharati", "Maneka Gandhi", "Shah Bano",
        "Vishaka", "Puttaswamy"
    ]

    print(f"Writing combined dataset to {output_path} (streaming mode)")
    
    with open(output_path, "w", encoding="utf-8") as out_f:
        out_f.write("[\n")
        first_item = True
        
        def write_batch(items):
            nonlocal first_item
            for item in items:
                if not first_item:
                    out_f.write(",\n")
                json.dump(item, out_f, ensure_ascii=False)
                first_item = False
            out_f.flush()

        # 1. NyayaRAG cases (keep only 10% BUT include landmarks)
        nyaya_path = processed_dir / "nyayarag_cases.json"
        if nyaya_path.exists():
            print(f"Loading NyayaRAG cases from {nyaya_path}")
            try:
                data = load_json(str(nyaya_path))
                landmarks = []
                others = []
                
                for case in data:
                    # Check if text or title contains landmark keywords
                    text_content = str(case.get("text", "")) + str(case.get("metadata", ""))
                    if any(k.lower() in text_content.lower() for k in LANDMARK_KEYWORDS):
                        landmarks.append(case)
                    else:
                        others.append(case)
                
                print(f"  Found {len(landmarks)} landmark cases in NyayaRAG (whitelisted)")
                
                # Keep 10% of OTHERS
                keep_ratio = 0.10
                keep_count = max(1, int(len(others) * keep_ratio))
                sampled_others = random.sample(others, keep_count)
                
                print(f"  Sampled {len(sampled_others)} from {len(others)} other cases")
                
                # Combine
                batch = landmarks + sampled_others
                
                for case in batch:
                    case.setdefault("metadata", {})
                    case["metadata"]["source"] = "NyayaRAG"
                
                write_batch(batch)
                
                # Free memory
                del data, landmarks, others, sampled_others, batch
                gc.collect()
                
            except Exception as e:
                print(f"❌ Error processing NyayaRAG: {e}")
        else:
            print("⚠️ NyayaRAG file not found; skipping.")

        # 2. Other scraped cases (keep only 50,000 BUT include landmarks)
        scraped_path = processed_dir / "processed_scraped_cases.json"
        if scraped_path.exists():
            print(f"Loading scraped cases from {scraped_path}")
            try:
                data = load_json(str(scraped_path))
                
                landmarks = []
                others = []
                
                for case in data:
                    text_content = str(case.get("text", "")) + str(case.get("metadata", ""))
                    if any(k.lower() in text_content.lower() for k in LANDMARK_KEYWORDS):
                        landmarks.append(case)
                    else:
                        others.append(case)
                        
                print(f"  Found {len(landmarks)} landmark cases in Scraped data (whitelisted)")

                # Keep only 50,000 of OTHERS
                max_scraped = 50_000
                if len(others) > max_scraped:
                    sampled_others = random.sample(others, max_scraped)
                    print(f"  Sampled {max_scraped:,} of {len(others):,} scraped cases")
                else:
                    sampled_others = others
                    
                batch = landmarks + sampled_others
                
                for case in batch:
                    case.setdefault("metadata", {})
                    case["metadata"]["source"] = "scraped"
                    
                    # FIX: Improve metadata if title is generic
                    title = case["metadata"].get("case_name", "")
                    if not title or title.strip().lower() in ["full document", "unknown", ""]:
                        # Extract first 100 chars of text as title
                        text_snippet = case.get("text", "")[:100].replace("\n", " ").strip()
                        if text_snippet:
                            case["metadata"]["case_name"] = text_snippet + "..."
                            
                write_batch(batch)
                
                # Free memory
                del data, landmarks, others, sampled_others, batch
                gc.collect()
                
            except Exception as e:
                print(f"❌ Error processing scraped cases: {e}")
        else:
            print("⚠️ Scraped cases file not found; skipping.")

        # 3. Constitution articles
        const_path = raw_dir / "constitution_complete_395_articles.json"
        if const_path.exists():
            print(f"Loading Constitution articles from {const_path}")
            try:
                const_data = load_json(str(const_path))
                articles = const_data.get("articles", {})
                batch = []
                for article_number, article_text in articles.items():
                    text = article_text
                    metadata = {
                        "case_name": f"Constitution Article {article_number}",
                        "citation": f"Constitution-{article_number}",
                        "court": "Constitution of India",
                        "date": "1949",
                        "source": "constitution",
                        "url": "",
                        "acts_mentioned": "",
                        "sections": "",
                        "case_id": f"constitution-{article_number}"
                    }
                    batch.append({"text": text, "metadata": metadata})
                
                write_batch(batch)
                del const_data, batch
                gc.collect()
            except Exception as e:
                print(f"❌ Error processing Constitution: {e}")
        else:
            print("⚠️ Constitution file not found; skipping.")

        # 4. Statutes (IPC, CrPC, IEA, HMA, ICA)
        statutes_path = processed_dir / "statutes.json"
        if statutes_path.exists():
            print(f"Loading statutes from {statutes_path}")
            try:
                statute_docs = load_json(str(statutes_path))
                # No sampling for statutes - keep all 100%
                print(f"  Adding {len(statute_docs)} statute sections")
                write_batch(statute_docs)
                del statute_docs
                gc.collect()
            except Exception as e:
                print(f"❌ Error processing statutes: {e}")

        # Close JSON array
        out_f.write("\n]")
        
    print("✅ Done.")

if __name__ == "__main__":
    main()
