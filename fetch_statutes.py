import requests
import json
import os
from pathlib import Path

# Configuration
DATA_DIR = Path("data/processed")
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = DATA_DIR / "statutes.json"

# Source URLs (CivicTech India)
SOURCES = {
    "IPC": {
        "url": "https://raw.githubusercontent.com/civictech-India/Indian-Law-Penal-Code-Json/main/ipc.json",
        "name": "Indian Penal Code, 1860"
    },
    "CrPC": {
        "url": "https://raw.githubusercontent.com/civictech-India/Indian-Law-Penal-Code-Json/main/crpc.json",
        "name": "Code of Criminal Procedure, 1973"
    },
    "IEA": {
        "url": "https://raw.githubusercontent.com/civictech-India/Indian-Law-Penal-Code-Json/main/iea.json",
        "name": "Indian Evidence Act, 1872"
    },
    "HMA": {
        "url": "https://raw.githubusercontent.com/civictech-India/Indian-Law-Penal-Code-Json/master/hma.json",
        "name": "Hindu Marriage Act, 1955"
    }
}

def fetch_json(url):
    """Fetch JSON content from URL"""
    print(f"⬇️ Fetching {url}...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"❌ Failed to fetch {url}: {e}")
        return None

def normalize_statute(data, act_code, act_name):
    """Normalize statute data into standard document format"""
    documents = []
    
    # Handle different JSON structures
    # Some might be lists of sections, others might be nested
    
    items = data if isinstance(data, list) else [data]
    
    for item in items:
        # Special handling for HMA (CSV inside JSON key)
        if "chapter,section,section_title,section_desc" in item:
            raw_val = item["chapter,section,section_title,section_desc"]
            if not raw_val: continue
            
            # Simple CSV parse (naive split, but works for this structure)
            parts = raw_val.split(',', 3)
            if len(parts) < 4: continue
            
            chapter = parts[0]
            section = parts[1]
            title = parts[2]
            desc = parts[3].strip('"') # Remove quotes
            
        else:
            # Standard handling
            section = item.get('Section') or item.get('section')
            chapter = item.get('chapter') or item.get('Chapter')
            title = item.get('section_title') or item.get('section_name') or item.get('title')
            desc = item.get('section_desc') or item.get('description') or item.get('text')
        
        if not desc or not section:
            continue
            
        # Create rich text representation
        full_text = f"{act_name}\nSection {section}: {title}\n\n{desc}"
        
        doc = {
            "text": full_text,
            "metadata": {
                "source": "Statute",
                "act": act_name,
                "act_code": act_code,
                "section": str(section),
                "chapter": str(chapter) if chapter else "N/A",
                "title": title or "Unknown",
                "citation": f"Section {section}, {act_code}"
            }
        }
        documents.append(doc)
        
    return documents

def main():
    all_statutes = []
    
    print("🚀 Starting Statute Fetch...")
    
    for code, info in SOURCES.items():
        data = fetch_json(info['url'])
        if data:
            docs = normalize_statute(data, code, info['name'])
            print(f"✅ Parsed {len(docs)} sections for {code}")
            all_statutes.extend(docs)
            
    # Manual fallback for Contract Act (if not found online)
    # Adding key sections for benchmark
    print("✍️ Adding manual entries for Indian Contract Act...")
    contract_docs = [
        {
            "text": "Indian Contract Act, 1872\nSection 10: What agreements are contracts\n\nAll agreements are contracts if they are made by the free consent of parties competent to contract, for a lawful consideration and with a lawful object, and are not hereby expressly declared to be void.",
            "metadata": {"source": "Statute", "act": "Indian Contract Act, 1872", "act_code": "ICA", "section": "10", "citation": "Section 10, ICA"}
        },
        {
            "text": "Indian Contract Act, 1872\nSection 11: Who are competent to contract\n\nEvery person is competent to contract who is of the age of majority according to the law to which he is subject, and who is of sound mind and is not disqualified from contracting by any law to which he is subject.",
            "metadata": {"source": "Statute", "act": "Indian Contract Act, 1872", "act_code": "ICA", "section": "11", "citation": "Section 11, ICA"}
        }
    ]
    all_statutes.extend(contract_docs)
    
    # Save combined file
    print(f"💾 Saving {len(all_statutes)} total statute documents to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_statutes, f, indent=2, ensure_ascii=False)
        
    print("✅ Done!")

if __name__ == "__main__":
    main()
