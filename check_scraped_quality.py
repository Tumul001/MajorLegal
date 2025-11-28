import json

# Load scraped cases
with open('data/processed/processed_scraped_cases.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Check first 5 cases
print(f"Total scraped cases: {len(data)}\n")

for i, case in enumerate(data[:5]):
    metadata = case.get('metadata', {})
    text = case.get('text', '')
    
    print(f"=== Case {i+1} ===")
    print(f"Metadata keys: {list(metadata.keys())}")
    print(f"Case name: {metadata.get('case_name', 'N/A')[:80]}")
    print(f"Citation: {metadata.get('citation', 'N/A')}")
    print(f"Case ID: {metadata.get('case_id', 'N/A')}")
    print(f"Court: {metadata.get('court', 'N/A')}")
    print(f"Date: {metadata.get('date', 'N/A')}")
    print(f"Text length: {len(text)} chars")
    print(f"Text preview: {text[:150]}...")
    print()
