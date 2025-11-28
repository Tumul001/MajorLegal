# Complete workflow test - ingestion through benchmarking
import json
import os

data_path = "data/processed/combined_cases.json"
if os.path.exists(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"✅ Data ingested: {len(data)} document chunks")
    # Count by source metadata
    source_counts = {}
    for doc in data:
        src = doc.get("metadata", {}).get("source", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1
    print("📊 Documents per source:")
    for src, cnt in source_counts.items():
        print(f"   - {src}: {cnt}")
    if data:
        print(f"   First doc has {len(data[0]['text'].split())} words")
else:
    print("❌ No data found")