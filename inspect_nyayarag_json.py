import os
from huggingface_hub import hf_hub_download
import zipfile
import json

# Download 1.Base Dataset.zip (should be cached)
file_path = hf_hub_download(repo_id="L-NLProc/NyayaRAG", filename="1.Base Dataset.zip", repo_type="dataset")

print("\nInspecting Base Dataset/SCI_judgements_56k.json:")
with zipfile.ZipFile(file_path, 'r') as zip_ref:
    with zip_ref.open('Base Dataset/SCI_judgements_56k.json') as f:
        # Read first line or chunk to see structure
        # It might be a list of dicts or jsonl
        content = f.read(1000)
        try:
            text = content.decode('utf-8')
            print(f"First 1000 chars:\n{text}")
        except Exception as e:
            print(f"Error decoding: {e}")

    # Try to load it as json if it looks like it
    try:
        with zip_ref.open('Base Dataset/SCI_judgements_56k.json') as f:
            data = json.load(f)
            if isinstance(data, list):
                print(f"\nLoaded as list. Length: {len(data)}")
                print("First item keys:", data[0].keys())
                print("First item sample:", json.dumps(data[0], indent=2)[:500])
            elif isinstance(data, dict):
                print(f"\nLoaded as dict. Keys: {data.keys()}")
    except Exception as e:
        print(f"\nCould not load as full JSON (might be too big or JSONL): {e}")
