import os
from huggingface_hub import hf_hub_download
import zipfile
import json

# Download 1.Base Dataset.zip (should be cached)
file_path = hf_hub_download(repo_id="L-NLProc/NyayaRAG", filename="1.Base Dataset.zip", repo_type="dataset")

print("\nInspecting Base Dataset/SCI_judgements_56k.json:")
with zipfile.ZipFile(file_path, 'r') as zip_ref:
    with zip_ref.open('Base Dataset/SCI_judgements_56k.json') as f:
        data = json.load(f)
        if isinstance(data, list):
            print(f"Loaded list of {len(data)} items")
            first_item = data[0]
            print("Keys:", list(first_item.keys()))
            # Print a preview of values for each key to identify the text field
            for k, v in first_item.items():
                print(f"Key: {k}, Type: {type(v)}")
                if isinstance(v, str):
                    print(f"  Value preview: {v[:100]}...")
        elif isinstance(data, dict):
             print(f"Loaded dict with keys: {list(data.keys())}")
