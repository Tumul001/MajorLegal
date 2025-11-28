import os
from huggingface_hub import hf_hub_download
import zipfile

# Download 1.Base Dataset.zip
print("Downloading 1.Base Dataset.zip...")
file_path = hf_hub_download(repo_id="L-NLProc/NyayaRAG", filename="1.Base Dataset.zip", repo_type="dataset")
print(f"Downloaded to: {file_path}")

# Inspect zip content
print("\nZip contents:")
with zipfile.ZipFile(file_path, 'r') as zip_ref:
    file_list = zip_ref.namelist()
    print(f"Total files: {len(file_list)}")
    print("First 10 files:")
    for f in file_list[:10]:
        print(f" - {f}")
    
    # Extract first file to inspect content if it's text/json
    first_file = file_list[0]
    if first_file.endswith('.json') or first_file.endswith('.txt'):
        print(f"\nContent of {first_file}:")
        with zip_ref.open(first_file) as f:
            content = f.read(500) # Read first 500 bytes
            print(content.decode('utf-8', errors='ignore'))
