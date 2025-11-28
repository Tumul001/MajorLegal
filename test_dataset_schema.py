"""Quick test to inspect ILDC dataset schema"""
from datasets import load_dataset

dataset = load_dataset("anuragiiser/ILDC_expert", split="train[:1]")
print("Dataset keys:", dataset.column_names)
print("\nFirst example:")
for key, value in dataset[0].items():
    if isinstance(value, str):
        print(f"{key}: {value[:200] if len(value) > 200 else value}")
    else:
        print(f"{key}: {value}")
