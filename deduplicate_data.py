# deduplicate_data_fixed.py
import json, os, hashlib
from collections import defaultdict
from typing import Dict, List, Tuple

DATA_PATH = os.path.join("data", "processed", "processed_cases.json")

def _text_hash(text: str) -> str:
    """Short deterministic hash of the document text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]

def _canonical_key(doc: Dict) -> Tuple[str, str]:
    case_name = doc.get("case_name", "").strip().lower()
    citation   = doc.get("metadata", {}).get("citation", "").strip().lower()

    # If we have a case name, use it (plus citation if any)
    if case_name:
        return (case_name, citation)

    # No case name → use a hash of the text (plus citation if any)
    text_hash = _text_hash(doc.get("text", ""))
    return (text_hash, citation)

def _record_score(doc: Dict) -> int:
    text_len = len(doc.get("text", ""))
    meta = doc.get("metadata", {})
    meta_fields = sum(1 for v in meta.values() if v)
    return text_len + meta_fields * 10

def deduplicate(records: List[Dict]) -> List[Dict]:
    groups = defaultdict(list)
    for rec in records:
        groups[_canonical_key(rec)].append(rec)

    cleaned, removed = [], 0
    for _, recs in groups.items():
        if len(recs) == 1:
            cleaned.append(recs[0])
        else:
            best = max(recs, key=_record_score)
            cleaned.append(best)
            removed += len(recs) - 1

    print(f"🔎 Processed {len(records)} records")
    print(f"🗑️ Removed {removed} duplicate entries")
    print(f"✅ Kept {len(cleaned)} unique records")
    return cleaned

def main() -> None:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned = deduplicate(data)

    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

    print(f"✅ Cleaned data written back to {DATA_PATH}")

if __name__ == "__main__":
    main()