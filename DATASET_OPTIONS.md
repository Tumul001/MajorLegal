# Dataset Size Options for Vector Store Rebuild

## Current Situation
- **Your original data:** 13,921 document chunks (from web scraping)
- **Current build:** Using InLegalBERT (768-dim) - **~2 hours** to complete
- **Progress:** 1% complete (16.9 sec/batch × 434 batches)

## Options

### Option A: Full Dataset (Production Mode)
**Pros:**
- All 13,921 chunks with InLegalBERT embeddings
- Best retrieval quality for real use
- Complete data coverage

**Cons:**  
- Takes ~2 hours to build
- Must wait to test features

**Command:**
```bash
# Already running - let it continue
```

### Option B: Subset for Testing (2000 chunks) ✅ RECOMMENDED
**Pros:**
- Ready in ~15 minutes
- Test all refactoring features immediately
- Sufficient for benchmarking and validation
- Can rebuild full dataset later

**Cons:**
- Limited data coverage (14% of full dataset)

**Command:**
```bash
# Cancel current build, then run:
python -c "import json; data=json.load(open('data/processed/processed_cases.json', encoding='utf-8')); json.dump(data[:2000], open('data/processed/subset_2000.json', 'w', encoding='utf-8'))"
python rebuild_vector_store.py --data-path data/processed/subset_2000.json --verify-embeddings
```

### Option C: Keep ILDC Dataset (95 chunks)
**Pros:**
- Already built and working
- Demonstrates all features

**Cons:**
- Only 95 chunks - not representative
- Poor retrieval quality

## Recommendation

**Use Option B** - Create 2000-chunk subset to:
1. Test Graph-RAG hybrid retrieval
2. Run legal validation
3. Execute benchmark
4. Demo the system

Then rebuild full 13k dataset overnight for production use.

## Latest Updates (2025-11-28)
- **Comprehensive Benchmark Suite**: Implemented a full benchmark suite with 100 diverse Indian legal scenarios.
- **Baselines**: Added comparisons for Vanilla LLM (Gemini 2.0 Flash), Simple RAG, and MajorLegal.
- **Robust Logging**: Benchmark results are now timestamped and saved to ench_result.txt.
- **Refactoring**: enchmark.py now supports batch processing, CLI arguments, and robust error handling.
