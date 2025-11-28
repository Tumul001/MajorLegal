# Dataset Availability Update

## ⚠️ NyayaAnumana Status

**Issue:** The full 702k NyayaAnumana dataset is **NOT publicly available** on Hugging Face yet.

### What I Found
- Papers mention the dataset exists (702,945 cases)
- Dataset is under L-NLProc organization
- Only research results and derived datasets are available
- Full dataset may require special access or institutional affiliation

## ✅ Available Alternatives

### Option 1: Your Existing 13k Dataset (RECOMMENDED)
- **Status:** ✅ You already have this
- **Size:** 13,921 document chunks
- **Quality:** Web-scraped, good coverage
- **Action:** Let current InLegalBERT build finish (~30 min remaining)
- **Total time:** Ready in <1 hour

### Option 2: L-NLProc/NyayaRAG
- **Status:** ✅ Available on HuggingFace  
- **Size:** 56,387 Supreme Court cases
- **Quality:** High (Supreme Court only)
- **Dataset path:** `L-NLProc/NyayaRAG`
- **Total time:** ~3-4 hours (ingest + build)

###  Option 3: ILDC (Small)
- **Status:** ✅ Already tested
- **Size:** 54 cases → 95 chunks
- **Quality:** Good for testing only
- **Action:** Already working

## 🎯 Recommendation

**Use Option 1** - Your existing 13k dataset with InLegalBERT:

**Advantages:**
- ✅ Data already ingested
- ✅ InLegalBERT build in progress (almost done)
- ✅ Proven to work
- ✅ Good size for production

**Next steps:**
1. Let current build finish (~30 min)
2. Build citation graph
3. Test the system
4. Run benchmark

## Alternative: NyayaRAG

If you want to try NyayaRAG (56k cases), I need to:
1. Add support for it in `ingest_data.py`
2. Handle its schema
3. Ingest data (~2 hours)
4. Build vectors (~4-5 hours)

**Total: ~7 hours** vs **30 minutes** for Option 1
- **NyayaRAG**: Sampled 10% (46,029 of 460,293 cases) to manage token budget
- **Scraped Cases**: Kept 50,000 of 112,012 deduplicated cases
- **Constitution**: All 396 articles fully indexed
- **Total Documents**: 96,425

#### Rationale
1. **Token Budget Constraints**: Full NyayaRAG (460k cases) would require ~184M tokens; quota was only 48.3M
2. **Smart Sampling**: 10% random sample maintains diversity while fitting budget
3. **Comprehensive Coverage**: Combination provides:
   - Supreme Court precedents (NyayaRAG)
   - Additional case law (scraped)
   - Constitutional provisions (full articles)

#### Performance Metrics
| Metric | Value |
|--------|-------|
| Build Time | 62.37 minutes |
| Tokens Consumed | 38.6M (79.9% of quota) |
| Embedding Model | voyage-law-2 |
| Index Size | ~4.2 GB |
| Query Retrieval | Top-15 chunks per query |

#### Data Quality
- ✅ Deduplication verified on scraped cases
- ✅ Random sampling ensures statistical representation
- ✅ All data sources now searchable via unified RAG system


---

## 🏁 Final Decision & Status (2025-11-26 20:13 IST)

### Selected Strategy: Hybrid Approach
We successfully implemented a hybrid dataset strategy that balances coverage, quality, and token budget:

1.  **Core Case Law:** 13,921 high-quality scraped cases (deduplicated).
2.  **Statutes:** Full text of IPC, CrPC, IEA, HMA, ICA (Essential for rule-based queries).
3.  **Landmark Cases:** Whitelisted important Supreme Court judgments.

### Outcome
- **Token Budget:** Stayed under 48M limit.
- **Retrieval Quality:** High relevance for both broad legal concepts and specific statute lookups.
- **Benchmark:** Validated with 0.38 F1 score (after fixing metrics).

**Verdict:** The current dataset is sufficient for the "MajorLegal" demo and research submission.


## Latest Updates (2025-11-28)
- **Comprehensive Benchmark Suite**: Implemented a full benchmark suite with 100 diverse Indian legal scenarios.
- **Baselines**: Added comparisons for Vanilla LLM (Gemini 2.0 Flash), Simple RAG, and MajorLegal.
- **Robust Logging**: Benchmark results are now timestamped and saved to ench_result.txt.
- **Refactoring**: enchmark.py now supports batch processing, CLI arguments, and robust error handling.
