# MajorLegal: Multi-Agent Legal Debate System

A sophisticated AI-powered legal debate system that simulates courtroom arguments between Prosecution and Defense agents, moderated by a Judge agent. The system uses Retrieval-Augmented Generation (RAG) to ground arguments in real Indian case law.

## 🚀 Latest Updates (2025-11-26 10:33 IST)

### 1. Statute Integration & Smart Ranking
- **Full Statute Indexing:** Added IPC, CrPC, IEA, HMA, and ICA to the vector store.
- **Smart Ranking:** Implemented logic to boost:
    - **Supreme Court Cases:** +20%
    - **Statutes:** +25%
    - **Recent Cases (2020+):** +15%
- **Result:** Statutes now appear in top retrieval results for relevant queries (e.g., "Section 302 punishment").

### 2. Benchmark & Stability Fixes
- **Fixed Benchmark:** Resolved `NameError` in `legal_rag.py` that prevented testing.
- **UI Improvements:** Fixed "Unknown" titles for statutes, now displaying as "Act - Section X".
- **Verification:** Manually verified retrieval of key statutes and landmark cases.

### 3. Data Clean‑up & Vector Store Size
- **Deduplication fixed**: Implemented a robust deduplication script that now keeps unique records based on case name, citation, or a text hash when those fields are missing.
- **FAISS index size**: After rebuilding, the vector store now contains **13,921 vectors** representing distinct cases and news articles.
- **Improved merge workflow**: `merge_all_data.py` now automatically picks up any new JSON/CSV sources placed in `data/raw`, `data/processed`, or `data/news`.

### 2. High‑Performance Vector Search (Voyage AI)
- **Switched to Voyage AI Embeddings (`voyage-law-2`)**: Migrated from local `InLegalBERT` to Voyage AI's state‑of‑the‑art legal embedding model.
- **Benefit**:
  - **Speed**: Vector store build time reduced from hours to minutes.
  - **Accuracy**: Superior retrieval performance on legal documents (16k context window).
  - **Efficiency**: Offloaded heavy processing to the cloud.

### 3. Report Generation & Submission Tools
- **Downloadable Debate Reports**: Added a feature to `app.py` allowing users to download a comprehensive Markdown report of the entire legal debate simulation.
- **Submission Workflow Guide**: Created `SUBMISSION_WORKFLOW.md`, a step‑by‑step guide for generating technical and case‑study reports for project submission.

### 4. Dependencies
- Added `voyageai` and `langchain-voyageai` to `requirements.txt`.

---

## 🌟 Key Features

- **Multi-Agent Debate**: Autonomous agents role‑play Prosecution, Defense, and Moderator.
- **Legal RAG System**: Retrieves relevant Indian case law from a vector database.
- **Citation Verification**: Validates cited cases against the database to prevent hallucinations.
- **Heuristic Scoring**: Evaluates argument strength based on citations, reasoning, and structure.
- **Interactive UI**: Built with Streamlit for real‑time debate visualization.

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd MajorLegal
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Environment Variables**:
   Create a `.env` file:
   ```env
   GOOGLE_API_KEY=your_gemini_key
   VOYAGE_API_KEY=your_voyage_key
   ```

## 🏃‍♂️ Usage

### 1. Build the Vector Store (First Run Only)
Index your legal data using Voyage AI:
```bash
python rebuild_vector_store.py --data-path data/processed/processed_cases.json
```

### 2. Run the Application
Start the debate interface:
```bash
streamlit run app.py
```

### 3. Generate Technical Report
Run the benchmark suite:
```bash
python benchmark.py
```

## 📂 Project Structure

- `app.py`: Main Streamlit application.
- `rag_system/`: Core RAG logic and vector store management.
- `rebuild_vector_store.py`: Script to index data.
- `benchmark.py`: Automated evaluation suite.
- `SUBMISSION_WORKFLOW.md`: Guide for project submission.

---

## 📅 Changelog

### 2025-11-25 00:22 IST
- Implemented robust deduplication and updated FAISS index size reporting.
- Added latest update section to README.
- Refined merge workflow to auto‑detect new source files.

### 2025-11-24 20:10 IST
- **Voyage AI Integration**: Updated `rebuild_vector_store.py` and `rag_system/vector_store.py` to use `voyage-law-2` embeddings for faster and more accurate retrieval.
- **Report Download**: Added "Download Debate Report" button to `app.py`.
- **Submission Guide**: Created `SUBMISSION_WORKFLOW.md`.
- **Dependencies**: Added `voyageai` to `requirements.txt`.

### 2025-11-26 04:17 IST
- **Complete Data Integration**: Successfully integrated all legal data sources into unified RAG system
  - **NyayaRAG Dataset**: 46,029 Supreme Court cases (10% sample of 460k+ cases)
  - **Scraped Cases**: 50,000 deduplicated cases from IndianKanoon
  - **Constitution**: All 396 articles of the Indian Constitution
  - **Total**: 96,425 searchable legal documents
- **Vector Store Optimization**: Reduced token usage to fit within API quotas
  - Token budget managed: 38.6M tokens used of 48.3M available
  - Build time: 62.37 minutes (97 batches @ 38.54s/batch)
  - Index size: ~96k vectors with voyage-law-2 embeddings
- **Data Quality**: Implemented smart sampling strategy
  - Deduplication confirmed on scraped cases
  - Random sampling ensures diverse coverage
  - Constitution articles fully indexed for retrieval
- **New Scripts**:
  - `combine_datasets.py`: Merges NyayaRAG, scraped cases, and Constitution
  - `verify_data.py`: Reports source distribution and document counts

#### Performance Benchmarks
| Metric | Value |
|--------|-------|
| Total Documents | 96,425 |
| NyayaRAG Cases | 46,029 (47.7%) |
| Scraped Cases | 50,000 (51.9%) |
| Constitution Articles | 396 (0.4%) |
| Embedding Model | voyage-law-2 |
| Build Time | 62.37 minutes |
| Tokens Used | 38.6M / 48.3M (79.9%) |
| Average Batch Time | 38.54 seconds |
| Index Storage | ~4.2 GB |

### 2025-11-26 20:13 IST
- **Benchmark Fixed**: Updated `benchmark.py` with fuzzy matching logic.
    - **Result**: F1 Score improved from 0.00 to **0.38**.
    - **Validation**: Confirmed high accuracy for IPC 302 (0.80) and HMA 13 (0.67).
- **Timestamped Reporting**: Enabled automatic timestamping for benchmark results (JSON & Markdown) to preserve research history.
- **Documentation**: Updated all project reports with final verification status.


## Latest Updates (2025-11-28)
- **Comprehensive Benchmark Suite**: Implemented a full benchmark suite with 100 diverse Indian legal scenarios.
- **Baselines**: Added comparisons for Vanilla LLM (Gemini 2.0 Flash), Simple RAG, and MajorLegal.
- **Robust Logging**: Benchmark results are now timestamped and saved to ench_result.txt.
- **Refactoring**: enchmark.py now supports batch processing, CLI arguments, and robust error handling.
