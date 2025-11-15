# Files to Keep (Essential for Running on Another Device)

## Core Application
- app.py                          # Main Streamlit application
- requirements.txt                # Python dependencies

## RAG System
- rag_system/
  - legal_rag.py                  # RAG implementation
  - vector_store.py               # FAISS vector store

## Evaluation & Metrics
- evaluation_metrics.py           # Evaluation framework
- test_evaluation.py              # Test suite

## Data Processing
- download_pretrained_datasets.py # Download datasets
- merge_datasets.py               # Merge & deduplicate
- rebuild_vector_store.py         # Build FAISS index
- scrape_parallel.py              # Parallel scraper (optional)

## Documentation
- README.md                       # Project documentation
- EVALUATION_METRICS.md           # Metrics explanation
- EVALUATION_RESULTS.md           # Latest results
- FIXES_SUMMARY.md                # Critical fixes summary
- NO_HUMAN_ANNOTATION.md          # Evaluation guide

## Configuration
- .env.example                    # Environment template
- .gitignore                      # Git ignore rules

## Data (if synced)
- data/
  - raw/merged_final_dataset.json
  - processed/processed_cases.json
  - faiss_index/

---

# Files to REMOVE (Temporary/Generated)

## Test Outputs
- automated_evaluation_report.json    # Generated test output
- test_evaluation_report.json         # Generated test output  
- test_suite_automated.json           # Generated test cases
- real_evaluation_report.json         # Generated evaluation results

## Wrapper Scripts (Not Needed)
- run_evaluation_safe.py              # UTF-8 wrapper (Windows-specific)

## Python Cache (Auto-generated)
- __pycache__/                        # Python bytecode cache
- rag_system/__pycache__/
- data_collection/__pycache__/
- **/*.pyc                            # Compiled Python files

## Environment (Device-specific)
- .env                                # Your API keys (DO NOT SYNC)
- venv/                              # Virtual environment (recreate on new device)

## Git (Auto-managed)
- .git/                              # Git repository (keeps automatically)

---

# Commands to Clean Up

```powershell
# Remove temporary JSON outputs
Remove-Item automated_evaluation_report.json, test_evaluation_report.json, test_suite_automated.json, real_evaluation_report.json -Force -ErrorAction SilentlyContinue

# Remove wrapper script
Remove-Item run_evaluation_safe.py -Force -ErrorAction SilentlyContinue

# Remove all Python cache
Get-ChildItem -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
Get-ChildItem -Recurse -Filter "*.pyc" | Remove-Item -Force

# Stage deletions
git add -A
git status
```

---

# On New Device - Setup Steps

```powershell
# 1. Clone repository
git clone https://github.com/Tumul001/MajorLegal.git
cd MajorLegal

# 2. Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create .env file (copy from .env.example)
copy .env.example .env
# Edit .env with your API keys

# 5. Download datasets (if not synced)
python download_pretrained_datasets.py

# 6. Build vector store (if not synced)
python rebuild_vector_store.py

# 7. Run application
streamlit run app.py
```

---

# .gitignore Should Include

```
# Environment
.env
venv/

# Python cache
__pycache__/
*.pyc
*.pyo

# Test outputs
*_report.json
test_suite_*.json

# Large data (optional - depends on your choice)
data/raw/*.json
data/processed/*.json
data/faiss_index/

# OS files
.DS_Store
Thumbs.db
```
