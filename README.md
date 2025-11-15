# Indian Legal RAG System - Multi-Agent AI Debate Platform

## 📊 System Overview

**Advanced multi-agent AI legal debate system with comprehensive Indian legal database**

- **13,965+ document chunks** searchable via semantic embeddings
- **Complete Constitution of India** (all 396 articles)
- **15,622 unique Indian legal cases** (HuggingFace InLegalNER + IndianKanoon)
- **Multi-agent orchestration** using LangGraph (Prosecution, Defense, Moderator)
- **Semantic search** using HuggingFace embeddings (local, free, no API)
- **ML-based confidence scoring** for argument quality assessment

---

## 🚀 Quick Start

### Prerequisites
```bash
# Ensure Python 3.11+ is installed
python --version

# Create virtual environment (recommended)
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate      # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 1. Configure Environment
Create a `.env` file:
```bash
# Required: Google Gemini API Key
GOOGLE_API_KEY=your_gemini_api_key_here

# Embeddings (HuggingFace is default)
USE_HUGGINGFACE_EMBEDDINGS=true
```

### 2. Build FAISS Index (First Time Only)
```bash
# Option 1: Use existing merged dataset (recommended)
python rebuild_vector_store.py

# Option 2: Download fresh datasets and rebuild
python download_pretrained_datasets.py  # Download HuggingFace datasets
python merge_datasets.py                 # Merge with existing data
python rebuild_vector_store.py          # Rebuild FAISS index
```

### 3. Launch the Application
```bash
streamlit run app.py
```

Open: **http://localhost:8501**

---

## 📂 Project Structure

```
Major Judicial/
├── app.py                              # 🎯 Main Streamlit application (multi-agent debate)
├── requirements.txt                    # Python dependencies
├── .env                               # Configuration (API keys - NOT in Git)
├── .env.example                       # Template for environment variables
├── .gitignore                         # Git ignore rules
│
├── rag_system/                        # Core RAG components
│   ├── legal_rag.py                   # Production RAG query engine
│   └── vector_store.py                # FAISS vector database with HuggingFace
│
├── data/                              # Data directories
│   ├── raw/                           # Source data
│   │   ├── constitution_complete_395_articles.json    # 396 articles
│   │   ├── indiankanoon_massive_cases.json           # 7,247 original cases
│   │   ├── inlegal_train.json                        # 10,995 HuggingFace cases
│   │   ├── inlegal_test.json                         # 4,501 HuggingFace cases
│   │   └── merged_final_dataset.json                 # 15,622 unique cases
│   │
│   ├── processed/                     # Processed data
│   │   └── processed_cases.json       # 13,965 document chunks
│   │
│   └── vector_store/                  # FAISS index
│       └── faiss_index/               # Semantic embeddings
│
├── download_pretrained_datasets.py    # Download HuggingFace InLegalNER datasets
├── merge_datasets.py                  # Merge and deduplicate datasets
├── rebuild_vector_store.py            # Build/rebuild FAISS index
└── scrape_parallel.py                 # Multi-threaded scraper (8 workers)
```

**Note:** Large data files (JSON, FAISS index) are excluded from Git. You must scrape data and build the index locally.

---

## 🎯 Key Features

### 1. **Multi-Agent Debate System**
- ✅ **Prosecution Agent:** Argues for conviction/liability
- ✅ **Defense Agent:** Argues for acquittal/defense
- ✅ **Moderator Agent:** Evaluates arguments and provides verdicts
- ✅ **LangGraph Orchestration:** State-based workflow with memory
- ✅ **Round-based Debate:** Configurable number of rounds (1-10)

### 2. **Complete Legal Database**
- ✅ Constitution of India (396 articles, 22 parts, 12 schedules)
- ✅ 15,622 unique Indian legal cases from:
  - **10,995 cases** from HuggingFace InLegalNER (training set)
  - **4,501 cases** from HuggingFace InLegalNER (test set)
  - **7,247 cases** from IndianKanoon (Supreme Court & High Courts)
  - Deduplicated to remove overlaps
- ✅ Major Acts: IPC, CrPC, Evidence Act, CPC, NDPS, POCSO, IT Act, etc.
- ✅ 13,965+ searchable document chunks

### 3. **Semantic Search (RAG)**
- **HuggingFace Embeddings** (all-MiniLM-L6-v2, 384 dimensions)
- Understands meaning: "arrest" finds "detention" even without exact match
- Local processing: No API calls, complete privacy
- FAISS vector database for fast similarity search

### 4. **AI-Powered Arguments**
- **Google Gemini 2.0 Flash** for generation
- Real case law citations with excerpts
- Constitutional references (Articles, Parts, Schedules)
- Statutory provisions cited
- Structured legal reasoning

### 5. **ML-Based Confidence Scoring**
- **Multi-factor algorithm** (4 components):
  - Citation Quality (40%): Case + statute count
  - Vector Similarity (30%): RAG relevance scores
  - Argument Structure (20%): Points, reasoning, weaknesses
  - Legal Reasoning Depth (10%): TextBlob objectivity analysis
- **Transparent breakdown:** Shows component scores
- **Normalized range:** 0.6-0.95 to avoid extremes

---

## 🔧 Configuration

### `.env` File

```bash
# Required: Google Gemini API Key
GOOGLE_API_KEY=your_gemini_api_key_here

# Embeddings Configuration (HuggingFace is default)
USE_HUGGINGFACE_EMBEDDINGS=true

# Optional: Alternative embedding providers
# OPENAI_API_KEY=your_openai_key_here
```

### Get Google Gemini API Key
1. Go to: https://makersuite.google.com/app/apikey
2. Click "Create API Key"
3. Copy and paste into `.env` file

**Rate Limits (Free Tier):**
- 15 requests per minute
- 1 million tokens per minute
- 1,500 requests per day

---

## 📚 Data Sources

### Constitution
- **Source:** Official Constitution of India text
- **File:** `data/raw/constitution_complete_395_articles.json`
- **Content:** All 396 articles with descriptions
- **Size:** ~2 MB

### Case Law
Multiple sources merged and deduplicated:

1. **HuggingFace InLegalNER Dataset**
   - **Training Set:** 10,995 annotated Indian legal judgments
   - **Test Set:** 4,501 annotated Indian legal judgments
   - **Source:** https://huggingface.co/datasets/opennyaiorg/InLegalNER
   - **Files:** `inlegal_train.json`, `inlegal_test.json`

2. **IndianKanoon Scraped Cases**
   - **Count:** 7,247 cases (Supreme Court & High Courts)
   - **Source:** https://indiankanoon.org
   - **File:** `indiankanoon_massive_cases.json`
   - **Coverage:** Criminal, Constitutional, Civil, Property, Family, Tax, Labor
   - **Date Range:** 1950-2024

3. **Merged Dataset**
   - **Total Unique Cases:** 15,622 (after deduplication)
   - **File:** `merged_final_dataset.json`
   - **Duplicates Removed:** 7,121 overlapping cases

---

## 🛠️ Rebuilding the Index

If you add more data or want to rebuild:

```bash
# Rebuild with existing merged dataset
python rebuild_vector_store.py

# Or download fresh data and rebuild
python download_pretrained_datasets.py  # Requires HuggingFace authentication
python merge_datasets.py                 # Merge and deduplicate
python rebuild_vector_store.py          # Rebuild FAISS index
```

**Process:**
1. Loads merged dataset (15,622 unique cases)
2. Converts to processed format with metadata
3. Chunks documents (13,965 chunks @ 2000 characters each)
4. Generates embeddings (HuggingFace all-MiniLM-L6-v2, 384 dimensions)
5. Builds FAISS index
6. Saves to `data/vector_store/faiss_index/`

**Time:** ~2.5-3 hours for 243,761 documents with semantic embeddings
**Output Size:** ~500-800 MB
**Memory Required:** ~800 MB-1 GB RAM

---

## 📈 Scraping More Data

### Scrape Additional Cases
```bash
python scrape_massive.py
```
- **Current:** 7,247 cases
- **Target:** 10,000+ cases (configurable)
- **Queries:** 266 diverse legal topics
- **Time:** 6-8 hours for 7,247 cases
- **Rate Limiting:** 1.5s delay + random 0-0.5s
- **Retries:** 2 attempts, then skip
- **Checkpoint:** Saves after every query
- **Resume:** Automatically continues from last checkpoint

### Scrape Constitution
```bash
python fetch_complete_constitution.py
```
- **Source:** Official government sources
- **Output:** `data/raw/constitution_complete_395_articles.json`
- **Time:** ~5 minutes

---

## 🧪 Testing

```bash
# Test RAG system initialization
python -c "from rag_system.legal_rag import ProductionLegalRAGSystem; print('✅ RAG imports working')"

# Test FAISS index loading
python -c "from rag_system.legal_rag import ProductionLegalRAGSystem; rag = ProductionLegalRAGSystem(); print(f'✅ Index loaded successfully')"

# Test semantic search
python -c "from rag_system.legal_rag import ProductionLegalRAGSystem; rag = ProductionLegalRAGSystem(); results = rag.retrieve_documents('Article 21 right to life', k=3); print(f'✅ Found {len(results)} results')"

# Launch Streamlit app
streamlit run app.py
```

---

## 🎓 How It Works

### 1. **User Input**
User enters a case description in the Streamlit interface:
- Example: "A person was arrested without warrant under Section 302 IPC"

### 2. **Multi-Agent Orchestration (LangGraph)**
- **State Machine:** Manages debate flow with persistent memory
- **Round-based:** Configurable rounds (1-10)
- **Conditional Routing:** Prosecution → Defense → Moderator → Next Round

### 3. **Prosecution Agent**
- Queries RAG system for relevant cases (top-5)
- Receives: Case excerpts, citations, constitutional references
- Generates: Argument with case citations, statutes, legal reasoning
- Confidence score calculated (4-factor algorithm)

### 4. **Defense Agent**
- Receives prosecution argument + queries RAG
- Generates: Counter-argument with opposing precedents
- Acknowledges prosecution points, presents weaknesses
- Confidence score calculated

### 5. **Moderator Agent**
- Evaluates both arguments
- Scores: Prosecution vs Defense (0-10)
- Provides: Round winner, reasoning, suggestions
- Tracks cumulative scores

### 6. **Final Judgment**
- After all rounds, moderator delivers final verdict
- Winner declared based on cumulative scores
- Comprehensive analysis of debate quality
- Key precedents and legal principles summarized

### 7. **Confidence Scoring**
```python
confidence = (
    citation_quality * 0.40 +      # Case + statute count
    similarity_factor * 0.30 +     # RAG relevance
    structure_quality * 0.20 +     # Organization
    reasoning_depth * 0.10         # Objectivity (TextBlob)
)
normalized = 0.6 + (confidence * 0.35)  # Scale to 0.6-0.95
```

---

## 💾 System Requirements

- **Python:** 3.11+ (3.13 may have library compatibility issues)
- **RAM:** 2-4 GB minimum, 8 GB recommended
- **Storage:** 
  - ~1.5 GB for scraped data (JSON files)
  - ~500 MB for FAISS index
  - ~500 MB for HuggingFace model cache
  - **Total:** ~2.5 GB
- **CPU:** Multi-core recommended for faster embedding generation
- **Internet:** Required for:
  - Initial model download (~420 MB for sentence-transformers)
  - Google Gemini API calls during debates
  - Scraping new data

---

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| **Total Documents** | 243,761 chunks |
| **Source Cases** | 7,247 cases |
| **Constitution Articles** | 396 articles |
| **Index Size** | ~500-800 MB |
| **RAG Query Time** | ~150-200 ms |
| **Argument Generation Time** | ~3-4 seconds |
| **Full Debate (3 rounds)** | ~45-60 seconds |
| **Embedding Dimension** | 384 |
| **Embedding Model** | all-MiniLM-L6-v2 |
| **Chunk Size** | 500 words |
| **Chunk Overlap** | 50 words |
| **LLM Model** | Gemini 2.0 Flash |

---

## 🔐 Privacy & Security

- ✅ **All data stored locally** (no cloud storage)
- ✅ **Embeddings generated locally** (no API calls for RAG)
- ✅ **No user data collection or tracking**
- ✅ **Open source** - audit the code yourself
- ⚠️ **Google Gemini API** used for debate generation (text sent to Google)
  - Only case descriptions and generated arguments sent
  - No personal information transmitted
  - Consider data sensitivity before using
- ⚠️ **API Keys** stored in `.env` file (excluded from Git)
  - Never commit `.env` to Git
  - Use `.env.example` as template

---

## 🚧 Known Issues & Troubleshooting

### Python 3.13 Compatibility
- Some ML libraries (torch, transformers) may have pre-compiled wheel issues
- **Workaround:** Use Python 3.11 if problems occur
- Current system tested with Python 3.11-3.13

### Rate Limiting (429 Error)
- **Issue:** Google Gemini free tier: 15 requests/minute
- **Symptom:** "ResourceExhausted: 429" errors during debates
- **Solution:** 
  - Reduce number of rounds (1-3 recommended)
  - Wait 60 seconds between debates
  - Consider upgrading to paid tier
  - Use multiple API keys (rotate in code)

### Large File in Git
- **Issue:** Cannot push to GitHub (files > 100 MB)
- **Solution:** Ensure `.gitignore` excludes:
  - `data/raw/*.json`
  - `data/vector_store/`
- **Fix if already committed:**
  ```bash
  git rm --cached data/raw/*.json
  git rm -r --cached data/vector_store/
  git commit -m "Remove large files"
  ```

### FAISS Index Not Found
- **Issue:** "FileNotFoundError: FAISS index not found"
- **Solution:** Run `python build_with_constitution.py`
- **Time:** ~2-3 hours

### HuggingFace Model Download Slow
- **First run downloads** ~420 MB model
- **Cached for future use** in `~/.cache/huggingface/`
- **One-time download** per machine

---

## 📝 Future Enhancements

### Phase 1: Data & Performance ✅ (COMPLETED)
- [x] Complete Constitution (396 articles)
- [x] 7,247+ real cases
- [x] Multi-agent orchestration (LangGraph)
- [x] ML-based confidence scoring
- [x] Semantic embeddings (HuggingFace)

### Phase 2: Advanced Features 🚧 (IN PROGRESS)
- [ ] Citation verification (cross-reference with database)
- [ ] Legal precedent graph visualization (NetworkX, Plotly)
- [ ] Argument strength radar charts
- [ ] Export to legal brief format (DOCX/PDF)
- [ ] Multi-language support (Hindi, Tamil, Telugu)

### Phase 3: Intelligence 🎯 (PLANNED)
- [ ] Outcome prediction (ML model)
- [ ] Case law similarity heatmap
- [ ] User rebuttal system (interactive debate)
- [ ] Analytics dashboard (historical stats)
- [ ] Voice narration (text-to-speech)

### Phase 4: Production Ready 🚀 (PLANNED)
- [ ] User authentication & sessions
- [ ] Query history & saved debates
- [ ] API endpoints (REST API)
- [ ] Deployment (Docker, cloud hosting)
- [ ] Performance optimization (caching, batching)

---

## 📞 Support & Troubleshooting

### Common Issues

**1. Index Not Found**
```bash
# Build the FAISS index first
python build_with_constitution.py
```

**2. API Key Error**
```bash
# Create .env file with:
GOOGLE_API_KEY=your_key_here
USE_HUGGINGFACE_EMBEDDINGS=true
```

**3. Import Errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**4. Rate Limit (429)**
- Reduce rounds to 1-3
- Wait 60s between debates
- Use multiple API keys

**5. Memory Error**
- Close other applications
- Reduce batch size in `build_with_constitution.py`
- Ensure 4+ GB RAM available

### Verification Commands
```bash
# Check Python version
python --version

# Test HuggingFace model
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Verify FAISS index
python -c "import os; print('✅ Index exists' if os.path.exists('data/vector_store/faiss_index') else '❌ Index missing')"

# Test RAG system
python -c "from rag_system.legal_rag import ProductionLegalRAGSystem; rag = ProductionLegalRAGSystem(); print('✅ RAG working')"
```

---

## 🙏 Credits & Technologies

### Data Sources
- **Legal Cases:** Indian Kanoon (indiankanoon.org)
- **Constitution:** Official Government of India sources
- **Acts & Statutes:** Public government databases

### AI & ML Stack
- **LLM:** Google Gemini 2.0 Flash (text generation)
- **Embeddings:** HuggingFace Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Store:** FAISS by Meta AI (similarity search)
- **Orchestration:** LangChain + LangGraph (multi-agent workflows)
- **NLP:** TextBlob (sentiment analysis)

### Framework & UI
- **Web Framework:** Streamlit (interactive UI)
- **Python:** 3.11+
- **Data Processing:** NumPy, Pandas

### Development Tools
- **Version Control:** Git, GitHub
- **Environment:** Python venv, dotenv
- **Scraping:** BeautifulSoup4, requests

---

## 📄 License & Disclaimer

### License
MIT License - Free for educational and research purposes.

### Disclaimer
⚠️ **This is an AI-powered research tool, not a substitute for legal advice.**

- Generated debates are for **educational purposes only**
- Do **not** rely on this for actual legal decisions
- Always consult qualified legal professionals
- Case citations should be independently verified
- AI may hallucinate or misinterpret legal principles
- No warranty or guarantee of accuracy

### Data Usage
- Legal data sourced from public government databases
- Fair use for educational and research purposes
- Respect original source terms of service
- Not affiliated with Indian Kanoon or Government of India

---

## 📈 Project Stats

- **Repository:** [github.com/Tumul001/MajorLegal](https://github.com/Tumul001/MajorLegal)
- **Last Updated:** November 15, 2025  
- **Version:** 3.0 (Multi-Agent System with ML Confidence Scoring)
- **Total Code:** ~1,500 lines (app.py: 1,463 lines)
- **Development Time:** 16 weeks
- **Status:** 🟢 Production Ready (80% complete)

---

**Built with ❤️ for Indian Legal Tech**
