# Indian Legal RAG System - AI Debate Platform

## 📊 System Overview

**Complete Indian legal database with AI-powered debate generation**

- **75,741 document chunks** searchable via semantic embeddings
- **Complete Constitution of India** (all 395 articles)
- **1,919 real court cases** from Indian Kanoon
- **Semantic search** using HuggingFace embeddings (local, free, no API)

---

## 🚀 Quick Start

### 1. Launch the App
```bash
streamlit run app.py
```

Open: **http://localhost:8501**

### 2. Test the System
```bash
python test_rag.py
```

---

## 📂 Project Structure

```
Major Judicial/
├── app.py                              # Main Streamlit application
├── requirements.txt                    # Python dependencies
├── .env                               # Configuration (API keys, embedding settings)
│
├── rag_system/                        # Core RAG components
│   ├── legal_rag.py                   # RAG query engine
│   ├── vector_store.py                # FAISS vector database
│   ├── simple_mock_embeddings.py      # Fallback embeddings
│   └── sources.py                     # Legal data sources
│
├── data/
│   ├── raw/                           # Source data
│   │   ├── constitution_complete_395_articles.json    # All 395 articles
│   │   └── indiankanoon_massive_cases.json           # 1,919 cases
│   │
│   └── vector_store/                  # FAISS index
│       └── faiss_index/               # Semantic vectors
│
├── scrape_indiankanoon.py             # Scraper for legal cases
├── scrape_massive.py                  # Bulk scraper (50K target)
├── fetch_complete_constitution.py     # Constitution scraper
├── build_with_constitution.py         # Index builder
└── test_rag.py                        # System tests
```

---

## 🎯 Key Features

### 1. **Complete Legal Database**
- ✅ Constitution of India (395 articles, 22 parts, 12 schedules)
- ✅ 1,919 real Indian court cases
- ✅ Major Acts: IPC, CrPC, Evidence Act, CPC, NDPS, POCSO, etc.

### 2. **Semantic Search**
- **HuggingFace Embeddings** (all-MiniLM-L6-v2, 384 dimensions)
- Understands meaning: "arrest" finds "detention" even without exact match
- Local processing: No API calls, complete privacy

### 3. **AI Debate System**
- Two-side legal debate generation
- Prosecution vs Defense arguments
- Real case law citations
- Constitutional references

---

## 🔧 Configuration

### `.env` File

```bash
# HuggingFace Embeddings (Recommended)
USE_HUGGINGFACE_EMBEDDINGS=true

# Google Gemini API Key (for AI debates)
GOOGLE_API_KEY=your_key_here

# Fallback Mock Embeddings (if HuggingFace fails)
USE_MOCK_EMBEDDINGS=false
```

---

## 📚 Data Sources

### Constitution
- **Source:** Official Constitution of India text
- **File:** `data/raw/constitution_complete_395_articles.json`
- **Content:** All 395 articles with descriptions

### Case Law
- **Source:** Indian Kanoon (indiankanoon.org)
- **File:** `data/raw/indiankanoon_massive_cases.json`
- **Content:** 1,919 Supreme Court and High Court cases
- **Coverage:** Criminal, Constitutional, Civil, Property, Family Law

---

## 🛠️ Rebuilding the Index

If you add more data or want to rebuild with different embeddings:

```bash
python build_with_constitution.py
```

**Time:** ~45 minutes for 75K documents with semantic embeddings

**Output:** `data/vector_store/faiss_index/`

---

## 📈 Scraping More Data

### Scrape Additional Cases
```bash
python scrape_massive.py
```
- **Target:** 50,000+ cases
- **Time:** 30-35 hours
- **Checkpoint:** Saves every 5 queries
- **Resume:** Automatically continues from last checkpoint

### Scrape Constitution Updates
```bash
python fetch_complete_constitution.py
```

---

## 🧪 Testing

```bash
# Test RAG retrieval
python test_rag.py

# Test specific query
python -c "from rag_system.legal_rag import IndianLegalRAG; rag = IndianLegalRAG(); results = rag.query('Article 21 right to life'); print(f'Found {len(results)} results')"
```

---

## 🎓 How It Works

### 1. **User Query**
User asks a legal question in the Streamlit app

### 2. **Semantic Search**
- Query converted to 384-dim vector
- FAISS finds similar documents
- Returns top-k most relevant chunks

### 3. **AI Debate Generation**
- Gemini AI generates two-side debate
- Uses retrieved documents as context
- Cites real cases and constitutional articles

### 4. **Display**
- Prosecution arguments
- Defense arguments
- Supporting case law
- Constitutional references

---

## 💾 System Requirements

- **Python:** 3.11+ (3.13 works but may have library compatibility issues)
- **RAM:** 8GB minimum, 16GB recommended
- **Storage:** ~2GB for full dataset + index
- **CPU:** Multi-core recommended for faster embedding generation

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| **Total Documents** | 75,741 chunks |
| **Index Size** | ~500 MB |
| **Query Time** | ~100-200ms |
| **Embedding Dimension** | 384 |
| **Model** | all-MiniLM-L6-v2 |

---

## 🔐 Privacy & Security

- ✅ **All data stored locally** (no cloud)
- ✅ **Embeddings generated locally** (no API calls)
- ✅ **No user data collection**
- ⚠️ **Gemini API** used only for debate generation (text sent to Google)

---

## 🚧 Known Issues

### Python 3.13 Compatibility
- Some ML libraries (torch, transformers) may have issues
- **Workaround:** Use Python 3.11 if problems occur
- Current system: Working with HuggingFace embeddings

---

## 📝 Future Enhancements

### Phase 1: Data Expansion ✅ (Current)
- [x] Complete Constitution (395 articles)
- [x] 1,919 real cases
- [ ] Target: 50,000 cases (in progress)

### Phase 2: Advanced Features
- [ ] Multi-language support (Hindi, regional languages)
- [ ] Case summarization
- [ ] Legal citation graph
- [ ] Precedent tracking

### Phase 3: Production Ready
- [ ] User authentication
- [ ] Query history
- [ ] Export to PDF
- [ ] API endpoints

---

## 📞 Support

**Issues?**
- Check `.env` configuration
- Verify FAISS index exists: `data/vector_store/faiss_index/`
- Test embeddings: `python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"`
- Rebuild index: `python build_with_constitution.py`

---

## 🙏 Credits

- **Legal Data:** Indian Kanoon (indiankanoon.org)
- **Constitution:** Official Government of India sources
- **Embeddings:** HuggingFace Sentence Transformers
- **Vector Store:** FAISS by Meta AI
- **AI Model:** Google Gemini
- **Framework:** LangChain + Streamlit

---

## 📄 License

Educational and research purposes. Legal data sourced from public government databases.

---

**Last Updated:** November 6, 2025  
**Version:** 2.0 (Complete Constitution + Semantic Embeddings)
