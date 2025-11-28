# MajorLegal Project Submission Workflow

This guide outlines the complete workflow for running the MajorLegal system and generating the necessary artifacts for your project report submission.

## 1. Project Overview
MajorLegal is a Multi-Agent Legal Debate System powered by RAG (Retrieval-Augmented Generation). It simulates a legal debate between a Prosecution and Defense agent, moderated by a Judge agent, using real Indian case law for citations.

## 2. Setup & Execution
Before generating reports, ensure the system is running correctly.

### Prerequisites
- Python 3.10+
- Google API Key (Gemini)
- HuggingFace Token (optional, for gated models)

### Installation
```bash
pip install -r requirements.txt
```

### Running the Application
To start the interactive debate interface:
```bash
streamlit run app.py
```

## 3. Generating Project Reports
For a complete project submission, you need two types of reports:
1.  **Technical Performance Report** (Quantitative)
2.  **Case Study Report** (Qualitative)

### A. Technical Performance Report (Benchmark)
This report evaluates the accuracy of the RAG system using a test set of legal questions.

**Steps:**
1.  Open a terminal.
2.  Run the benchmark script:
    ```bash
    python benchmark.py
    ```
3.  **Output:** This will generate a timestamped report file (e.g., `research_report_20251126_200939.md`) containing:
    - Precision/Recall/F1 Scores for citation retrieval.
    - Detailed analysis of test cases.
    - System configuration details.

### B. Case Study Report (Debate Simulation)
This report demonstrates the system's ability to handle complex legal arguments in a real-world scenario.

**Steps:**
1.  Run the app: `streamlit run app.py`
2.  Enter a legal case description (e.g., "A case involving Section 302 IPC where the accused claims self-defense...").
3.  Click **"Start Legal Debate"**.
4.  Wait for the debate to complete (Prosecution -> Defense -> Moderator).
5.  In the sidebar, look for the **"📄 Download Debate Report"** button.
6.  Click it to download `case_report.md`.

## 4. Final Project Report Structure
When compiling your final PDF/Word document for submission, structure it as follows:

### 1. Title Page
- Project Name: MajorLegal
- Your Name/Team Details
- Date

### 2. Executive Summary
- Brief overview of the problem (Legal Research Automation).
- Solution description (Multi-Agent RAG System).
- Key results (e.g., "Achieved 85% citation accuracy").

### 3. System Architecture
- **Embeddings:** InLegalBERT (or Google Embedding-004).
- **Vector Store:** FAISS.
- **LLM:** Google Gemini 2.5 Flash.
- **Agent Workflow:** Prosecution <-> Defense -> Moderator.

### 4. Technical Evaluation
- *Copy contents from `research_report.md` here.*
- Include the Precision/Recall tables.

### 5. Case Study
- *Copy contents from `case_report.md` here.*
- Show a specific example of the system in action.

### 6. Conclusion & Future Work
- Summary of achievements.
- Potential improvements (e.g., switching to Voyage AI embeddings, expanding dataset).

## 5. Submission Checklist
- [ ] Codebase (zipped or GitHub link)
- [ ] `requirements.txt`
- [ ] Final Report (PDF) containing Technical & Case Study sections
- [ ] Video Demo (optional, screen record the Streamlit app)

## Latest Updates (2025-11-28)
- **Comprehensive Benchmark Suite**: Implemented a full benchmark suite with 100 diverse Indian legal scenarios.
- **Baselines**: Added comparisons for Vanilla LLM (Gemini 2.0 Flash), Simple RAG, and MajorLegal.
- **Robust Logging**: Benchmark results are now timestamped and saved to ench_result.txt.
- **Refactoring**: enchmark.py now supports batch processing, CLI arguments, and robust error handling.
