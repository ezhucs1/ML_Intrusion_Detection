# 🚀 Quick Start Guide

## Step 1: Create and Activate Virtual Environment

```bash
cd rag_demo

# Create virtual environment (if not already created)
python3 -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

## Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 3: Install Ollama & Phi-3

### Install Ollama:
- **macOS/Linux**: `curl -fsSL https://ollama.ai/install.sh | sh`
- **Windows**: Download from https://ollama.ai/download

### Pull Phi-3 Model:
```bash
ollama pull phi3:mini
```

## Step 4: Ingest Your Data

```bash
python src/rag_data_ingestion.py
```

This will:
- Load CSV files from `../CSE543_Group1/data_original/`
- Create embeddings (takes 5-15 minutes)
- Store in vector database

## Step 5: Launch Demo

```bash
streamlit run src/web_demo_rag.py
```

Open http://localhost:8501 in your browser.

## Step 6: Ask Questions!

Try these example questions:
- "What are the characteristics of DDoS attacks?"
- "How many attack samples are in Monday's data?"
- "What is the average flow duration for benign traffic?"

---

**Troubleshooting:**
- If "Collection not found": Run step 3 again
- If "Ollama error": Make sure Ollama is running (`ollama serve`)
- If "No CSV files": Check that files exist in `../CSE543_Group1/data_original/`

