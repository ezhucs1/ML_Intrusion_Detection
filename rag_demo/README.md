# RAG-Powered Intrusion Detection Demo

A Retrieval-Augmented Generation (RAG) system for querying network intrusion detection datasets using Llama 3.2 3B LLM.

## Overview

This demo allows you to ask natural language questions about your intrusion detection CSV files and get AI-powered answers based on the actual data. The system:

- Retrieves relevant information from CSV files in `data_original/` and `Testing_data/`
- Generates natural language answers using Llama 3.2 3B LLM
- Shows source citations from the dataset

## Quick Start

### 1. Prerequisites

- Python 3.8+
- Access to CSV files in `../CSE543_Group1/data_original/` and `../CSE543_Group1/Testing_data/`
- 4GB+ RAM (for embeddings and LLM)

### 2. Installation

```bash
# Navigate to rag_demo directory
cd rag_demo

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Install Ollama (choose your platform)
# macOS/Linux:
curl -fsSL https://ollama.ai/install.sh | sh

# Windows: Download from https://ollama.ai/download

# Pull Llama 3.2 3B model (recommended for RAG)
ollama pull llama3.2:3b
```

### 3. Data Ingestion (Create Vector Embeddings)

**IMPORTANT: Run this command to convert CSV files into vector embeddings before using the demo.**

```bash
cd /home/ezhucs1/detection_ML/rag_demo
source venv/bin/activate
python web_demo.py --ingest
```

**What this does:**
- Loads CSV files from `../CSE543_Group1/data_original/` (training data)
- Loads CSV files from `../CSE543_Group1/Testing_data/` (test data)
- Creates embeddings using sentence-transformers model
- Stores data chunks in ChromaDB vector database
- Takes 5-15 minutes depending on data size

**Note:** You only need to run this once. The vector database is stored in `chroma_db/` and persists between sessions. If you've already run ingestion, you can skip this step.

### 4. Start Ollama Service

**IMPORTANT: Ollama must be running for the demo to work.**

In a separate terminal, start Ollama:
```bash
ollama serve
```

**Keep this terminal open** - Ollama must stay running.

### 5. Launch Demo

In a new terminal:
```bash
cd /home/ezhucs1/detection_ML/rag_demo
source venv/bin/activate
streamlit run web_demo.py
```

The demo will open at `http://localhost:8501`

## Usage

### Presentation Day Quick Commands

**1. Activate environment:**
```bash
cd /home/ezhucs1/detection_ML/rag_demo
source venv/bin/activate
```

**2. Start Ollama (Terminal 1):**
```bash
ollama serve
```

**3. Create vector embeddings (if not done already):**
```bash
python web_demo.py --ingest
```

**4. Launch demo (Terminal 2):**
```bash
cd /home/ezhucs1/detection_ML/rag_demo
source venv/bin/activate
streamlit run web_demo.py
```

### Example Questions

You can ask questions like:

```
What are the characteristics of DDoS attacks in the dataset?
How many attack samples are in Monday's data?
What is the average flow duration for benign traffic?
What features are most important for detecting port scans?
Compare the packet rates between attack and normal traffic
What are the statistics for port scan attacks?
Show me the distribution of attack types in the testing data
What is the average SYN flag count for different attack types?
```

### How It Works

1. User asks a question in natural language
2. System retrieves relevant chunks from CSV files using semantic search
3. LLM generates an answer based on retrieved context
4. Sources are shown so you can verify the information

## Project Structure

```
rag_demo/
├── web_demo.py                   # Main application (all-in-one)
├── chroma_db/                    # Vector database (created after ingestion)
├── requirements.txt              # Python dependencies
├── setup.sh                      # Setup script
├── stop_services.sh              # Stop services script
└── README.md                     # This file
```

**Note:** All functionality is consolidated in `web_demo.py` for easy setup and presentation.

## Stopping Services

### Stop Streamlit

If Streamlit is running, stop it with:
```bash
# Find the process
lsof -i :8501

# Kill the process (replace PID with actual process ID)
kill -9 <PID>

# Or use pkill
pkill -f streamlit
```

### Stop Ollama

If Ollama is running, stop it with:
```bash
# Find the process
lsof -i :11434

# Kill the process (replace PID with actual process ID)
kill -9 <PID>

# Or use pkill
pkill -f ollama
```

### Stop All Services at Once

```bash
# Kill both Streamlit and Ollama
pkill -f streamlit
pkill -f ollama

# Or find and kill by port
fuser -k 8501/tcp  # Streamlit
fuser -k 11434/tcp # Ollama
```

## Configuration

### Change Data Directory

Edit `web_demo.py` and modify the `ingest_data` function call:

```python
data_dir = "/path/to/your/csv/files"
testing_data_dir = "/path/to/testing/files"
```

### Adjust Chunk Size

In `web_demo.py`, modify the `chunk_csv_data` function:

```python
chunks = chunk_csv_data(df, filename, chunk_size=100)  # Change 100 to desired size
```

### Use Different LLM

In `web_demo.py`, modify the `RAGService` initialization:

```python
rag = RAGService(
    llm_model="llama3.2:3b",  # or other Ollama models like "qwen2.5:3b", "gemma:2b"
    use_ollama=True
)
```

## Troubleshooting

### "Collection not found" Error

Run data ingestion first:
- Use the "Run Data Ingestion" button in the sidebar, or
- Run: `python web_demo.py --ingest`

### "Ollama model not found"

Pull the model:
```bash
ollama pull llama3.2:3b
```

### "Ollama connection failed"

Make sure Ollama is running:
```bash
ollama serve
```

### "Ollama connection failed"

Make sure Ollama is running:
```bash
ollama serve
```

### "No CSV files found"

Ensure CSV files are in `../CSE543_Group1/data_original/` and `../CSE543_Group1/Testing_data/` or specify custom paths.

### Slow Responses

- Reduce `max_rows_per_file` in ingestion
- Use smaller chunk size
- Reduce `top_k` in RAG service

## Features

- Free and open-source (ChromaDB, sentence-transformers, Llama 3.2 3B)
- Fast responses with Llama 3.2 3B (optimized for RAG)
- Context-aware answers from your actual data
- Reduced hallucination with improved prompt engineering
- Source citations for transparency
- Interactive Streamlit interface
- Supports both training and testing data

## Notes

- First ingestion may take 10-15 minutes
- Vector database is stored locally in `chroma_db/`
- Default processes 10,000 rows per file (adjustable)
- Requires Ollama running for LLM inference
- Both `data_original/` and `Testing_data/` folders are included by default

## Related

- Main project: `../CSE543_Group1/`
- Dataset: CIC-IDS-2017
- Ollama: https://ollama.ai
- ChromaDB: https://www.trychroma.com/
