# Presentation Day Quick Guide

## Pre-Presentation Setup (Do this before your presentation)

### 1. Start Ollama Service
```bash
ollama serve
```
**Keep this terminal open** - Ollama must be running for the demo to work.

### 2. Launch the Demo
In a **new terminal**:
```bash
cd /home/ezhucs1/detection_ML/rag_demo
source venv/bin/activate
streamlit run web_demo.py
```

### 3. Verify Everything Works
- Open browser to `http://localhost:8501`
- Check sidebar shows "RAG Service: Active"
- Try asking a test question

## During Presentation

### Quick Demo Flow

1. **Show the Interface**
   - Clean, professional Q&A interface
   - Sidebar shows system status

2. **Ask Example Questions:**
   ```
   What are the characteristics of DDoS attacks in the dataset?
   How many attack samples are in Monday's data?
   What is the average flow duration for benign traffic?
   What features are most important for detecting port scans?
   Compare the packet rates between attack and normal traffic
   ```

3. **Show Sources**
   - Click on "Sources" expander to show where data came from
   - Demonstrates transparency and traceability

### If Something Goes Wrong

**Ollama not connecting:**
- Check if `ollama serve` is running
- Restart: `pkill -f ollama && ollama serve`

**No data:**
- Click "Run Data Ingestion" in sidebar (if needed)
- Or run: `python web_demo.py --ingest`

**Streamlit not working:**
- Restart: `pkill -f streamlit && streamlit run web_demo.py`

## After Presentation

### Stop Services
```bash
./stop_services.sh
```

Or manually:
```bash
pkill -f streamlit
pkill -f ollama
```

## Key Points to Highlight

1. **Single File Solution** - Everything in `web_demo.py` for easy setup
2. **RAG Architecture** - Retrieves relevant data chunks, then generates answers
3. **Source Citations** - Shows exactly where information came from
4. **Free & Open Source** - ChromaDB, sentence-transformers, Phi-3
5. **Handles Both Training and Testing Data** - Comprehensive coverage

## Troubleshooting

- **"Collection not found"** → Run data ingestion (button in sidebar)
- **"Ollama connection failed"** → Start Ollama: `ollama serve`
- **"Model not found"** → Pull model: `ollama pull phi3:mini`
- **Port already in use** → Kill existing process: `./stop_services.sh`

