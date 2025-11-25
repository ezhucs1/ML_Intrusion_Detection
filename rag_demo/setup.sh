#!/bin/bash

# RAG Demo Setup Script

echo "=========================================="
echo "RAG Demo Setup"
echo "=========================================="

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8+"
    exit 1
fi

echo "✓ Python found: $(python3 --version)"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Check Ollama
if ! command -v ollama &> /dev/null; then
    echo ""
    echo "⚠️  Ollama not found. Please install from: https://ollama.ai"
    echo "   After installing, run: ollama pull phi3:mini"
else
    echo "✓ Ollama found"
    
    # Check if phi3:mini is available
    if ollama list | grep -q "phi3:mini"; then
        echo "✓ Phi-3 model found"
    else
        echo ""
        echo "📥 Pulling Phi-3 model (this may take a few minutes)..."
        ollama pull phi3:mini
    fi
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Virtual environment is activated."
echo ""
echo "Next steps:"
echo "1. Activate venv (if not already): source venv/bin/activate"
echo "2. Run data ingestion: python src/rag_data_ingestion.py"
echo "3. Launch demo: streamlit run src/web_demo_rag.py"
echo ""
echo "To deactivate venv later: deactivate"
echo ""

