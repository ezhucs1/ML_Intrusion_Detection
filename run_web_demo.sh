#!/bin/bash
# Run the Interactive Web Demo for Intrusion Detection System

echo "=========================================="
echo "  Intrusion Detection System - Web Demo"
echo "=========================================="
echo ""

# Check if virtual environment exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
else
    echo "⚠️  Virtual environment not found. Creating one..."
    python3 -m venv .venv
    source .venv/bin/activate
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Check if model exists
if [ ! -f "models/intrusion_detection_model.pkl" ]; then
    echo ""
    echo "⚠️  WARNING: Model not found!"
    echo "Please train a model first by running:"
    echo "  python src/train_model.py"
    echo ""
    read -p "Do you want to continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "Starting Streamlit web demo..."
echo "The demo will open in your browser at: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run streamlit
streamlit run src/web_demo.py

