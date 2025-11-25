#!/bin/bash
# Setup script for Intrusion Detection System

echo "=========================================="
echo "  Intrusion Detection System Setup"
echo "=========================================="

# Create virtual environment
echo ""
echo "1. Creating virtual environment..."
python3 -m venv .venv

# Activate virtual environment
echo "2. Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "3. Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "To activate the environment:"
echo "  source .venv/bin/activate"
echo ""
echo "To train the model:"
echo "  python src/train_model.py"
echo ""







