#!/bin/bash
# Run the complete intrusion detection pipeline

echo "=========================================="
echo "  Intrusion Detection System - Full Pipeline"
echo "=========================================="

# Activate virtual environment
source .venv/bin/activate

echo ""
echo "Step 1: Training Model"
echo "-----------------------------------"
python src/train_model.py

echo ""
echo "Step 2: Testing Model"
echo "-----------------------------------"
python src/test_model.py

echo ""
echo "=========================================="
echo "  Pipeline Complete!"
echo "=========================================="

