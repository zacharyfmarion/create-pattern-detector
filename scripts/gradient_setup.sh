#!/bin/bash
# Setup script for Paperspace Gradient
# Run this after starting your notebook/instance

set -e

echo "=== Setting up Crease Pattern Detector ==="

# Install dependencies
echo "Installing dependencies..."
pip install -q -r requirements.txt

# Verify GPU
echo ""
echo "=== GPU Info ==="
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}'); print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB' if torch.cuda.is_available() else '')"

# Quick sanity check
echo ""
echo "=== Running Quick Sanity Check ==="
python scripts/sanity_check.py --fold-dir data/output/synthetic/raw/tier-a --quick

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "To start training:"
echo "  python scripts/train.py --fold-dir data/output/synthetic/raw/tier-a --image-size 512 --epochs 5 --batch-size 8"
