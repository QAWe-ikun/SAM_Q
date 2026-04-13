#!/bin/bash
# Download and preprocess dataset for SAM-Q
# Usage: ./scripts/download_data.sh

set -e

echo "========================================="
echo "SAM-Q Dataset Download"
echo "========================================="

# Create data directory
mkdir -p data

# Example: Download from a hypothetical URL
# Uncomment and modify these lines for actual dataset
# wget https://example.com/samq_dataset.zip -O data/dataset.zip
# unzip data/dataset.zip -d data/
# rm data/dataset.zip

echo ""
echo "Dataset directory structure:"
echo "  data/"
echo "  ├── annotations.json"
echo "  ├── plane_images/"
echo "  ├── object_images/"
echo "  └── masks/"
echo ""
echo "Please place your dataset in the data/ directory."
echo "See README.md for dataset format details."
