#!/bin/bash

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install libraries
pip install --upgrade pip
pip install scikit-learn pandas matplotlib yfinance

# Run the script
python pca_trader.py
