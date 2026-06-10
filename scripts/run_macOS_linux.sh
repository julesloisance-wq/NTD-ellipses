#!/bin/bash

echo "Starting the Ellipse Detection pipeline..."

# Check if virtual environment exists, create if not
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate the virtual environment
source venv/bin/activate

# Install required libraries
echo "Installing dependencies..."
python3 -m pip install --upgrade pip -q
pip install -r requirements.txt -q

# Run the main program
echo "Running the script..."
python3 main.py

# Keep the terminal open when finished
echo "Processing complete. Press any key to close..."
read -n 1