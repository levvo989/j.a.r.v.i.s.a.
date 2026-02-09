#!/bin/bash
echo "Activating Python environment..."
source venv/bin/activate

echo "Installing required models..."
python -c "import subprocess; subprocess.run(['ollama','pull','llama3'])"

echo "Starting Local AI Server..."
python app.py
