#!/bin/bash

# Grok-style Chat UI Launcher for DGX Spark TRT-LLM Cluster
# One-command setup: installs dependencies and starts the server

set -e

cd "$(dirname "$0")"

echo "🚀 DGX Spark Chat UI - Starting..."
echo ""

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8+"
    exit 1
fi

echo "📦 Installing Python dependencies..."
python3 -m pip install --break-system-packages fastapi uvicorn[standard] aiohttp python-multipart > /dev/null 2>&1 || {
    echo "⚠️  Trying alternative installation method..."
    python3 -m pip install fastapi uvicorn[standard] aiohttp python-multipart 2>&1 | grep -v "already satisfied" || true
}

echo ""
echo "✅ Dependencies installed"
echo ""
echo "🎨 Starting Chat Server..."
echo "   → Open: http://localhost:7860"
echo "   → TRT-LLM API: http://localhost:8355/v1"
echo ""
echo "   Press Ctrl+C to stop"
echo ""

cd chat
python3 -m uvicorn app:app --host 0.0.0.0 --port 7860 --reload
