#!/bin/bash
# Quick start script for System Architecture Dashboard

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║   System Architecture Dashboard - Quick Start                 ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install -q -r requirements.txt

# Copy environment file if not exists
if [ ! -f ".env" ]; then
    echo "⚙️  Creating .env file..."
    cp .env.example .env
    echo "   Edit .env to customize settings"
fi

# Start server
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║              Starting Backend Server...                       ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "✓ Backend running at http://localhost:5000"
echo "✓ API health check: http://localhost:5000/api/health"
echo ""
echo "Open dashboard at:"
echo "  file://$(pwd)/templates/index.html"
echo ""
echo "Or visit:"
echo "  http://localhost:5000/templates/index.html"
echo ""
echo "Press Ctrl+C to stop server"
echo ""

python3 -m backend.app
