#!/bin/bash
# Retrieval Playground - Quick Start Script for Mac/Linux

set -e

echo ""
echo "🧩 Retrieval Playground - Docker Workshop Setup"
echo "================================================"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed!"
    echo ""
    echo "Please install Docker Desktop from:"
    echo "  Mac: https://docs.docker.com/desktop/install/mac-install/"
    echo "  Linux: https://docs.docker.com/desktop/install/linux-install/"
    echo ""
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo "❌ Docker is not running!"
    echo ""
    echo "Please start Docker Desktop and try again."
    echo ""
    exit 1
fi

echo "✅ Docker is installed and running"
echo ""

# Check for .env file
if [ ! -f .env ]; then
    echo "⚠️  .env file not found!"
    echo ""
    echo "Creating .env from template..."
    if [ -f .env.example ]; then
        cp .env.example .env
        echo "✅ .env file created"
        echo ""
        echo "⚠️  IMPORTANT: Edit the .env file and add your API keys!"
        echo "   Open .env in a text editor and replace the placeholder values."
        echo ""
        read -p "Press Enter after you've updated the .env file..."
    else
        echo "❌ .env.example not found. Creating basic template..."
        cat > .env << 'EOF'
GOOGLE_API_KEY=your_google_api_key_here
QDRANT_URL=your_qdrant_url_here
QDRANT_KEY=your_qdrant_key_here
EOF
        echo "✅ .env file created"
        echo ""
        echo "⚠️  IMPORTANT: Edit the .env file and add your API keys!"
        echo ""
        exit 1
    fi
fi

echo "Building Docker image (this may take 5-10 minutes on first run)..."
echo ""
docker-compose build

echo ""
echo "✅ Build complete!"
echo ""
echo "Starting Jupyter Notebook server..."
echo ""
docker-compose up -d

echo ""
echo "✅ Jupyter Notebook is running!"
echo ""
echo "================================================"
echo "📝 Access your notebooks at:"
echo ""
echo "   http://localhost:8888"
echo ""
echo "================================================"
echo ""
echo "📚 Navigate to: retrieval_playground/tutorial/"
echo ""
echo "💡 Useful commands:"
echo "   Stop:     docker-compose down"
echo "   Restart:  docker-compose restart"
echo "   Logs:     docker-compose logs -f"
echo ""
