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

# Free space from failed/interrupted builds (common cause of "No space left on device")
RECLAIMABLE=$(docker system df --format '{{.Reclaimable}}' 2>/dev/null | head -1 || echo "")
if [ -n "$RECLAIMABLE" ]; then
    echo "Docker disk usage:"
    docker system df
    echo ""
    DANGLING=$(docker images -f "dangling=true" -q 2>/dev/null | wc -l | tr -d ' ')
    if [ "$DANGLING" -gt 0 ]; then
        echo "Cleaning $DANGLING dangling image layer(s) from previous builds..."
        docker image prune -f
        echo ""
    fi
fi

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

BUILT=0
IMAGE_NAME="retrieval-playground-retrieval-playground:latest"
if docker image inspect "$IMAGE_NAME" &>/dev/null; then
    echo "Found existing workshop image ($IMAGE_NAME)."
    read -p "Rebuild image? [y/N] " REBUILD
    if [[ ! "$REBUILD" =~ ^[Yy]$ ]]; then
        echo "Skipping build, using existing image."
        echo ""
    else
        echo "Building Docker image (this may take 5-10 minutes)..."
        echo ""
        docker-compose build
        BUILT=1
    fi
else
    echo "Building Docker image (this may take 5-10 minutes on first run)..."
    echo "Tip: the image is ~11 GB (PyTorch, Docling, sentence-transformers)."
    echo "If the build fails with 'No space left on device', run:"
    echo "  docker system prune -a"
    echo "and increase Docker Desktop disk limit (Settings → Resources)."
    echo ""
    docker-compose build
    BUILT=1
fi

if [ "$BUILT" -eq 1 ]; then
    echo ""
    echo "✅ Build complete!"
fi
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
