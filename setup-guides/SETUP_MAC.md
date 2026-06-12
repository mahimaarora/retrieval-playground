# 🍎 Mac Setup Guide

Simple 3-step setup for the Retrieval Playground workshop.

---

## Step 1: Install Docker

### Option A: Using Homebrew (Recommended)

```bash
# Install Docker and Docker Compose
brew install docker docker-compose

# Install Colima (lightweight Docker runtime for Mac)
brew install colima

# Start Colima
colima start

# Verify installation
docker --version
docker-compose --version
```

**Don't have Homebrew?** Install it first:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### Option B: Docker Desktop (GUI Alternative)

1. Visit: https://www.docker.com/products/docker-desktop/
2. Download for Mac (choose Apple Silicon or Intel)
3. Install and open Docker Desktop
4. Wait for Docker to start (whale icon in menu bar)

---

## Step 2: Get Workshop Files & Add API Keys

1. **Download the repository:**
   ```bash
   # Option A: Download ZIP from GitHub
   # Visit: https://github.com/mahimaarora/retrieval-playground
   # Click "Code" → "Download ZIP" → Extract it
   
   # Option B: Clone with git
   git clone https://github.com/mahimaarora/retrieval-playground.git
   cd retrieval-playground
   ```

2. **Create .env file with your API keys:**
   ```bash
   # Copy the example file
   cp .env.example .env
   
   # Open in text editor
   open -e .env
   ```

3. **Add your keys** (provided by instructor):
   ```
   GOOGLE_API_KEY=your_key_here
   QDRANT_URL=your_url_here
   QDRANT_KEY=your_key_here
   ```
   
   Save and close the file.

---

## Step 3: Start the Workshop

1. **Open Terminal in the retrieval-playground folder**

2. **Run the start script:**
   ```bash
   ./start-workshop.sh
   ```

3. **Wait 5-10 minutes for first-time setup**

4. **Open your browser to:**
   ```
   http://localhost:8888
   ```

5. **Navigate to:** `retrieval_playground/tutorial/`

---

## ✅ You're Done!

Start with notebook: `1A_Pre_Chunking_Methods.ipynb`

---

## Common Commands

```bash
# Stop the workshop
docker-compose down

# Start again
./start-workshop.sh

# View logs if something goes wrong
docker-compose logs -f
```

---

## Need Help?

- **Docker not starting?** Open Docker Desktop app first
- **Port 8888 busy?** Stop other Jupyter notebooks or change port in `docker-compose.yml`
- **Script permission error?** Run: `chmod +x start-workshop.sh`

Ask your instructor for help! 🙋
