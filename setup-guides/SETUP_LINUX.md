# Linux Setup Guide

Simple 3-step setup for the Retrieval Playground workshop.

---

## Prerequisites

- **20 GB+ free disk space** for Docker (the workshop image is ~11 GB)
- Internet connection
- API keys from your instructor (or create your own — see [setup-guides/README.md](README.md))

---

## Step 1: Install Docker

### Ubuntu/Debian

```bash
# Update package list
sudo apt-get update

# Install prerequisites
sudo apt-get install -y ca-certificates curl gnupg lsb-release

# Add Docker's official GPG key
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Set up repository
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine and Compose plugin
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Start Docker service
sudo systemctl start docker
sudo systemctl enable docker

# Allow running Docker without sudo
sudo usermod -aG docker $USER
newgrp docker   # apply group now (or log out and back in)
```

### Other Linux distros

Follow the official guide: [Docker Engine install](https://docs.docker.com/engine/install/)

### Verify installation

```bash
docker --version
docker compose version
docker run hello-world
```

---

## Step 2: Get Workshop Files & Add API Keys

```bash
git clone https://github.com/mahimaarora/retrieval-playground.git
cd retrieval-playground

cp .env.example .env
# Edit with your preferred editor (nano, vim, gedit, etc.)
nano .env
# or
vim .env
# or
gedit .env
```

**Option A: Use Instructor's Qdrant (Read-only, Pre-ingested Data)**

Instructor will provide credentials for a shared Qdrant cluster with pre-ingested data:
```env
GOOGLE_API_KEY=your_key_here
QDRANT_URL=instructor_provided_url
QDRANT_KEY=instructor_provided_key
```

**Option B: Create Your Own Qdrant (For Ingestion Practice)**

If you want to try document ingestion yourself:

1. Visit https://qdrant.tech and sign up
2. Click "Create Cluster"
3. Enter cluster name: `scipy_workshop`
4. Select a region (any region)
5. Click "Create Free Cluster"
6. Copy your **Cluster Endpoint** (QDRANT_URL)
7. Copy your **API Key** (QDRANT_KEY)
8. Add to .env:
   ```env
   GOOGLE_API_KEY=your_key_here
   QDRANT_URL=your_cluster_endpoint
   QDRANT_KEY=your_api_key
   ```

Save and close the file.

> **Note:** `./start-workshop.sh` can also create `.env` from `.env.example` if you skip this step — just edit the file before the container starts.

---

## Step 3: Start the Workshop

```bash
./start-workshop.sh
```

- **First run:** builds the image (~5–10 minutes). Say **N** to "Rebuild image?" on later runs unless you changed `Dockerfile` or `requirements.txt`.
- **Open:** [http://localhost:8888](http://localhost:8888)
- **Notebooks:** `retrieval_playground/tutorial/`
- **Optional check:** run `setup-guides/verify_setup.ipynb`

Start with: `1A_Pre_Chunking_Methods.ipynb`

---

## Common Commands

```bash
# Stop the workshop
docker compose down

# Start again (no rebuild needed for notebook/code/data changes)
./start-workshop.sh

# Or restart the running container
docker compose restart

# View logs
docker compose logs -f

# Check if container is running
docker ps
```

### When to rebuild vs restart

| You changed… | Action |
| --- | --- |
| Notebooks, `src/`, `utils/`, `data/` | `docker compose up -d` or `./start-workshop.sh` (choose **N** to rebuild) |
| `Dockerfile`, `requirements.txt` | `./start-workshop.sh` and choose **Y** to rebuild |

---

## Troubleshooting

| Issue | Solution |
| --- | --- |
| Permission denied on `docker` | Run `newgrp docker` or log out and back in after `usermod` |
| `docker compose: command not found` | Install the plugin: `sudo apt-get install docker-compose-plugin` |
| No space left on device | `docker system prune -a` then retry; ensure 20 GB+ free for Docker |
| Port 8888 busy | `pkill jupyter` or change the port in `docker-compose.yml` |
| `./start-workshop.sh: Permission denied` | `chmod +x start-workshop.sh` |

Ask your instructor for help!
