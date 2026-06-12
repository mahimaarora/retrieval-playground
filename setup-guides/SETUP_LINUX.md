# Linux Setup Guide

Simple 3-step setup for the Retrieval Playground workshop.

---

## Step 1: Install Docker CLI

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

# Install Docker Engine and Docker Compose
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Start Docker service
sudo systemctl start docker
sudo systemctl enable docker

# Add your user to docker group (no sudo needed)
sudo usermod -aG docker $USER

# Log out and log back in for changes to take effect
```

### Fedora/RHEL/CentOS

```bash
# Install Docker Engine
sudo dnf install -y dnf-plugins-core
sudo dnf config-manager --add-repo https://download.docker.com/linux/fedora/docker-ce.repo
sudo dnf install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Start Docker service
sudo systemctl start docker
sudo systemctl enable docker

# Add your user to docker group
sudo usermod -aG docker $USER

# Log out and log back in
```

### Verify Installation

```bash
# Check Docker version
docker --version

# Check Docker Compose version  
docker compose version

# Test Docker is running
docker run hello-world
```

---

## Step 2: Get Workshop Files & Add API Keys

1. **Download the repository:**
  ```bash
   # Clone the repository
   git clone https://github.com/mahimaarora/retrieval-playground.git
   cd retrieval-playground
  ```
2. **Create .env file with your API keys:**
  ```bash
   # Copy the example file
   cp .env.example .env

   # Edit with your preferred editor (nano, vim, gedit, etc.)
   nano .env
   # or
   vim .env
   # or
   gedit .env
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

1. **Make the start script executable:**
  ```bash
   chmod +x start-workshop.sh
  ```
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

# Check if container is running
docker ps
```

---

## Need Help?

- **Permission denied?** Make sure you logged out and back in after adding user to docker group
- **Docker not starting?** Run: `sudo systemctl status docker`
- **Port 8888 busy?** Stop other Jupyter notebooks: `pkill jupyter`

Ask your instructor for help! 🙋