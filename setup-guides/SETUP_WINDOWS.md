# Windows Setup Guide

Simple 3-step setup for the Retrieval Playground workshop.

---

## Step 1: Install Docker

### Option A: Using WSL2 + Docker CLI (Recommended)

1. **Install WSL2:**
  ```powershell
   # Run in PowerShell as Administrator
   wsl --install

   # Restart your computer
  ```
2. **After restart, open Ubuntu from Start menu**
3. **Install Docker in WSL2:**
  ```bash
   # Update packages
   sudo apt-get update

   # Install Docker
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh

   # Add your user to docker group
   sudo usermod -aG docker $USER

   # Start Docker service
   sudo service docker start

   # Verify
   docker --version
  ```
4. **Install Docker Compose:**
  ```bash
   sudo apt-get install docker-compose-plugin
  ```

### Option B: Docker Desktop (GUI Alternative)

1. Visit: [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)
2. Download Docker Desktop for Windows
3. Run installer (enable WSL 2)
4. Restart computer
5. Start Docker Desktop

---

## Step 2: Get Workshop Files & Add API Keys

1. **Download the repository:**
  ```
   Option A: Download ZIP
   - Visit: https://github.com/mahimaarora/retrieval-playground
   - Click "Code" → "Download ZIP"
   - Right-click ZIP → "Extract All"

   Option B: Clone with git
   - Open Command Prompt or PowerShell
   - Run: git clone https://github.com/mahimaarora/retrieval-playground.git
  ```
2. **Create .env file with your API keys:**
  - Open the `retrieval-playground` folder
  - Find `.env.example` file
  - Right-click → Copy
  - Right-click in folder → Paste
  - Rename the copy to `.env` (remove .example)
3. **Edit .env file:**
  - Right-click `.env` → Open with Notepad
  - Replace the placeholder values with keys from instructor:
  - Save and close

---

## Step 3: Start the Workshop

1. **Open the retrieval-playground folder**
2. **Double-click:** `start-workshop.bat`
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

Open Command Prompt or PowerShell in the folder:

```cmd
REM Stop the workshop
docker compose down

REM Start again
start-workshop.bat

REM View logs if something goes wrong
docker compose logs -f
```

---

## Need Help?

- **Docker not starting?** Open Docker Desktop app first
- **Port 8888 busy?** Stop other Jupyter notebooks or change port in `docker-compose.yml`
- **WSL 2 error?** Follow Docker's instructions to enable WSL 2

Ask your instructor for help! 🙋