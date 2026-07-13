# 🧩 Retrieval Playground - Complete Setup Guide

Welcome! This guide will help you set up everything needed for the workshop.

---

## Prerequisites

Before starting, make sure you have:

- Computer with 8GB+ free disk space
- Internet connection
- Admin/sudo privileges (to install Docker)

---

## Setup Steps

### Step 1: Choose Your Setup Method

**Recommended: Docker Setup (Easiest)**

- **🍎 [Mac Setup Guide](SETUP_MAC.md)** - macOS users
- **🪟 [Windows Setup Guide](SETUP_WINDOWS.md)** - Windows 10/11 users
- **🐧 [Linux Setup Guide](SETUP_LINUX.md)** - Ubuntu, Fedora, Debian users

**Alternative: Manual Setup (Advanced)**

- **⚙️ [Manual Setup Without Docker](SETUP_WITHOUT_DOCKER.md)** - For advanced users

Each guide covers:

1. Installation (Docker or Python setup)
2. Getting workshop files
3. Configuring API keys
4. Starting the environment

---

### Step 2: Get Your Google Gemini API Key

1. **Visit:** [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. **Sign in** with your Google account
3. **Click:** "Get API Key" or "Create API Key"
4. **Create key in new project** or select existing project
5. **Copy the API key** (starts with `AIza...`)

> 💡 **Free tier:** Google Gemini offers free API access with generous limits for testing.

---

### Step 3: Configure Environment Variables

After completing the Docker setup from your OS guide:

1. **Navigate to the retrieval-playground folder**
2. **Copy the example file:**
  ```bash
   cp .env.example .env
  ```
3. **Edit the .env file:**
  - **Mac/Linux:** `nano .env` or `open -e .env`
  - **Windows:** `notepad .env`
4. **Add your API key:**
  ```env
   GOOGLE_API_KEY=your_gemini_api_key
  ```
5. **Save and close the file**

---

## Verify Your Setup

After completing all steps, verify everything works:

1. **Start the workshop environment** (from your OS guide)
2. **Open:** [http://localhost:8888](http://localhost:8888)
3. **Navigate to:** `setup-guides/`
4. **Open:** `verify_setup.ipynb`
5. **Run all cells** - they should all pass ✅

This notebook checks:

- All required packages are installed
- API keys are configured correctly
- Connections to services work

---

## Start Learning

Once verification passes:

1. **Navigate to:** `retrieval_playground/tutorial/`
2. **Start with:** `1A_Pre_Chunking_Methods.ipynb`
3. **Follow the notebooks in order:**
  - `1A_Pre_Chunking_Methods.ipynb` - Document chunking strategies
  - `1B_Pre_Query_Methods.ipynb` - Query enhancement
  - `2A_Basic_Mid_Retrieval_Methods.ipynb` - Basic retrieval methods
  - `2B_Advanced_Mid_Retrieval_Methods.ipynb` - Advanced retrieval methods
  - `3_Post_Retrieval.ipynb` - Post-processing strategies
  - `4_Evaluation.ipynb` - Evaluation metrics
  - `5_Agentic_RAG.ipynb` - Agentic RAG

---

## Common Commands

```bash
# Stop the workshop
docker compose down

# Start the workshop
./start-workshop.sh           # Mac/Linux
start-workshop.bat            # Windows

# View logs
docker compose logs -f

# Restart after notebook/code/data changes (no rebuild needed)
docker compose restart
```

---

## 🆘 Troubleshooting


| Issue                      | Solution                                       |
| -------------------------- | ---------------------------------------------- |
| Import errors in notebooks | Run `verify_setup.ipynb` to check installation |
| "GOOGLE_API_KEY not found" | Check `.env` file has correct key              |
| Docker not starting        | Make sure Docker service is running            |
| Port 8888 in use           | Stop other Jupyter instances or change port    |
| No space left on device    | `docker system prune -a` — image needs ~11 GB  |
| `docker compose` not found | Install compose plugin (see SETUP_LINUX.md)    |


## Need Help?

- **During workshop:** Ask your instructor
- **Setup issues:** Check your OS-specific guide
- **GitHub issues:** [https://github.com/mahimaarora/retrieval-playground/issues](https://github.com/mahimaarora/retrieval-playground/issues)

---

**Happy Learning! 🎓**