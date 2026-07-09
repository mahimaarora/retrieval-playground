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

### Step 2: Get API Keys (Required)

You need API keys to run the notebooks. Follow the instructions below:

#### 🔑 Google Gemini API Key (REQUIRED)

1. **Visit:** [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. **Sign in** with your Google account
3. **Click:** "Get API Key" or "Create API Key"
4. **Create key in new project** or select existing project
5. **Copy the API key** (starts with `AIza...`)
6. **Save it** - you'll add this to `.env` file later

> 💡 **Free tier:** Google Gemini offers free API access with generous limits for testing.

#### 🔑 OpenAI API Key (Optional Alternative)

If you prefer to use OpenAI instead of Gemini:

1. **Visit:** [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. **Sign in** or create an account
3. **Click:** "Create new secret key"
4. **Copy the key** (starts with `sk-...`)
5. **Save it** - you'll need this for `.env` file

> ⚠️ **Note:** OpenAI requires payment information on file.

---

### Step 3: Set Up Qdrant (Good to Have)

Qdrant is a vector database used in the workshop. Two options:

#### Option A: Use Instructor's Shared Instance (During Workshop)

Your instructor will provide:

- `QDRANT_URL` - Database URL
- `QDRANT_KEY` - Access key

Skip to Step 4 and use these values.

#### Option B: Create Your Own Free Qdrant Account

1. **Visit:** [https://cloud.qdrant.io/](https://cloud.qdrant.io/)
2. **Click:** "Get Started Free"
3. **Sign up** with email or GitHub
4. **Verify** your email
5. **Create a cluster:**
  - Click "Create Cluster"
  - Choose "Free Tier" (1GB, perfect for learning)
  - Select region closest to you
  - Click "Create"
6. **Get your credentials:**
  - Click on your cluster name
  - Copy **Cluster URL** (this is your `QDRANT_URL`)
  - Go to "API Keys" tab
  - Click "Create API Key"
  - Copy the key (this is your `QDRANT_KEY`)

> 💡 **Free tier:** Qdrant offers 1GB free forever, no credit card needed.

---

### Step 4: Configure Environment Variables

After completing the Docker setup from your OS guide, configure your API keys:

1. **Navigate to the retrieval-playground folder**
2. **Copy the example file:**
  ```bash
   cp .env.example .env
  ```
3. **Edit the .env file:**
  - **Mac/Linux:** `nano .env` or `open -e .env`
  - **Windows:** `notepad .env`
4. **Add your keys:**
  ```env
   # Required: Google Gemini API Key
   GOOGLE_API_KEY=AIza...your_actual_key_here

   # Required: Qdrant credentials (from instructor or your account)
   QDRANT_URL=https://your-cluster.cloud.qdrant.io
   QDRANT_KEY=your_qdrant_key_here
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
| "Cannot connect to Qdrant" | Verify `QDRANT_URL` and `QDRANT_KEY` in `.env` |
| Docker not starting        | Make sure Docker service is running            |
| Port 8888 in use           | Stop other Jupyter instances or change port    |
| No space left on device    | `docker system prune -a` — image needs ~11 GB  |
| `docker compose` not found | Install compose plugin (see SETUP_LINUX.md)    |


## Need Help?

- **During workshop:** Ask your instructor
- **Setup issues:** Check your OS-specific guide
- **API key issues:** Review Step 2 above
- **GitHub issues:** [https://github.com/mahimaarora/retrieval-playground/issues](https://github.com/mahimaarora/retrieval-playground/issues)

---

**Happy Learning! 🎓**