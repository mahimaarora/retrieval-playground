# ⚙️ Manual Setup (Without Docker)

For advanced users who prefer not to use Docker.

---

## Prerequisites

- Python 3.12 or higher
- Git
- System dependencies: `tesseract-ocr`, `poppler-utils` (for PDF processing)

---

## Step 1: Install System Dependencies

### Mac

```bash
# Using Homebrew
brew install tesseract poppler
```

### Linux (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr poppler-utils
```

### Linux (Fedora/RHEL)

```bash
sudo dnf install -y tesseract poppler-utils
```

### Windows

- **Tesseract:** Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
- **Poppler:** Download from [GitHub](https://github.com/oschwartz10612/poppler-windows/releases/)
- Add both to your PATH environment variable

---

## Step 2: Clone the Repository

```bash
git clone https://github.com/mahimaarora/retrieval-playground.git
cd retrieval-playground
```

---

## Step 3: Set Up Python Virtual Environment

```bash
# Create virtual environment
python3.12 -m venv venv

# Activate virtual environment
# Mac/Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

---

## Step 4: Install Python Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install the package in editable mode
pip install -e .
```

This will install all required packages from `requirements.txt`.

---

## Step 5: Configure Environment Variables

```bash
# Copy the example file
cp .env.example .env

# Edit .env file with your preferred editor
# Mac/Linux:
nano .env
# or
vim .env

# Windows:
notepad .env
```

Add your API keys:

**Use Instructor's Qdrant (Read-only, Pre-ingested Data)**

Instructor will provide credentials for a shared Qdrant cluster with pre-ingested data:

```env
# Required: Google Gemini API Key
GOOGLE_API_KEY=your_key_here

# Required: Qdrant credentials (provided by instructor)
QDRANT_URL=instructor_provided_url
QDRANT_KEY=instructor_provided_key
```
---

## Step 6: Set Up Jupyter Kernel

```bash
python -m ipykernel install --user --name=venv --display-name "scipy_tutorial"
```

---

## Step 7: Launch Jupyter Notebook

```bash
jupyter notebook
```

This will open Jupyter in your browser.

---

## Step 8: Select the Correct Kernel

1. Open any tutorial notebook
2. Go to **Kernel** → **Change kernel** → **scipy_tutorial**
3. Start with: `retrieval_playground/tutorial/1A_Pre_Chunking_Methods.ipynb`

---

## ✅ Verify Installation

Navigate to `setup-guides/verify_setup.ipynb` and run all cells to verify everything is working.

---

## 📚 Start Learning

Once verified:

1. Navigate to: `retrieval_playground/tutorial/`
2. Start with: `1A_Pre_Chunking_Methods.ipynb`
3. Follow notebooks in order

---

## 🛠️ Common Commands

```bash
# Activate virtual environment
source venv/bin/activate           # Mac/Linux
venv\Scripts\activate              # Windows

# Deactivate virtual environment
deactivate

# Update dependencies
pip install --upgrade -e .

# Start Jupyter
jupyter notebook
```

---

## 🆘 Troubleshooting

### Import Errors

```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Reinstall package
pip install -e .
```

### Missing System Dependencies

```bash
# Mac
brew install tesseract poppler

# Linux
sudo apt-get install tesseract-ocr poppler-utils
```

### Jupyter Kernel Not Found

```bash
# Reinstall kernel
python -m ipykernel install --user --name=venv --display-name "scipy_tutorial"

# List installed kernels
jupyter kernelspec list
```

---

## 💡 Why Use Docker Instead?

Docker setup is **recommended** because:

- ✅ No system dependency installation needed
- ✅ Consistent environment across all operating systems
- ✅ Faster setup (one command vs multiple steps)
- ✅ Easier troubleshooting

See the [Docker setup guides](README.md) for easier installation.

---

**Need help?** Ask your instructor or check the main [Setup Guide](README.md)
