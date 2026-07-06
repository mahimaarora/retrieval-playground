# SciPy Workshop Dataset: Recent Scientific Python Papers

**Curated for:** SciPy 2026 Workshop - "**Engineering Better Retrieval for RAG**"  
**Dataset Size:** 11 papers  
**Total Pages:** 392  
**Total Size:** 39.4 MB  
**Year Range:** 2023-2026  
**Focus:** Modern scientific Python tools, AI/ML for science, and cutting-edge research

---

## 📁 Dataset Structure

```
data/
├── DATASET_README.md                    # This file
├── workshop_metadata.json               # Paper metadata (title, pages, size, topic, year, link)
├── workshop_example_queries.json        # Example queries for each workshop section
│
└── workshop_data/                       # All 11 PDFs (39.4 MB total, 392 pages)
    ├── 2023_Dataframe_Libraries_Evaluation.pdf      (13 pages, 1.6 MB)
    ├── 2023_LLM_Python_Optimization.pdf             (20 pages, 0.8 MB)
    ├── 2024_GPU_CUDA_ML.pdf                         (106 pages, 1.1 MB)
    ├── 2024_Physics_Informed_Diffusion.pdf          (26 pages, 4.1 MB)
    ├── 2024_Quantum_GNN_Molecular.pdf               (20 pages, 1.3 MB)
    ├── 2025_Agent_Laboratory.pdf                    (56 pages, 3.1 MB)
    ├── 2025_AutoClimDS.pdf                          (7 pages, 6.1 MB)
    ├── 2025_FunDiff_Function_Spaces.pdf             (34 pages, 8.2 MB)
    ├── 2025_PyTorch_JAX_SciPy_Comparison.pdf        (14 pages, 1.1 MB)
    ├── 2025_Scientific_Intelligence_Survey.pdf      (68 pages, 2.1 MB)
    └── 2026_CMIP_Forge.pdf                          (28 pages, 10.0 MB)
```

---

## 🚀 Quick Start

### 1. Dataset is Already Included!

All 11 papers are included in this repository at `data/workshop_data/`. No download needed!

```bash
cd retrieval-playground/data/workshop_data
ls -lh *.pdf
# Shows all 11 papers (39.4 MB total, 392 pages)
```

### 2. Verify Dataset

```bash
# Check all PDFs
ls -lh workshop_data/*.pdf | wc -l
# Should show 11 files

# Review metadata
cat workshop_metadata.json | jq '.papers[] | {title, pages, size: .file_size}'
```

---

## 📋 Metadata Schema

Each paper in `workshop_metadata.json` has:

```json
{
  "title": "Full paper title",
  "filename": "YYYY_Short_Title.pdf",
  "pages": 84,
  "file_size": "3.5 MB",
  "topic": "AI | Climate Science | Data Science | ...",
  "year_published": YYYY,
  "link": "https://arxiv.org/abs/...",
  "proceedings": "arXiv"
}
```

---

## 📚 Citation Information

If you use this dataset in your research or workshop materials:

```bibtex
@dataset{scipy_workshop_dataset_2026,
  title = {SciPy Workshop Dataset: Recent Scientific Python Papers (2023-2026)},
  author = {Mahima Arora},
  year = {2026},
  publisher = {SciPy Conference},
  note = {Curated for RAG optimization workshop at SciPy 2026}
}
```

---

## 🎉 Happy RAG Building!

---

**Last Updated:** June 2026  
**Dataset Version:** 1.0  
**Maintained by:** retrieval-playground contributors