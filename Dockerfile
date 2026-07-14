# Retrieval Playground - Workshop Docker Image
# Python 3.12 base image
FROM python:3.12-slim

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    libgl1-mesa-glx \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire project first
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[notebook]" && \
    pip install --no-cache-dir "langchain-community>=0.3,<0.4.2"

# Pre-download Docling models from HuggingFace (avoids download during workshop)
RUN python -c "\
from docling.document_converter import DocumentConverter; \
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableStructureOptions, TableFormerMode; \
from docling.datamodel.base_models import InputFormat; \
from docling.document_converter import PdfFormatOption; \
print('Pre-downloading Docling models...'); \
pipeline_opts = PdfPipelineOptions( \
    do_table_structure=True, \
    table_structure_options=TableStructureOptions(mode=TableFormerMode.ACCURATE) \
); \
converter = DocumentConverter( \
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts)} \
); \
print('✅ Docling models cached successfully'); \
"

# Create necessary directories
RUN mkdir -p /workspace/retrieval_playground/data/sample_research_papers

# Expose Jupyter port
EXPOSE 8888

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV JUPYTER_ENABLE_LAB=yes

# Create startup script
RUN echo '#!/bin/bash\n\
echo ""\n\
echo "🧩 Retrieval Playground - Workshop Environment"\n\
echo "============================================"\n\
echo ""\n\
if [ ! -f .env ]; then\n\
    echo "⚠️  WARNING: .env file not found!"\n\
    echo "Please create a .env file with your API keys."\n\
    echo "See .env.example for the required format."\n\
    echo ""\n\
fi\n\
echo "Starting Jupyter Notebook..."\n\
echo ""\n\
echo "📝 Once started, you will see a URL like:"\n\
echo "   http://127.0.0.1:8888/tree?token=..."\n\
echo ""\n\
echo "Copy and paste that URL into your browser."\n\
echo ""\n\
echo "============================================"\n\
echo ""\n\
python -m jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token="" --NotebookApp.password=""\n\
' > /workspace/start.sh && chmod +x /workspace/start.sh

# Default command
CMD ["/workspace/start.sh"]
