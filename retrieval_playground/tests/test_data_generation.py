#!/usr/bin/env python3
"""
Synthetic test data generation for RAG evaluation.
Reads PDF files from data/sample_research_papers/ and generates test samples using ModelManager.
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from retrieval_playground.utils import config
from pydantic import BaseModel, Field, ValidationError
from docling.document_converter import DocumentConverter
from retrieval_playground.utils.model_manager import model_manager
from retrieval_playground.utils.pylogger import get_python_logger

# Initialize logger
logger = get_python_logger(log_level="info")


class TestSample(BaseModel):
    """Pydantic model for test sample validation."""
    user_input: str = Field(..., min_length=10, description="A natural question about the content")
    reference_context: str = Field(..., min_length=20, description="The passage from the text containing the answer")
    reference: str = Field(..., min_length=5, description="The ground truth answer")
    source_file: str = Field(None, description="Source PDF filename")


def get_model():
    """Get the LLM model from ModelManager."""
    return model_manager.get_llm()

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF using docling."""
    try:
        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        return result.document.export_to_markdown()
    except Exception as e:
        logger.error(f"❌ Failed to extract text from {Path(pdf_path).name}: {e}")
        raise

def generate_test_samples(text_content: str, model, filename: str) -> List[TestSample]:
    """Generate test samples from text content using LLM."""
    prompt = f"""
    Based on the following research paper content, generate 1 test sample for RAG evaluation.
    
    For each sample, create:
    1. user_input: A natural question someone might ask about this content
    2. reference_context: The passage from the text that contains the answer, along with broader surrounding context (quote directly from the paper)
    3. reference: A clear, concise answer based on the reference_context
    
    Return the response as a JSON array with this exact structure (single sample):
    [
        {{
            "user_input": "question here",
            "reference_context": "exact passage from text",
            "reference": "ground truth answer"
        }}
    ]
    
    Paper content:
    {text_content}  
    
    Make sure the question is meaningful and represents an important aspect of the paper.
    """
    
    try:
        response = model.invoke(prompt)
        # Extract JSON from response
        response_text = response.content.strip()
        
        # Find JSON array in response
        start_idx = response_text.find('[')
        end_idx = response_text.rfind(']') + 1
        
        if start_idx != -1 and end_idx != 0:
            json_str = response_text[start_idx:end_idx]
            raw_samples = json.loads(json_str)
            
            # Validate and create TestSample objects
            validated_samples = []
            for sample_data in raw_samples:
                try:
                    # Add source filename
                    sample_data['source_file'] = filename
                    # Validate with Pydantic
                    validated_sample = TestSample(**sample_data)
                    validated_samples.append(validated_sample)
                except ValidationError as ve:
                    logger.warning(f"⚠️ Validation error in {filename}: {ve}")
                    continue
            
            return validated_samples
        else:
            logger.warning(f"⚠️ Could not extract JSON from LLM response for {filename}")
            return []
            
    except json.JSONDecodeError as je:
        logger.error(f"❌ JSON decode error for {filename}: {je}")
        return []
    except Exception as e:
        logger.error(f"❌ Error generating samples for {filename}: {e}")
        return []

def main():
    """Main function to process all PDFs and generate test data."""
    logger.info("🚀 Starting test data generation...")
    
    # Setup
    model = get_model()
    pdf_dir = config.SAMPLE_PAPERS_DIR
    all_samples = []
    
    # Check if directory exists
    if not pdf_dir.exists():
        logger.error(f"❌ Directory {pdf_dir} not found")
        raise FileNotFoundError(f"Directory {pdf_dir} not found")
    
    # Process each PDF file
    pdf_files = list(pdf_dir.glob("*.pdf"))
    logger.info(f"📁 Processing {len(pdf_files)} PDF files...")
    
    if not pdf_files:
        logger.warning("⚠️ No PDF files found")
        return
    
    for pdf_path in pdf_files:
        try:
            # Extract text from PDF
            text_content = extract_text_from_pdf(str(pdf_path))
            
            # Generate test samples
            samples = generate_test_samples(text_content, model, pdf_path.name)
            all_samples.extend(samples)
            
        except Exception as e:
            logger.error(f"❌ Error processing {pdf_path.name}: {e}")
            continue
    
    # Convert Pydantic models to dictionaries and save to JSON file
    output_file = "retrieval_playground/tests/test_queries.json"
    
    try:
        samples_dict = [sample.model_dump() for sample in all_samples]
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(samples_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Generated {len(all_samples)} test samples saved to {output_file}")
        
    except Exception as e:
        logger.error(f"❌ Failed to save test samples: {e}")
        raise

def cli_main():
    """Entry point for console script."""
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Operation cancelled by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        exit(1)

if __name__ == "__main__":
    cli_main()
