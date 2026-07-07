"""
Docling Multimodal Parser for extracting text, tables, and images from PDFs.

Adapted from: https://github.com/mahimaarora/multimodal-parser
"""

import os
import base64
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Union

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableStructureOptions,
    TableFormerMode,
    PictureDescriptionApiOptions,
)
from docling.datamodel.base_models import InputFormat
from docling.chunking import HybridChunker
from docling_core.types.doc import DocItemLabel, TextItem, TableItem, PictureItem

from .chunk_models import TextChunk, TableChunk, ImageChunk
from retrieval_playground.utils.model_manager import model_manager
from retrieval_playground.utils import config
from retrieval_playground.utils import config

logger = logging.getLogger(__name__)

Chunk = Union[TextChunk, TableChunk, ImageChunk]


class DoclingMultimodalParser:
    """PDF parser using Docling for text, table, and image extraction."""

    def __init__(
        self,
        images_scale: float = 2.0,
        table_mode: str = "accurate",
        do_cell_matching: bool = True,
        generate_descriptions: bool = True,
        images_output_dir: Optional[str] = None,
    ):
        """
        Initialize the multimodal parser.

        Args:
            images_scale: Scale factor for extracted images (default 2.0 = 144 DPI)
            table_mode: Table extraction mode - "accurate" or "fast"
            do_cell_matching: Map table structure to PDF cells
            generate_descriptions: Enable AI-generated descriptions for images/tables
            images_output_dir: Directory to save extracted images (defaults to data/images)
        """
        # Use config constant for images directory
        self.images_output_dir = Path(images_output_dir).resolve() if images_output_dir else config.DOCLING_IMAGES_DIR
        self.images_output_dir.mkdir(parents=True, exist_ok=True)

        # Get LLM from model manager for AI descriptions
        self.llm = model_manager.get_llm() if generate_descriptions else None
        self.generate_descriptions = generate_descriptions

        # Get API key from environment for Gemini descriptions
        self.api_key = os.getenv("GOOGLE_API_KEY")

        # Configure PDF pipeline
        pdf_pipeline_options = PdfPipelineOptions(
            generate_page_images=False,
            generate_picture_images=True,
            images_scale=images_scale,
            do_table_structure=True,
            table_structure_options=TableStructureOptions(
                do_cell_matching=do_cell_matching,
                mode=TableFormerMode.ACCURATE if table_mode == "accurate" else TableFormerMode.FAST,
            ),
        )

        # Enable image descriptions via Gemini API
        if self.api_key and generate_descriptions:
            pdf_pipeline_options.do_picture_description = True
            pdf_pipeline_options.enable_remote_services = True
            pdf_pipeline_options.picture_description_options = PictureDescriptionApiOptions(
                url="https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "x-goog-api-client": "docling-reader/1.0.0",
                },
                params={"model": config.MODEL_NAME, "max_tokens": 2048},
                prompt="""Describe this image for search and retrieval purposes. Write a clear, complete description that includes:

1. Type: State what kind of image this is (diagram, flowchart, architecture diagram, chart, graph, table, photo, screenshot, etc.)
2. Content: Describe the main elements, components, and relationships shown
3. Purpose: Explain what this image illustrates or demonstrates
4. Key details: Mention any labels, text, numbers, or important visual elements

Write in complete sentences. Make it detailed enough that someone could find this image by searching for its content.""",
                timeout=90.0,
                scale=1.0,
            )
        elif generate_descriptions and not self.api_key:
            logger.warning("GOOGLE_API_KEY not set. AI descriptions for images disabled.")

        self.converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline_options)}
        )

    def _convert_to_relative_path(self, absolute_path: Path) -> str:
        """Convert absolute path to relative path from project root."""
        abs_str = str(absolute_path.resolve())
        if "retrieval-playground" in abs_str:
            parts = abs_str.split("retrieval-playground/")
            return parts[-1] if len(parts) > 1 else abs_str
        return abs_str

    def parse(self, pdf_path: str) -> List[Chunk]:
        """
        Parse a PDF document and extract text, tables, and images.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of TextChunk, TableChunk, and ImageChunk objects
        """
        logger.info(f"    🔍 Parsing multimodal content from: {Path(pdf_path).name}")
        pdf_path = Path(pdf_path)

        result = self.converter.convert(str(pdf_path))
        document = result.document

        chunks: List[Chunk] = []
        chunk_idx = 0
        image_counter = 1  # Counter for deterministic image naming (starts at 1)

        # Step 1: Extract text chunks using HybridChunker
        chunker = HybridChunker(
            tokenizer=None,      # Uses built-in tokenizer
            max_tokens=512,      # 512 tokens (same as baseline for fair comparison)
            merge_peers=True,    # Merge sibling elements
            merge_threshold=0.5, # Merge if >=50% similar
        )

        for chunk in chunker.chunk(dl_doc=document):
            text = chunker.contextualize(chunk)
            if text and text.strip():
                source_page = chunk.meta.doc_items[0].prov[0].page_no if chunk.meta.doc_items and chunk.meta.doc_items[0].prov else None
                parent_heading = chunk.meta.headings[0] if chunk.meta.headings else None
                chunks.append(TextChunk(
                    chunk_id=f"text_{chunk_idx}",
                    sequence_number=chunk_idx,
                    source_document=pdf_path.name,
                    source_page=source_page,
                    parent_heading=parent_heading,
                    content=text,
                    extraction_timestamp=datetime.now()
                ))
                chunk_idx += 1

        # Step 2: Extract tables and images from document items
        current_heading: Optional[str] = None
        for item, _level in document.iterate_items():
            if self._is_heading(item):
                if isinstance(item, TextItem) and item.text:
                    current_heading = item.text
                continue

            if hasattr(item, 'label') and item.label == DocItemLabel.CAPTION:
                continue

            source_page = item.prov[0].page_no if item.prov else None
            chunk = None

            if isinstance(item, TableItem):
                chunk = self._create_table_chunk(item, document, chunk_idx, pdf_path.name, source_page, current_heading)
            elif isinstance(item, PictureItem):
                chunk = self._create_image_chunk(item, document, chunk_idx, pdf_path.name, source_page, current_heading, image_counter)
                if chunk:
                    image_counter += 1  # Increment only for successfully created images

            if chunk:
                chunks.append(chunk)
                chunk_idx += 1

        text_count = sum(1 for c in chunks if isinstance(c, TextChunk))
        table_count = sum(1 for c in chunks if isinstance(c, TableChunk))
        image_count = sum(1 for c in chunks if isinstance(c, ImageChunk))

        logger.info(f"    ✓ Extracted {len(chunks)} chunks ({text_count} text, {table_count} tables, {image_count} images)")
        return chunks

    def _is_heading(self, item) -> bool:
        """Check if item is a section heading."""
        return (
            hasattr(item, 'label') and
            item.label in [DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE] and
            isinstance(item, TextItem)
        )

    def _create_table_chunk(self, item: TableItem, document, idx: int, source_doc: str,
                            source_page: Optional[int], parent_heading: Optional[str]) -> TableChunk:
        """Create a TableChunk from a TableItem."""
        df = item.export_to_dataframe(doc=document)
        df = df.astype(str)
        df.columns = [str(col) for col in df.columns]

        caption = self._get_caption(item, document)
        content = self._describe_table(df, caption)

        return TableChunk(
            chunk_id=f"table_{idx}",
            sequence_number=idx,
            source_document=source_doc,
            source_page=source_page,
            parent_heading=parent_heading,
            dataframe=df,
            content=content,
            extraction_timestamp=datetime.now()
        )

    def _create_image_chunk(self, item: PictureItem, document, idx: int, source_doc: str,
                            source_page: Optional[int], parent_heading: Optional[str],
                            image_num: int = 0) -> ImageChunk:
        """Create an ImageChunk from a PictureItem."""
        image_base64 = None
        image_format = None
        image_path = None

        # Extract image data from document.pictures using self_ref index
        item_ref = getattr(item, 'self_ref', None)
        if item_ref and item_ref.startswith("#/pictures/") and hasattr(document, 'pictures'):
            try:
                pic_idx = int(item_ref.split("/")[-1])
                if pic_idx < len(document.pictures):
                    pic = document.pictures[pic_idx]
                    image_data = self._extract_image_data(pic)
                    if image_data:
                        image_base64 = image_data["base64"]
                        image_format = image_data["format"]
            except (ValueError, IndexError):
                pass

        if image_base64:
            doc_name = Path(source_doc).stem
            image_filename = f"{doc_name}_image_{image_num}.{image_format}"
            image_path = self.images_output_dir / image_filename
            try:
                with open(image_path, "wb") as f:
                    f.write(base64.b64decode(image_base64))
                image_path = self._convert_to_relative_path(image_path)
            except Exception as e:
                logger.warning(f"    ⚠️ Failed to save image: {e}")
                image_path = None

        # Get description from annotations and caption
        caption = self._get_caption(item, document)
        annotation = self._get_annotation(item)

        # Infer image type
        image_type = self._infer_image_type(annotation, caption) if (annotation or caption) else "other"

        # Build content
        if annotation and caption:
            content = f"{annotation}\n\nCaption: {caption}"
        elif annotation:
            content = annotation
        elif caption:
            content = caption
        else:
            content = f"Image ({image_format})" if image_format else "Image"

        return ImageChunk(
            chunk_id=f"image_{idx}",
            sequence_number=idx,
            source_document=source_doc,
            source_page=source_page,
            parent_heading=parent_heading,
            content=content,
            image_path=image_path,
            image_base64=image_base64,
            image_format=image_format,
            image_type=image_type,
            extraction_timestamp=datetime.now()
        )

    def _extract_image_data(self, item) -> Optional[dict]:
        """Extract base64 image data from a PictureItem."""
        if not hasattr(item, 'image') or item.image is None:
            return None

        img = item.image
        if hasattr(img, 'uri') and img.uri:
            uri = str(img.uri)
            if uri.startswith("data:"):
                header, base64_data = uri.split(",", 1)
                mime = header.split(";")[0].replace("data:", "")
                return {"base64": base64_data, "format": mime.split("/")[-1]}

        return None

    def _get_caption(self, item, document) -> str:
        """Extract caption from item by resolving references."""
        if not hasattr(item, 'captions') or not item.captions:
            return ""

        texts = []
        for ref in item.captions:
            try:
                if hasattr(ref, 'cref') and ref.cref.startswith("#/texts/"):
                    text_idx = int(ref.cref.split("/")[-1])
                    if text_idx < len(document.texts):
                        text = getattr(document.texts[text_idx], 'text', None)
                        if text:
                            texts.append(text)
                elif hasattr(ref, 'text') and ref.text:
                    texts.append(ref.text)
            except Exception:
                pass

        return " ".join(texts)

    def _get_annotation(self, item) -> str:
        """Extract AI-generated description from annotations."""
        if not hasattr(item, 'annotations') or not item.annotations:
            return ""

        for ann in item.annotations:
            kind = getattr(ann, 'kind', None) or (ann.get('kind') if isinstance(ann, dict) else None)
            text = getattr(ann, 'text', None) or (ann.get('text') if isinstance(ann, dict) else None)
            if kind == "description" and text:
                return text

        return ""

    def _describe_table(self, df, caption: str = "") -> str:
        """Generate AI description for a table."""
        if df.empty:
            return caption or "Empty table"

        headers = df.columns.tolist()
        fallback = f"Table with {len(df)} rows and {len(headers)} columns. Headers: {', '.join(headers)}"
        if caption:
            fallback = f"{caption} | {fallback}"

        if not self.llm:
            return fallback

        prompt = f"""Describe this table for search and retrieval purposes. Write a clear, complete description that includes:

1. Content: What type of data or information does this table contain?
2. Structure: What are the main columns and what do they represent?
3. Purpose: What comparisons, metrics, or insights does this table show?
4. Use cases: What specific questions can this table help answer?

TABLE INFORMATION
- Columns: {', '.join(headers)}
- Row count: {len(df)}
{f"- Caption: {caption}" if caption else ""}

SAMPLE DATA (first {min(5, len(df))} rows):
{df.head(5).to_markdown()}

Write in complete sentences. Make it detailed enough that someone could find this table by searching for its content or purpose."""

        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            logger.warning(f"    ⚠️ Table description failed: {e}")
            return fallback

    def _infer_image_type(self, description: str, caption: str = "") -> str:
        """Infer image type from description and caption using LLM."""
        if not description and not caption:
            return "other"

        if not self.llm:
            return "other"

        valid_types = ["photo", "diagram", "chart", "logo", "screenshot", "other"]

        context = ""
        if description:
            context += f"DESCRIPTION: {description}\n"
        if caption:
            context += f"CAPTION: {caption}\n"

        prompt = f"""Classify this image into ONE type:
- photo: real objects, people, places, scenes
- diagram: flowcharts, architecture, process flows, schematics
- chart: bar charts, pie charts, graphs, data visualizations
- logo: company logos, brand marks, icons, symbols
- screenshot: UI screenshots, application windows, web pages
- other: anything else

{context}
Respond with ONLY the type (one word, lowercase): photo, diagram, chart, logo, screenshot, or other"""

        try:
            result = self.llm.invoke(prompt).content.strip().lower()
            if result in valid_types:
                return result
            for t in valid_types:
                if t in result:
                    return t
            return "other"
        except Exception:
            return "other"
