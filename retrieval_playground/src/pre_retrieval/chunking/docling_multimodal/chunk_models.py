"""Chunk Models for Multimodal Document Parsing."""

from pydantic import BaseModel, Field, model_validator, ConfigDict
from enum import Enum
from typing import Optional, List
from datetime import datetime
import pandas as pd


class ChunkType(Enum):
    """Enum for different types of content chunks."""
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"


class BaseChunk(BaseModel):
    """Base class for all chunk types."""
    chunk_id: str = Field(description="Unique identifier for the chunk")
    chunk_type: ChunkType = Field(description="Type of content chunk")
    content: str = Field(default="", description="Main content of the chunk (text, table description, or image description)")
    sequence_number: int = Field(default=0, description="Order of chunk in document (0-indexed)")
    source_document: Optional[str] = Field(default=None, description="Path or name of source document")
    source_page: Optional[int] = Field(default=None, description="Source page number (1-indexed)")
    parent_heading: Optional[str] = Field(default=None, description="Text of the current section heading")
    extraction_timestamp: Optional[datetime] = Field(default=None, description="When the chunk was extracted")

    class Config:
        use_enum_values = True


class TextChunk(BaseChunk):
    """Chunk containing text content."""
    chunk_type: ChunkType = Field(default=ChunkType.TEXT)
    word_count: Optional[int] = Field(default=None, description="Number of words in text")
    char_count: Optional[int] = Field(default=None, description="Number of characters in text")

    @model_validator(mode='after')
    def compute_metrics(self):
        """Auto-compute word and character counts if not provided."""
        if self.word_count is None and self.content:
            self.word_count = len(self.content.split())
        if self.char_count is None and self.content:
            self.char_count = len(self.content)
        return self


class TableChunk(BaseChunk):
    """Chunk containing table data as pandas DataFrame."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    chunk_type: ChunkType = Field(default=ChunkType.TABLE)
    dataframe: pd.DataFrame = Field(description="Table data as pandas DataFrame")
    columns: Optional[List[str]] = Field(default=None, description="List of column names")
    num_rows: Optional[int] = Field(default=None, description="Number of rows")
    num_cols: Optional[int] = Field(default=None, description="Number of columns")

    @model_validator(mode='after')
    def compute_dimensions(self):
        """Auto-compute table dimensions if not provided."""
        if self.dataframe is not None:
            if self.num_rows is None:
                self.num_rows = len(self.dataframe)
            if self.num_cols is None:
                self.num_cols = len(self.dataframe.columns)
        return self

    def get_columns(self) -> List[str]:
        """Get the column names from the dataframe."""
        if self.columns is not None:
            return self.columns
        return self.dataframe.columns.tolist()


class ImageChunk(BaseChunk):
    """Chunk containing image data."""
    chunk_type: ChunkType = Field(default=ChunkType.IMAGE)
    image_path: Optional[str] = Field(default=None, description="Path where extracted image is saved")
    image_format: Optional[str] = Field(default=None, description="Image format (png, jpg, gif, webp, etc.)")
    image_type: Optional[str] = Field(default="other", description="Type of image (photo, diagram, chart, etc.)")
    image_base64: Optional[str] = Field(default=None, description="Base64 encoded image data extracted from PDF")
