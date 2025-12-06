"""
Base classes for OCR providers.
All OCR providers must implement the OCRProvider abstract base class.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime


@dataclass
class FieldExtraction:
    """Represents a single extracted field from a document."""
    field_name: str
    value: Any
    confidence: Optional[float] = None
    raw_text: Optional[str] = None  # Original text before normalization
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "field_name": self.field_name,
            "value": self.value,
            "confidence": self.confidence,
            "raw_text": self.raw_text,
        }


@dataclass
class OCRResult:
    """
    Result from an OCR extraction operation.
    Contains extracted fields, raw text, and metadata.
    """
    document_id: str
    provider_name: str
    
    # Extracted data
    fields: Dict[str, Any]  # Field name -> extracted value
    raw_text: Optional[str] = None  # Full extracted text
    
    # Metadata
    success: bool = True
    error_message: Optional[str] = None
    
    # Timing
    processing_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Token usage (for LLM-based providers)
    input_tokens: int = 0
    output_tokens: int = 0
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "document_id": self.document_id,
            "provider_name": self.provider_name,
            "fields": self.fields,
            "raw_text": self.raw_text,
            "success": self.success,
            "error_message": self.error_message,
            "processing_time_ms": self.processing_time_ms,
            "timestamp": self.timestamp.isoformat(),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_error(cls, document_id: str, provider_name: str, 
                   error: str) -> "OCRResult":
        """Create a result representing an error."""
        return cls(
            document_id=document_id,
            provider_name=provider_name,
            fields={},
            success=False,
            error_message=error,
        )


class OCRProvider(ABC):
    """
    Abstract base class for OCR providers.
    
    All OCR providers (OpenAI, Textract, Google Doc AI, etc.) must
    implement this interface to be used with the evaluation framework.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the provider with configuration.
        
        Args:
            config: Provider-specific configuration dictionary
        """
        self.config = config
        self._name = self.__class__.__name__
    
    @property
    def name(self) -> str:
        """Return the provider name."""
        return self._name
    
    @abstractmethod
    def extract(
        self, 
        document_path: Path,
        schema: Dict[str, Any],
        document_id: Optional[str] = None
    ) -> OCRResult:
        """
        Extract fields from a document according to the schema.
        
        Args:
            document_path: Path to the document file (PDF)
            schema: Field extraction schema defining what to extract
            document_id: Optional document identifier
        
        Returns:
            OCRResult containing extracted fields and metadata
        """
        pass
    
    @abstractmethod
    def extract_text(self, document_path: Path) -> str:
        """
        Extract raw text from a document without field parsing.
        
        Args:
            document_path: Path to the document file
        
        Returns:
            Extracted text content
        """
        pass
    
    def validate_config(self) -> bool:
        """
        Validate provider configuration.
        
        Returns:
            True if configuration is valid
        
        Raises:
            ValueError if configuration is invalid
        """
        return True
    
    def health_check(self) -> bool:
        """
        Check if the provider is operational.
        
        Returns:
            True if provider is healthy and ready
        """
        return True
    
    def get_capabilities(self) -> Dict[str, bool]:
        """
        Return provider capabilities.
        
        Returns:
            Dictionary of capability flags
        """
        return {
            "supports_pdf": True,
            "supports_images": False,
            "supports_handwriting": False,
            "supports_tables": False,
            "supports_forms": False,
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"


