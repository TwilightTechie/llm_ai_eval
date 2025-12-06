"""
OpenAI Vision-based OCR provider.
Uses GPT-4 Vision to extract text and fields from documents.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

from openai import OpenAI

from .base import OCRProvider, OCRResult
from .registry import ProviderRegistry
from ..utils.logging_config import get_logger, PerformanceLogger, APICallLogger
from ..utils.pdf import PDFConverter
from ..utils.rate_limiter import create_rate_limiter, CompositeRateLimiter

logger = get_logger(__name__)
api_logger = APICallLogger(logger)


@ProviderRegistry.register("openai")
class OpenAIVisionProvider(OCRProvider):
    """
    OCR provider using OpenAI's GPT-4 Vision model.
    Converts PDFs to images and uses vision capabilities for extraction.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize OpenAI Vision provider.
        
        Args:
            config: Provider configuration including:
                - model: Model to use (default: gpt-4o)
                - api_key_env: Environment variable for API key
                - max_tokens: Maximum tokens in response
                - temperature: Model temperature
                - rate_limit: Rate limiting configuration
        """
        super().__init__(config)
        self._name = "openai"
        
        # Get API key
        api_key_env = config.get('api_key_env', 'OPENAI_API_KEY')
        api_key = os.environ.get(api_key_env)
        
        if not api_key:
            raise ValueError(
                f"OpenAI API key not found. Set {api_key_env} environment variable."
            )
        
        # Initialize client
        self.client = OpenAI(api_key=api_key)
        
        # Model settings
        self.model = config.get('model', 'gpt-4o')
        self.max_tokens = config.get('max_tokens', 4096)
        self.temperature = config.get('temperature', 0.0)
        
        # PDF converter
        self.pdf_converter = PDFConverter(dpi=200)
        
        # Rate limiter
        rate_config = config.get('rate_limit', {})
        rate_config['enabled'] = rate_config.get('enabled', True)
        rate_config['requests_per_minute'] = rate_config.get('requests_per_minute', 60)
        rate_config['tokens_per_minute'] = rate_config.get('tokens_per_minute', 150000)
        self.rate_limiter = create_rate_limiter(rate_config)
        
        logger.info(f"OpenAI Vision provider initialized with model: {self.model}")
    
    def _build_extraction_prompt(self, schema: Dict[str, Any]) -> str:
        """Build the extraction prompt from schema."""
        schema_name = schema.get('schema_name', 'document')
        fields = schema.get('fields', [])
        
        # Build field descriptions
        field_descriptions = []
        for field in fields:
            name = field['name']
            field_type = field.get('type', 'string')
            required = field.get('required', False)
            description = field.get('description', '')
            
            req_str = "(required)" if required else "(optional)"
            field_descriptions.append(f"  - {name} ({field_type}) {req_str}: {description}")
        
        fields_text = "\n".join(field_descriptions)
        
        # Check for custom prompt template
        if 'extraction_prompt' in schema:
            prompt = schema['extraction_prompt'].replace('{field_list}', fields_text)
        else:
            prompt = f"""Extract information from this {schema_name} document.

Return a JSON object with the following fields:
{fields_text}

Rules:
1. Extract values exactly as they appear in the document
2. For fields not found or unclear, use null
3. For list fields, return an array of items
4. For date fields, use YYYY-MM-DD format if possible
5. Return ONLY valid JSON, no additional text

JSON Output:"""
        
        return prompt
    
    def extract(
        self,
        document_path: Path,
        schema: Dict[str, Any],
        document_id: Optional[str] = None
    ) -> OCRResult:
        """
        Extract fields from a document using GPT-4 Vision.
        
        Args:
            document_path: Path to PDF document
            schema: Field extraction schema
            document_id: Optional document identifier
        
        Returns:
            OCRResult with extracted fields
        """
        document_path = Path(document_path)
        doc_id = document_id or document_path.stem
        
        logger.info(f"Extracting fields from: {document_path.name}")
        
        start_time = time.perf_counter()
        
        try:
            # Convert PDF to base64 images
            with PerformanceLogger(logger, f"PDF conversion: {document_path.name}"):
                base64_images = self.pdf_converter.convert_to_base64(document_path)
            
            if not base64_images:
                return OCRResult.from_error(
                    doc_id, self.name, "Failed to convert PDF to images"
                )
            
            # Build prompt
            prompt = self._build_extraction_prompt(schema)
            
            # Build messages with images
            content = [{"type": "text", "text": prompt}]
            
            for i, b64_img in enumerate(base64_images):
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{b64_img}",
                        "detail": "high"
                    }
                })
                logger.debug(f"Added page {i + 1} to request")
            
            messages = [{"role": "user", "content": content}]
            
            # Estimate tokens and acquire rate limit
            estimated_tokens = len(prompt) // 4 + len(base64_images) * 1000
            self.rate_limiter.acquire(estimated_tokens)
            
            # Call OpenAI API
            with api_logger.log_call(self.name, "chat.completions", 
                                     extra={"document_id": doc_id}):
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
            
            # Parse response
            response_text = response.choices[0].message.content
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            
            api_logger.log_tokens(input_tokens, output_tokens)
            
            # Extract JSON from response
            fields = self._parse_json_response(response_text)
            
            processing_time = (time.perf_counter() - start_time) * 1000
            
            logger.info(
                f"Extraction complete for {doc_id}: "
                f"{len(fields)} fields, {processing_time:.0f}ms"
            )
            
            return OCRResult(
                document_id=doc_id,
                provider_name=self.name,
                fields=fields,
                raw_text=response_text,
                success=True,
                processing_time_ms=processing_time,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                metadata={
                    "model": self.model,
                    "pages": len(base64_images),
                }
            )
            
        except Exception as e:
            logger.error(f"Extraction failed for {doc_id}: {e}", exc_info=True)
            processing_time = (time.perf_counter() - start_time) * 1000
            
            return OCRResult(
                document_id=doc_id,
                provider_name=self.name,
                fields={},
                success=False,
                error_message=str(e),
                processing_time_ms=processing_time,
            )
    
    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """Parse JSON from model response."""
        # Try direct JSON parse
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from markdown code block
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            if end > start:
                try:
                    return json.loads(response_text[start:end].strip())
                except json.JSONDecodeError:
                    pass
        
        # Try to find JSON object in text
        if "{" in response_text and "}" in response_text:
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            try:
                return json.loads(response_text[start:end])
            except json.JSONDecodeError:
                pass
        
        logger.warning(f"Could not parse JSON from response: {response_text[:200]}...")
        return {}
    
    def extract_text(self, document_path: Path) -> str:
        """
        Extract raw text from a document.
        
        Args:
            document_path: Path to PDF document
        
        Returns:
            Extracted text content
        """
        document_path = Path(document_path)
        
        logger.info(f"Extracting text from: {document_path.name}")
        
        try:
            # Convert PDF to images
            base64_images = self.pdf_converter.convert_to_base64(document_path)
            
            if not base64_images:
                raise ValueError("Failed to convert PDF to images")
            
            prompt = """Extract all text from this document.
Return the complete text content, preserving the structure and layout as much as possible.
Include all visible text, numbers, dates, and other content."""
            
            content = [{"type": "text", "text": prompt}]
            
            for b64_img in base64_images:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{b64_img}",
                        "detail": "high"
                    }
                })
            
            messages = [{"role": "user", "content": content}]
            
            # Acquire rate limit
            self.rate_limiter.acquire(len(base64_images) * 1000)
            
            with api_logger.log_call(self.name, "chat.completions.text"):
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Text extraction failed: {e}", exc_info=True)
            raise
    
    def validate_config(self) -> bool:
        """Validate provider configuration."""
        api_key_env = self.config.get('api_key_env', 'OPENAI_API_KEY')
        if not os.environ.get(api_key_env):
            raise ValueError(f"API key not found in {api_key_env}")
        return True
    
    def health_check(self) -> bool:
        """Check if OpenAI API is accessible."""
        try:
            # Simple API call to verify connectivity
            self.client.models.list()
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def get_capabilities(self) -> Dict[str, bool]:
        """Return provider capabilities."""
        return {
            "supports_pdf": True,
            "supports_images": True,
            "supports_handwriting": True,
            "supports_tables": True,
            "supports_forms": True,
        }


