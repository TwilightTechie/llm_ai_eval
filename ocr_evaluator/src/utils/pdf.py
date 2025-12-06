"""
PDF to Image conversion utilities for OCR processing.
"""

import base64
import io
from pathlib import Path
from typing import List, Optional, Union
from PIL import Image

from .logging_config import get_logger, PerformanceLogger

logger = get_logger(__name__)


class PDFConverter:
    """
    Converts PDF documents to images for OCR processing.
    Supports both pdf2image (poppler) and PyMuPDF backends.
    """
    
    def __init__(self, dpi: int = 200, image_format: str = "PNG"):
        """
        Initialize PDF converter.
        
        Args:
            dpi: Resolution for PDF rendering (default: 200)
            image_format: Output image format (PNG, JPEG)
        """
        self.dpi = dpi
        self.image_format = image_format
        self._backend = None
        self._detect_backend()
    
    def _detect_backend(self):
        """Detect available PDF processing backend."""
        # Try PyMuPDF first (faster, no external dependencies)
        try:
            import fitz  # PyMuPDF
            self._backend = "pymupdf"
            logger.info("Using PyMuPDF backend for PDF conversion")
            return
        except ImportError:
            pass
        
        # Fall back to pdf2image (requires poppler)
        try:
            from pdf2image import convert_from_path
            self._backend = "pdf2image"
            logger.info("Using pdf2image backend for PDF conversion")
            return
        except ImportError:
            pass
        
        logger.warning(
            "No PDF backend available. Install PyMuPDF (pip install pymupdf) "
            "or pdf2image with poppler."
        )
        self._backend = None
    
    @PerformanceLogger(get_logger(__name__), "PDF to Images Conversion")
    def convert_to_images(
        self, 
        pdf_path: Union[str, Path],
        max_pages: Optional[int] = None
    ) -> List[Image.Image]:
        """
        Convert PDF to list of PIL Images.
        
        Args:
            pdf_path: Path to PDF file
            max_pages: Maximum number of pages to convert (None = all)
        
        Returns:
            List of PIL Image objects, one per page
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if self._backend == "pymupdf":
            return self._convert_pymupdf(pdf_path, max_pages)
        elif self._backend == "pdf2image":
            return self._convert_pdf2image(pdf_path, max_pages)
        else:
            raise RuntimeError("No PDF backend available")
    
    def _convert_pymupdf(
        self, 
        pdf_path: Path, 
        max_pages: Optional[int]
    ) -> List[Image.Image]:
        """Convert using PyMuPDF."""
        import fitz
        
        images = []
        doc = fitz.open(pdf_path)
        
        try:
            num_pages = min(len(doc), max_pages) if max_pages else len(doc)
            logger.debug(f"Converting {num_pages} pages from {pdf_path.name}")
            
            for page_num in range(num_pages):
                page = doc[page_num]
                
                # Calculate zoom factor based on DPI
                zoom = self.dpi / 72  # PDF default is 72 DPI
                matrix = fitz.Matrix(zoom, zoom)
                
                # Render page to pixmap
                pixmap = page.get_pixmap(matrix=matrix)
                
                # Convert to PIL Image
                img_data = pixmap.tobytes("png")
                image = Image.open(io.BytesIO(img_data))
                images.append(image)
                
                logger.debug(f"Converted page {page_num + 1}/{num_pages}")
        finally:
            doc.close()
        
        return images
    
    def _convert_pdf2image(
        self, 
        pdf_path: Path, 
        max_pages: Optional[int]
    ) -> List[Image.Image]:
        """Convert using pdf2image."""
        from pdf2image import convert_from_path
        
        kwargs = {"dpi": self.dpi, "fmt": self.image_format.lower()}
        
        if max_pages:
            kwargs["last_page"] = max_pages
        
        logger.debug(f"Converting PDF with pdf2image: {pdf_path.name}")
        images = convert_from_path(pdf_path, **kwargs)
        
        return images
    
    def convert_to_base64(
        self, 
        pdf_path: Union[str, Path],
        max_pages: Optional[int] = None
    ) -> List[str]:
        """
        Convert PDF pages to base64-encoded images.
        Useful for API calls that require base64 input.
        
        Args:
            pdf_path: Path to PDF file
            max_pages: Maximum pages to convert
        
        Returns:
            List of base64-encoded image strings
        """
        images = self.convert_to_images(pdf_path, max_pages)
        base64_images = []
        
        for i, image in enumerate(images):
            buffer = io.BytesIO()
            image.save(buffer, format=self.image_format)
            buffer.seek(0)
            
            b64_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
            base64_images.append(b64_string)
            
            logger.debug(f"Encoded page {i + 1} to base64 ({len(b64_string)} chars)")
        
        return base64_images
    
    def get_page_count(self, pdf_path: Union[str, Path]) -> int:
        """Get the number of pages in a PDF."""
        pdf_path = Path(pdf_path)
        
        if self._backend == "pymupdf":
            import fitz
            doc = fitz.open(pdf_path)
            count = len(doc)
            doc.close()
            return count
        elif self._backend == "pdf2image":
            from pdf2image import pdfinfo_from_path
            info = pdfinfo_from_path(pdf_path)
            return info.get("Pages", 0)
        else:
            raise RuntimeError("No PDF backend available")


