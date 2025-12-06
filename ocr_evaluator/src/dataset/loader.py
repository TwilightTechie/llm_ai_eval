"""
Dataset loader for OCR evaluation.
Handles loading documents, ground truth, and metadata.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Iterator
from dataclasses import dataclass, field
from datetime import datetime

from ..utils.logging_config import get_logger, PerformanceLogger

logger = get_logger(__name__)


@dataclass
class GroundTruth:
    """Ground truth data for a document."""
    document_id: str
    schema_name: str
    fields: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GroundTruth":
        """Create GroundTruth from dictionary."""
        return cls(
            document_id=data['document_id'],
            schema_name=data.get('schema', 'unknown'),
            fields=data.get('fields', {}),
            metadata=data.get('metadata', {}),
        )
    
    @classmethod
    def from_json_file(cls, path: Path) -> "GroundTruth":
        """Load ground truth from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'document_id': self.document_id,
            'schema': self.schema_name,
            'fields': self.fields,
            'metadata': self.metadata,
        }
    
    def get_field(self, name: str) -> Any:
        """Get a field value."""
        return self.fields.get(name)


@dataclass
class Document:
    """Represents a document in the evaluation dataset."""
    document_id: str
    file_path: Path
    ground_truth: Optional[GroundTruth] = None
    schema_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def filename(self) -> str:
        """Get the document filename."""
        return self.file_path.name
    
    @property
    def exists(self) -> bool:
        """Check if document file exists."""
        return self.file_path.exists()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'document_id': self.document_id,
            'file_path': str(self.file_path),
            'schema_name': self.schema_name,
            'has_ground_truth': self.ground_truth is not None,
            'metadata': self.metadata,
        }


@dataclass
class Dataset:
    """Collection of documents for evaluation."""
    name: str
    documents: List[Document]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def __len__(self) -> int:
        return len(self.documents)
    
    def __iter__(self) -> Iterator[Document]:
        return iter(self.documents)
    
    def __getitem__(self, index: int) -> Document:
        return self.documents[index]
    
    def filter_by_schema(self, schema_name: str) -> "Dataset":
        """Filter documents by schema name."""
        filtered = [d for d in self.documents if d.schema_name == schema_name]
        return Dataset(
            name=f"{self.name}_{schema_name}",
            documents=filtered,
            metadata={**self.metadata, 'filtered_by': schema_name},
        )
    
    def filter_by_metadata(self, key: str, value: Any) -> "Dataset":
        """Filter documents by metadata field."""
        filtered = [
            d for d in self.documents 
            if d.metadata.get(key) == value or 
               (d.ground_truth and d.ground_truth.metadata.get(key) == value)
        ]
        return Dataset(
            name=f"{self.name}_{key}_{value}",
            documents=filtered,
            metadata={**self.metadata, f'filtered_{key}': value},
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        schemas = {}
        with_ground_truth = 0
        
        for doc in self.documents:
            schema = doc.schema_name or 'unknown'
            schemas[schema] = schemas.get(schema, 0) + 1
            if doc.ground_truth:
                with_ground_truth += 1
        
        return {
            'total_documents': len(self.documents),
            'with_ground_truth': with_ground_truth,
            'schemas': schemas,
            'metadata': self.metadata,
        }


class DatasetLoader:
    """
    Loads evaluation datasets from filesystem.
    
    Expected directory structure:
    dataset_dir/
    ├── documents/          # PDF files
    │   ├── doc_001.pdf
    │   └── doc_002.pdf
    ├── ground_truth/       # JSON ground truth files
    │   ├── doc_001.json
    │   └── doc_002.json
    └── metadata.json       # Optional dataset metadata
    """
    
    def __init__(self, dataset_dir: Path):
        """
        Initialize dataset loader.
        
        Args:
            dataset_dir: Root directory of the dataset
        """
        self.dataset_dir = Path(dataset_dir)
        self.documents_dir = self.dataset_dir / "documents"
        self.ground_truth_dir = self.dataset_dir / "ground_truth"
        self.metadata_file = self.dataset_dir / "metadata.json"
        
        self._validate_structure()
    
    def _validate_structure(self):
        """Validate dataset directory structure."""
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")
        
        if not self.documents_dir.exists():
            logger.warning(f"Documents directory not found: {self.documents_dir}")
        
        if not self.ground_truth_dir.exists():
            logger.warning(f"Ground truth directory not found: {self.ground_truth_dir}")
    
    @PerformanceLogger(get_logger(__name__), "Loading dataset")
    def load(self, schema_filter: Optional[str] = None) -> Dataset:
        """
        Load the complete dataset.
        
        Args:
            schema_filter: Optional schema name to filter by
        
        Returns:
            Loaded Dataset
        """
        # Load metadata
        metadata = {}
        if self.metadata_file.exists():
            with open(self.metadata_file) as f:
                metadata = json.load(f)
        
        # Find all documents
        documents = []
        
        if self.documents_dir.exists():
            for doc_path in sorted(self.documents_dir.glob("*.pdf")):
                doc_id = doc_path.stem
                
                # Load ground truth if available
                ground_truth = self._load_ground_truth(doc_id)
                
                # Determine schema
                schema_name = None
                if ground_truth:
                    schema_name = ground_truth.schema_name
                
                # Apply schema filter
                if schema_filter and schema_name != schema_filter:
                    continue
                
                # Get document metadata
                doc_metadata = metadata.get('documents', {}).get(doc_id, {})
                
                document = Document(
                    document_id=doc_id,
                    file_path=doc_path,
                    ground_truth=ground_truth,
                    schema_name=schema_name,
                    metadata=doc_metadata,
                )
                documents.append(document)
        
        dataset = Dataset(
            name=self.dataset_dir.name,
            documents=documents,
            metadata=metadata,
        )
        
        stats = dataset.get_statistics()
        logger.info(
            f"Loaded dataset '{dataset.name}': {stats['total_documents']} documents, "
            f"{stats['with_ground_truth']} with ground truth"
        )
        
        return dataset
    
    def _load_ground_truth(self, document_id: str) -> Optional[GroundTruth]:
        """Load ground truth for a document."""
        gt_path = self.ground_truth_dir / f"{document_id}.json"
        
        if not gt_path.exists():
            return None
        
        try:
            return GroundTruth.from_json_file(gt_path)
        except Exception as e:
            logger.error(f"Failed to load ground truth for {document_id}: {e}")
            return None
    
    def load_document(self, document_id: str) -> Optional[Document]:
        """Load a single document by ID."""
        doc_path = self.documents_dir / f"{document_id}.pdf"
        
        if not doc_path.exists():
            logger.warning(f"Document not found: {doc_path}")
            return None
        
        ground_truth = self._load_ground_truth(document_id)
        
        return Document(
            document_id=document_id,
            file_path=doc_path,
            ground_truth=ground_truth,
            schema_name=ground_truth.schema_name if ground_truth else None,
        )
    
    def list_documents(self) -> List[str]:
        """List all document IDs in the dataset."""
        if not self.documents_dir.exists():
            return []
        return [p.stem for p in sorted(self.documents_dir.glob("*.pdf"))]
    
    def add_ground_truth(
        self, 
        document_id: str, 
        schema_name: str,
        fields: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add ground truth for a document.
        
        Args:
            document_id: Document identifier
            schema_name: Schema name
            fields: Ground truth field values
            metadata: Optional metadata
        """
        self.ground_truth_dir.mkdir(parents=True, exist_ok=True)
        
        gt_data = {
            'document_id': document_id,
            'schema': schema_name,
            'fields': fields,
            'metadata': metadata or {},
        }
        
        gt_path = self.ground_truth_dir / f"{document_id}.json"
        with open(gt_path, 'w') as f:
            json.dump(gt_data, f, indent=2)
        
        logger.info(f"Added ground truth for {document_id}")
    
    @classmethod
    def create_dataset(
        cls,
        dataset_dir: Path,
        name: str,
        description: str = ""
    ) -> "DatasetLoader":
        """
        Create a new empty dataset structure.
        
        Args:
            dataset_dir: Directory for the new dataset
            name: Dataset name
            description: Dataset description
        
        Returns:
            DatasetLoader for the new dataset
        """
        dataset_dir = Path(dataset_dir)
        
        # Create directories
        (dataset_dir / "documents").mkdir(parents=True, exist_ok=True)
        (dataset_dir / "ground_truth").mkdir(parents=True, exist_ok=True)
        
        # Create metadata file
        metadata = {
            'name': name,
            'description': description,
            'created_at': datetime.utcnow().isoformat(),
            'documents': {},
        }
        
        with open(dataset_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Created dataset structure at: {dataset_dir}")
        
        return cls(dataset_dir)


