"""
Schema management for field extraction.
Handles loading, validation, and management of extraction schemas.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import yaml

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class FieldSchema:
    """Schema definition for a single field."""
    name: str
    field_type: str = "string"
    required: bool = False
    description: str = ""
    format: Optional[str] = None  # For dates, etc.
    item_schema: Optional[List[Dict]] = None  # For list types
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FieldSchema":
        """Create FieldSchema from dictionary."""
        return cls(
            name=data['name'],
            field_type=data.get('type', 'string'),
            required=data.get('required', False),
            description=data.get('description', ''),
            format=data.get('format'),
            item_schema=data.get('item_schema'),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'name': self.name,
            'type': self.field_type,
            'required': self.required,
            'description': self.description,
        }
        if self.format:
            result['format'] = self.format
        if self.item_schema:
            result['item_schema'] = self.item_schema
        return result


@dataclass
class ExtractionSchema:
    """Complete extraction schema for a document type."""
    schema_name: str
    version: str = "1.0"
    description: str = ""
    fields: List[FieldSchema] = field(default_factory=list)
    extraction_prompt: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExtractionSchema":
        """Create ExtractionSchema from dictionary."""
        fields = [
            FieldSchema.from_dict(f) for f in data.get('fields', [])
        ]
        return cls(
            schema_name=data.get('schema_name', 'unknown'),
            version=data.get('version', '1.0'),
            description=data.get('description', ''),
            fields=fields,
            extraction_prompt=data.get('extraction_prompt'),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for provider use."""
        return {
            'schema_name': self.schema_name,
            'version': self.version,
            'description': self.description,
            'fields': [f.to_dict() for f in self.fields],
            'extraction_prompt': self.extraction_prompt,
        }
    
    def get_required_fields(self) -> List[str]:
        """Get list of required field names."""
        return [f.name for f in self.fields if f.required]
    
    def get_field(self, name: str) -> Optional[FieldSchema]:
        """Get a field by name."""
        for field in self.fields:
            if field.name == name:
                return field
        return None
    
    def validate_extraction(self, extracted: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate extracted data against schema.
        
        Returns:
            Dictionary with validation results
        """
        results = {
            'valid': True,
            'missing_required': [],
            'type_mismatches': [],
            'extra_fields': [],
        }
        
        # Check required fields
        for field in self.fields:
            if field.required:
                if field.name not in extracted or extracted[field.name] is None:
                    results['missing_required'].append(field.name)
                    results['valid'] = False
        
        # Check for extra fields
        field_names = {f.name for f in self.fields}
        for key in extracted.keys():
            if key not in field_names:
                results['extra_fields'].append(key)
        
        return results


class SchemaManager:
    """
    Manages extraction schemas.
    Loads schemas from YAML files and provides access by name.
    """
    
    def __init__(self, schemas_dir: Optional[Path] = None):
        """
        Initialize schema manager.
        
        Args:
            schemas_dir: Directory containing schema YAML files
        """
        self.schemas_dir = Path(schemas_dir) if schemas_dir else None
        self._schemas: Dict[str, ExtractionSchema] = {}
        
        if self.schemas_dir and self.schemas_dir.exists():
            self._load_schemas()
    
    def _load_schemas(self):
        """Load all schemas from the schemas directory."""
        if not self.schemas_dir:
            return
        
        for schema_file in self.schemas_dir.glob("*.yaml"):
            try:
                self.load_schema(schema_file)
            except Exception as e:
                logger.error(f"Failed to load schema {schema_file}: {e}")
    
    def load_schema(self, schema_path: Path) -> ExtractionSchema:
        """
        Load a schema from a YAML file.
        
        Args:
            schema_path: Path to schema YAML file
        
        Returns:
            Loaded ExtractionSchema
        """
        schema_path = Path(schema_path)
        
        with open(schema_path) as f:
            data = yaml.safe_load(f)
        
        schema = ExtractionSchema.from_dict(data)
        self._schemas[schema.schema_name] = schema
        
        logger.info(
            f"Loaded schema: {schema.schema_name} v{schema.version} "
            f"({len(schema.fields)} fields)"
        )
        
        return schema
    
    def get_schema(self, name: str) -> Optional[ExtractionSchema]:
        """Get a schema by name."""
        return self._schemas.get(name)
    
    def register_schema(self, schema: ExtractionSchema):
        """Register a schema programmatically."""
        self._schemas[schema.schema_name] = schema
        logger.info(f"Registered schema: {schema.schema_name}")
    
    def list_schemas(self) -> List[str]:
        """List all available schema names."""
        return list(self._schemas.keys())
    
    def create_schema_from_fields(
        self,
        name: str,
        fields: List[Dict[str, Any]],
        description: str = ""
    ) -> ExtractionSchema:
        """
        Create and register a new schema from field definitions.
        
        Args:
            name: Schema name
            fields: List of field definitions
            description: Schema description
        
        Returns:
            Created ExtractionSchema
        """
        schema = ExtractionSchema(
            schema_name=name,
            description=description,
            fields=[FieldSchema.from_dict(f) for f in fields],
        )
        self.register_schema(schema)
        return schema
    
    def save_schema(self, schema_name: str, output_path: Optional[Path] = None):
        """
        Save a schema to YAML file.
        
        Args:
            schema_name: Name of schema to save
            output_path: Output path (defaults to schemas_dir)
        """
        schema = self.get_schema(schema_name)
        if not schema:
            raise ValueError(f"Schema not found: {schema_name}")
        
        if output_path is None:
            if not self.schemas_dir:
                raise ValueError("No schemas directory configured")
            output_path = self.schemas_dir / f"{schema_name}.yaml"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(schema.to_dict(), f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Saved schema to: {output_path}")


