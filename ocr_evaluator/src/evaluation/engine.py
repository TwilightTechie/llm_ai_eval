"""
Main evaluation engine with multiprocessing support.
Orchestrates the complete evaluation workflow.
"""

import os
import json
import time
import resource
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
import traceback

import yaml

from ..providers import ProviderRegistry, OCRProvider, OCRResult
from ..dataset import DatasetLoader, Dataset, Document, SchemaManager
from .ragas_evaluator import RagasEvaluator, DocumentEvaluation
from ..utils.logging_config import get_logger, PerformanceLogger, setup_logging

logger = get_logger(__name__)


@dataclass
class ProviderResult:
    """Results for a single provider across all documents."""
    provider_name: str
    document_evaluations: List[DocumentEvaluation]
    ocr_results: List[OCRResult]
    aggregate_metrics: Dict[str, float] = field(default_factory=dict)
    total_processing_time_ms: float = 0.0
    total_tokens: int = 0
    errors: List[Dict[str, Any]] = field(default_factory=list)
    
    def calculate_aggregates(self):
        """Calculate aggregate metrics across all documents."""
        if not self.document_evaluations:
            return
        
        # Collect all metric values from RAGAS evaluations
        metric_values: Dict[str, List[float]] = {}
        
        for doc_eval in self.document_evaluations:
            for metric_name, value in doc_eval.aggregate_metrics.items():
                if isinstance(value, (int, float)) and not metric_name.startswith('error'):
                    if metric_name not in metric_values:
                        metric_values[metric_name] = []
                    metric_values[metric_name].append(value)
        
        # Calculate averages for all RAGAS metrics
        for metric_name, values in metric_values.items():
            if values:
                self.aggregate_metrics[metric_name] = sum(values) / len(values)
        
        # Calculate totals
        self.total_processing_time_ms = sum(
            r.processing_time_ms for r in self.ocr_results
        )
        self.total_tokens = sum(
            r.input_tokens + r.output_tokens for r in self.ocr_results
        )
        
        # Success rate (OCR extraction success)
        successful = sum(1 for r in self.ocr_results if r.success)
        self.aggregate_metrics['ocr_success_rate'] = successful / len(self.ocr_results) if self.ocr_results else 0
        
        # Average processing time
        if self.ocr_results:
            self.aggregate_metrics['avg_processing_time_ms'] = (
                self.total_processing_time_ms / len(self.ocr_results)
            )


@dataclass
class EvaluationResult:
    """Complete evaluation result across all providers."""
    evaluation_id: str
    dataset_name: str
    timestamp: datetime
    provider_results: Dict[str, ProviderResult]
    config: Dict[str, Any]
    evaluation_name: str = ""
    summary: Dict[str, Any] = field(default_factory=dict)
    
    def calculate_summary(self):
        """Calculate summary statistics using RAGAS metrics."""
        self.summary = {
            'total_documents': 0,
            'total_providers': len(self.provider_results),
            'best_provider': None,
            'provider_rankings': [],
            'ragas_metrics_used': [],
        }
        
        rankings = []
        for provider_name, result in self.provider_results.items():
            result.calculate_aggregates()
            
            self.summary['total_documents'] = max(
                self.summary['total_documents'],
                len(result.document_evaluations)
            )
            
            # Use RAGAS metrics for ranking (prioritize factual_correctness, then semantic_similarity)
            primary_score = result.aggregate_metrics.get(
                'avg_factual_correctness',
                result.aggregate_metrics.get('avg_semantic_similarity', 
                result.aggregate_metrics.get('avg_answer_correctness', 0))
            )
            
            ranking_entry = {
                'provider': provider_name,
                'primary_score': primary_score,
                'avg_factual_correctness': result.aggregate_metrics.get('avg_factual_correctness', None),
                'avg_semantic_similarity': result.aggregate_metrics.get('avg_semantic_similarity', None),
                'avg_answer_correctness': result.aggregate_metrics.get('avg_answer_correctness', None),
                'avg_answer_relevancy': result.aggregate_metrics.get('avg_answer_relevancy', None),
                'avg_faithfulness': result.aggregate_metrics.get('avg_faithfulness', None),
                'ocr_success_rate': result.aggregate_metrics.get('ocr_success_rate', 0),
                'avg_processing_time_ms': result.aggregate_metrics.get('avg_processing_time_ms', 0),
                'document_count': len(result.document_evaluations),
                'error_count': len(result.errors),
            }
            rankings.append(ranking_entry)
        
        # Sort by primary score (highest first)
        rankings.sort(key=lambda x: x['primary_score'] or 0, reverse=True)
        self.summary['provider_rankings'] = rankings
        
        if rankings:
            self.summary['best_provider'] = rankings[0]['provider']
        
        # Track which RAGAS metrics were used
        if rankings and rankings[0]:
            for key in rankings[0]:
                if key.startswith('avg_') and rankings[0][key] is not None:
                    metric_name = key.replace('avg_', '')
                    self.summary['ragas_metrics_used'].append(metric_name)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'evaluation_id': self.evaluation_id,
            'evaluation_name': self.evaluation_name,
            'dataset_name': self.dataset_name,
            'timestamp': self.timestamp.isoformat(),
            'summary': self.summary,
            'config': self.config,
            'provider_results': {
                name: {
                    'provider_name': result.provider_name,
                    'aggregate_metrics': result.aggregate_metrics,
                    'total_processing_time_ms': result.total_processing_time_ms,
                    'total_tokens': result.total_tokens,
                    'document_count': len(result.document_evaluations),
                    'error_count': len(result.errors),
                }
                for name, result in self.provider_results.items()
            },
        }
    
    def save(self, output_path: Path):
        """Save results to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        
        logger.info(f"Saved evaluation results to: {output_path}")


def _set_resource_limits(max_memory_mb: int):
    """Set resource limits for worker process."""
    if max_memory_mb > 0:
        try:
            # Convert MB to bytes
            max_bytes = max_memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes))
        except (ValueError, resource.error) as e:
            logger.warning(f"Could not set memory limit: {e}")


def _worker_init(max_memory_mb: int):
    """Initialize worker process."""
    _set_resource_limits(max_memory_mb)


def _process_document(
    document_path: str,
    document_id: str,
    ground_truth_fields: Dict[str, Any],
    schema_dict: Dict[str, Any],
    provider_name: str,
    provider_config: Dict[str, Any],
    eval_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Process a single document (runs in worker process).
    
    Returns dict with ocr_result and evaluation.
    """
    try:
        # Re-initialize provider in worker process
        provider = ProviderRegistry.get_provider(provider_name, provider_config, force_new=True)
        
        # Run OCR
        ocr_result = provider.extract(
            Path(document_path),
            schema_dict,
            document_id
        )
        
        # Initialize evaluator and run evaluation
        evaluator = RagasEvaluator(eval_config)
        
        evaluation = evaluator.evaluate_document(
            document_id=document_id,
            provider_name=provider_name,
            extracted_fields=ocr_result.fields,
            ground_truth_fields=ground_truth_fields,
            context=ocr_result.raw_text or "",
        )
        
        return {
            'success': True,
            'document_id': document_id,
            'ocr_result': ocr_result.to_dict(),
            'evaluation': {
                'document_id': evaluation.document_id,
                'provider_name': evaluation.provider_name,
                'field_evaluations': [fe.to_dict() for fe in evaluation.field_evaluations],
                'aggregate_metrics': evaluation.aggregate_metrics,
            },
        }
        
    except Exception as e:
        logger.error(f"Worker error processing {document_id}: {e}")
        return {
            'success': False,
            'document_id': document_id,
            'error': str(e),
            'traceback': traceback.format_exc(),
        }


class EvaluationEngine:
    """
    Main evaluation engine.
    Orchestrates OCR extraction and RAGAS evaluation with multiprocessing.
    """
    
    def __init__(self, config_path: Optional[Path] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize evaluation engine.
        
        Args:
            config_path: Path to evaluation.yaml
            config: Configuration dictionary (overrides config_path)
        """
        self.config = config or {}
        
        if config_path and not config:
            config_path = Path(config_path)
            if config_path.exists():
                with open(config_path) as f:
                    self.config = yaml.safe_load(f)
        
        # Setup logging from config
        setup_logging(self.config)
        
        # Multiprocessing settings
        mp_config = self.config.get('multiprocessing', {})
        self.use_multiprocessing = mp_config.get('enabled', True)
        self.max_workers = mp_config.get('max_workers', 4)
        self.batch_size = mp_config.get('batch_size', 10)
        self.timeout_seconds = mp_config.get('timeout_seconds', 300)
        
        # Resource limits
        resources = mp_config.get('resources', {})
        self.max_memory_mb = resources.get('max_memory_mb', 2048)
        
        # Schema manager
        self.schema_manager = None
        
        logger.info(
            f"Evaluation engine initialized: "
            f"multiprocessing={self.use_multiprocessing}, "
            f"workers={self.max_workers}, "
            f"batch_size={self.batch_size}"
        )
    
    def load_schemas(self, schemas_dir: Path):
        """Load extraction schemas."""
        self.schema_manager = SchemaManager(schemas_dir)
        logger.info(f"Loaded schemas: {self.schema_manager.list_schemas()}")
    
    @PerformanceLogger(get_logger(__name__), "Full evaluation run")
    def run_evaluation(
        self,
        dataset: Dataset,
        providers: List[str],
        provider_configs: Dict[str, Dict[str, Any]],
        schema_name: str,
        evaluation_name: Optional[str] = None,
        evaluation_id: Optional[str] = None,
    ) -> EvaluationResult:
        """
        Run complete evaluation across providers and documents.
        
        Args:
            dataset: Dataset to evaluate
            providers: List of provider names to use
            provider_configs: Configuration for each provider
            schema_name: Schema to use for extraction
            evaluation_name: Human-readable name for this evaluation
            evaluation_id: Unique identifier for this run
        
        Returns:
            EvaluationResult with all metrics
        """
        if evaluation_id is None:
            evaluation_id = f"eval_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        if evaluation_name is None:
            evaluation_name = evaluation_id
        
        logger.info(
            f"Starting evaluation '{evaluation_name}' ({evaluation_id}): "
            f"{len(dataset)} documents, {len(providers)} providers"
        )
        
        # Get schema
        if self.schema_manager:
            schema = self.schema_manager.get_schema(schema_name)
            if schema:
                schema_dict = schema.to_dict()
            else:
                logger.warning(f"Schema {schema_name} not found, using empty schema")
                schema_dict = {'schema_name': schema_name, 'fields': []}
        else:
            schema_dict = {'schema_name': schema_name, 'fields': []}
        
        # Run evaluation for each provider
        provider_results = {}
        
        for provider_name in providers:
            logger.info(f"Evaluating provider: {provider_name}")
            
            provider_config = provider_configs.get(provider_name, {})
            
            result = self._evaluate_provider(
                dataset=dataset,
                provider_name=provider_name,
                provider_config=provider_config,
                schema_dict=schema_dict,
            )
            
            provider_results[provider_name] = result
        
        # Build final result
        eval_result = EvaluationResult(
            evaluation_id=evaluation_id,
            evaluation_name=evaluation_name,
            dataset_name=dataset.name,
            timestamp=datetime.utcnow(),
            provider_results=provider_results,
            config=self.config,
        )
        
        eval_result.calculate_summary()
        
        logger.info(
            f"Evaluation '{evaluation_name}' complete: "
            f"best_provider={eval_result.summary.get('best_provider')}"
        )
        
        return eval_result
    
    def _evaluate_provider(
        self,
        dataset: Dataset,
        provider_name: str,
        provider_config: Dict[str, Any],
        schema_dict: Dict[str, Any],
    ) -> ProviderResult:
        """Evaluate a single provider across all documents."""
        
        # Prepare document tasks
        tasks = []
        for doc in dataset:
            if not doc.ground_truth:
                logger.warning(f"Skipping {doc.document_id}: no ground truth")
                continue
            
            tasks.append({
                'document_path': str(doc.file_path),
                'document_id': doc.document_id,
                'ground_truth_fields': doc.ground_truth.fields,
            })
        
        if not tasks:
            logger.warning(f"No documents with ground truth for {provider_name}")
            return ProviderResult(
                provider_name=provider_name,
                document_evaluations=[],
                ocr_results=[],
            )
        
        # Process documents
        if self.use_multiprocessing and len(tasks) > 1:
            results = self._process_parallel(
                tasks, provider_name, provider_config, schema_dict
            )
        else:
            results = self._process_sequential(
                tasks, provider_name, provider_config, schema_dict
            )
        
        # Collect results
        ocr_results = []
        document_evaluations = []
        errors = []
        
        for result in results:
            if result['success']:
                # Reconstruct OCRResult
                ocr_data = result['ocr_result']
                ocr_result = OCRResult(
                    document_id=ocr_data['document_id'],
                    provider_name=ocr_data['provider_name'],
                    fields=ocr_data['fields'],
                    raw_text=ocr_data.get('raw_text'),
                    success=ocr_data['success'],
                    error_message=ocr_data.get('error_message'),
                    processing_time_ms=ocr_data['processing_time_ms'],
                    input_tokens=ocr_data.get('input_tokens', 0),
                    output_tokens=ocr_data.get('output_tokens', 0),
                )
                ocr_results.append(ocr_result)
                
                # Reconstruct DocumentEvaluation
                eval_data = result['evaluation']
                from .ragas_evaluator import FieldEvaluation
                
                field_evals = [
                    FieldEvaluation(
                        field_name=fe['field_name'],
                        question=fe.get('question', f"What is the {fe['field_name'].replace('_', ' ')}?"),
                        expected_value=fe['expected_value'],
                        extracted_value=fe['extracted_value'],
                        context=fe.get('context', ''),
                        factual_correctness=fe.get('factual_correctness'),
                        semantic_similarity=fe.get('semantic_similarity'),
                        answer_correctness=fe.get('answer_correctness'),
                        answer_relevancy=fe.get('answer_relevancy'),
                        faithfulness=fe.get('faithfulness'),
                        metrics=fe.get('metrics', {}),
                    )
                    for fe in eval_data['field_evaluations']
                ]
                
                doc_eval = DocumentEvaluation(
                    document_id=eval_data['document_id'],
                    provider_name=eval_data['provider_name'],
                    field_evaluations=field_evals,
                    aggregate_metrics=eval_data['aggregate_metrics'],
                )
                document_evaluations.append(doc_eval)
            else:
                errors.append({
                    'document_id': result['document_id'],
                    'error': result.get('error'),
                    'traceback': result.get('traceback'),
                })
        
        return ProviderResult(
            provider_name=provider_name,
            document_evaluations=document_evaluations,
            ocr_results=ocr_results,
            errors=errors,
        )
    
    def _process_sequential(
        self,
        tasks: List[Dict[str, Any]],
        provider_name: str,
        provider_config: Dict[str, Any],
        schema_dict: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Process documents sequentially."""
        results = []
        
        # Initialize provider once
        provider = ProviderRegistry.get_provider(provider_name, provider_config)
        evaluator = RagasEvaluator(self.config)
        
        for i, task in enumerate(tasks):
            logger.info(
                f"Processing {task['document_id']} ({i + 1}/{len(tasks)})"
            )
            
            try:
                with PerformanceLogger(
                    logger, 
                    f"Document: {task['document_id']}",
                    extra={'document_id': task['document_id']}
                ):
                    # Run OCR
                    ocr_result = provider.extract(
                        Path(task['document_path']),
                        schema_dict,
                        task['document_id']
                    )
                    
                    # Run evaluation
                    evaluation = evaluator.evaluate_document(
                        document_id=task['document_id'],
                        provider_name=provider_name,
                        extracted_fields=ocr_result.fields,
                        ground_truth_fields=task['ground_truth_fields'],
                        context=ocr_result.raw_text or "",
                    )
                    
                    results.append({
                        'success': True,
                        'document_id': task['document_id'],
                        'ocr_result': ocr_result.to_dict(),
                        'evaluation': {
                            'document_id': evaluation.document_id,
                            'provider_name': evaluation.provider_name,
                            'field_evaluations': [fe.to_dict() for fe in evaluation.field_evaluations],
                            'aggregate_metrics': evaluation.aggregate_metrics,
                        },
                    })
                    
            except Exception as e:
                logger.error(f"Error processing {task['document_id']}: {e}")
                results.append({
                    'success': False,
                    'document_id': task['document_id'],
                    'error': str(e),
                    'traceback': traceback.format_exc(),
                })
        
        return results
    
    def _process_parallel(
        self,
        tasks: List[Dict[str, Any]],
        provider_name: str,
        provider_config: Dict[str, Any],
        schema_dict: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Process documents in parallel using multiprocessing."""
        results = []
        
        logger.info(
            f"Processing {len(tasks)} documents with {self.max_workers} workers"
        )
        
        # Use ThreadPoolExecutor instead of ProcessPoolExecutor
        # because RAGAS and OpenAI clients don't pickle well
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {}
            
            for task in tasks:
                future = executor.submit(
                    _process_document,
                    task['document_path'],
                    task['document_id'],
                    task['ground_truth_fields'],
                    schema_dict,
                    provider_name,
                    provider_config,
                    self.config,
                )
                future_to_task[future] = task
            
            # Collect results
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result(timeout=self.timeout_seconds)
                    results.append(result)
                    
                    status = "✓" if result['success'] else "✗"
                    logger.info(f"{status} Completed: {task['document_id']}")
                    
                except TimeoutError:
                    logger.error(f"Timeout processing {task['document_id']}")
                    results.append({
                        'success': False,
                        'document_id': task['document_id'],
                        'error': 'Timeout',
                    })
                except Exception as e:
                    logger.error(f"Error processing {task['document_id']}: {e}")
                    results.append({
                        'success': False,
                        'document_id': task['document_id'],
                        'error': str(e),
                        'traceback': traceback.format_exc(),
                    })
        
        return results

