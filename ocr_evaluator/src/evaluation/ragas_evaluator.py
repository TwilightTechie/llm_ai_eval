"""
RAGAS-based evaluation for OCR outputs.
Integrates RAGAS library (v0.3.9+) for comprehensive evaluation metrics.

All evaluation is done using RAGAS metrics - no simple string matching.
"""

import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from ragas import evaluate
from ragas.metrics import (
    FactualCorrectness,
    SemanticSimilarity,
    AnswerCorrectness,
    AnswerRelevancy,
    Faithfulness,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset as HFDataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from ..utils.logging_config import get_logger, PerformanceLogger

logger = get_logger(__name__)


@dataclass
class FieldEvaluation:
    """
    Evaluation result for a single field using RAGAS metrics.
    
    All scores are from RAGAS (0.0 to 1.0 scale).
    """
    field_name: str
    question: str  # The question asked (e.g., "What is the patient name?")
    expected_value: Any  # Ground truth value
    extracted_value: Any  # OCR extracted value
    context: str  # Document context used
    
    # RAGAS metric scores (all 0.0 to 1.0)
    factual_correctness: Optional[float] = None
    semantic_similarity: Optional[float] = None
    answer_correctness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    faithfulness: Optional[float] = None
    
    # Store all metrics in a dict for flexibility
    metrics: Dict[str, float] = field(default_factory=dict)
    
    def get_primary_score(self) -> float:
        """Get primary evaluation score (factual_correctness or semantic_similarity)."""
        if self.factual_correctness is not None:
            return self.factual_correctness
        if self.semantic_similarity is not None:
            return self.semantic_similarity
        if self.answer_correctness is not None:
            return self.answer_correctness
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'field_name': self.field_name,
            'question': self.question,
            'expected_value': self.expected_value,
            'extracted_value': self.extracted_value,
            'context': self.context[:500] if self.context else '',  # Truncate for readability
            'factual_correctness': self.factual_correctness,
            'semantic_similarity': self.semantic_similarity,
            'answer_correctness': self.answer_correctness,
            'answer_relevancy': self.answer_relevancy,
            'faithfulness': self.faithfulness,
            'metrics': self.metrics,
        }


@dataclass
class DocumentEvaluation:
    """Evaluation result for a single document using RAGAS metrics."""
    document_id: str
    provider_name: str
    field_evaluations: List[FieldEvaluation]
    aggregate_metrics: Dict[str, float] = field(default_factory=dict)
    raw_ragas_result: Optional[Dict[str, Any]] = None
    
    def get_field_evaluation(self, field_name: str) -> Optional[FieldEvaluation]:
        """Get evaluation for a specific field."""
        for fe in self.field_evaluations:
            if fe.field_name == field_name:
                return fe
        return None
    
    def get_average_score(self, metric_name: str) -> float:
        """Get average score for a specific metric across all fields."""
        scores = []
        for fe in self.field_evaluations:
            score = getattr(fe, metric_name, None)
            if score is not None:
                scores.append(score)
        return sum(scores) / len(scores) if scores else 0.0
    
    @property
    def avg_factual_correctness(self) -> float:
        return self.get_average_score('factual_correctness')
    
    @property
    def avg_semantic_similarity(self) -> float:
        return self.get_average_score('semantic_similarity')
    
    @property
    def avg_answer_correctness(self) -> float:
        return self.get_average_score('answer_correctness')
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'document_id': self.document_id,
            'provider_name': self.provider_name,
            'field_evaluations': [fe.to_dict() for fe in self.field_evaluations],
            'aggregate_metrics': self.aggregate_metrics,
        }


# Metric name mapping for config
METRIC_NAME_MAP = {
    'factual_correctness': 'FactualCorrectness',
    'semantic_similarity': 'SemanticSimilarity',
    'answer_correctness': 'AnswerCorrectness',
    'answer_relevancy': 'AnswerRelevancy',
    'faithfulness': 'Faithfulness',
}


class RagasEvaluator:
    """
    Evaluator using RAGAS library (v0.3.9+) for OCR output assessment.
    
    All evaluation is performed using RAGAS metrics - no simple string matching.
    
    Supported metrics:
    - FactualCorrectness: Are extracted fields factually correct vs ground truth?
    - SemanticSimilarity: How semantically similar is extraction to ground truth?
    - AnswerCorrectness: Combined correctness score (factual + semantic)
    - AnswerRelevancy: Is the extraction relevant to the question asked?
    - Faithfulness: Does extraction stay faithful to the source document context?
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize RAGAS evaluator.
        
        Args:
            config: Evaluation configuration including:
                - metrics.enabled: List of metrics to use
                - metrics.llm.model: LLM model for evaluation
                - metrics.llm.api_key_env: Environment variable for API key
        """
        self.config = config
        
        # Get LLM settings
        llm_config = config.get('metrics', {}).get('llm', {})
        api_key_env = llm_config.get('api_key_env', 'OPENAI_API_KEY')
        api_key = os.environ.get(api_key_env)
        
        if not api_key:
            raise ValueError(f"API key not found in {api_key_env}")
        
        model = llm_config.get('model', 'gpt-4o-mini')
        temperature = llm_config.get('temperature', 0.0)
        
        # Initialize LLM and embeddings for RAGAS (v0.3.9+ API)
        base_llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key,
        )
        base_embeddings = OpenAIEmbeddings(api_key=api_key)
        
        # Wrap for RAGAS
        self.llm = LangchainLLMWrapper(base_llm)
        self.embeddings = LangchainEmbeddingsWrapper(base_embeddings)
        
        # Get enabled metrics from config
        enabled_metric_names = config.get('metrics', {}).get('enabled', [
            'factual_correctness',
            'semantic_similarity',
            'answer_correctness',
        ])
        
        # Initialize metric instances
        self.metrics = []
        self.metric_names = []
        
        for metric_name in enabled_metric_names:
            metric_instance = self._create_metric(metric_name)
            if metric_instance:
                self.metrics.append(metric_instance)
                self.metric_names.append(metric_name)
                logger.debug(f"Enabled RAGAS metric: {metric_name}")
            else:
                logger.warning(f"Could not initialize metric: {metric_name}")
        
        if not self.metrics:
            # Default to core metrics
            logger.warning("No valid metrics specified, using defaults")
            self.metrics = [
                SemanticSimilarity(embeddings=self.embeddings),
                AnswerCorrectness(llm=self.llm),
            ]
            self.metric_names = ['semantic_similarity', 'answer_correctness']
        
        logger.info(
            f"RAGAS evaluator initialized with {len(self.metrics)} metrics: "
            f"{self.metric_names}, model: {model}"
        )
    
    def _create_metric(self, metric_name: str):
        """Create a metric instance by name."""
        try:
            if metric_name == 'factual_correctness':
                return FactualCorrectness(llm=self.llm)
            elif metric_name == 'semantic_similarity':
                return SemanticSimilarity(embeddings=self.embeddings)
            elif metric_name == 'answer_correctness':
                return AnswerCorrectness(llm=self.llm)
            elif metric_name == 'answer_relevancy':
                return AnswerRelevancy(llm=self.llm, embeddings=self.embeddings)
            elif metric_name == 'faithfulness':
                return Faithfulness(llm=self.llm)
            else:
                logger.warning(f"Unknown metric: {metric_name}")
                return None
        except Exception as e:
            logger.error(f"Failed to create metric {metric_name}: {e}")
            return None
    
    def _prepare_ragas_dataset(
        self,
        extracted_fields: Dict[str, Any],
        ground_truth_fields: Dict[str, Any],
        context: str = ""
    ) -> tuple[HFDataset, List[str]]:
        """
        Prepare data in RAGAS v0.3.9 expected format.
        
        RAGAS v0.3.9 expects:
        - user_input: The question/query
        - response: The answer/extracted value
        - reference: Ground truth/expected value
        - retrieved_contexts: List of context strings
        
        Returns:
            Tuple of (HFDataset, list of field names in order)
        """
        user_inputs = []
        responses = []
        references = []
        retrieved_contexts = []
        field_names_order = []
        
        for field_name, expected_value in ground_truth_fields.items():
            extracted_value = extracted_fields.get(field_name)
            
            # Format values as strings
            if isinstance(expected_value, list):
                if expected_value and isinstance(expected_value[0], dict):
                    # Handle list of dicts (e.g., medications)
                    expected_str = "; ".join(
                        ", ".join(f"{k}: {v}" for k, v in item.items())
                        for item in expected_value
                    )
                else:
                    expected_str = ", ".join(str(v) for v in expected_value)
            elif expected_value is None:
                expected_str = ""
            else:
                expected_str = str(expected_value)
            
            if isinstance(extracted_value, list):
                if extracted_value and isinstance(extracted_value[0], dict):
                    extracted_str = "; ".join(
                        ", ".join(f"{k}: {v}" for k, v in item.items())
                        for item in extracted_value
                    )
                else:
                    extracted_str = ", ".join(str(v) for v in extracted_value)
            elif extracted_value is None:
                extracted_str = ""
            else:
                extracted_str = str(extracted_value)
            
            # Create question from field name
            question = f"What is the {field_name.replace('_', ' ')}?"
            
            user_inputs.append(question)
            responses.append(extracted_str)
            references.append(expected_str)
            retrieved_contexts.append([context] if context else ["Document content not available"])
            field_names_order.append(field_name)
        
        dataset = HFDataset.from_dict({
            'user_input': user_inputs,
            'response': responses,
            'reference': references,
            'retrieved_contexts': retrieved_contexts,
        })
        
        return dataset, field_names_order
    
    @PerformanceLogger(get_logger(__name__), "RAGAS evaluation")
    def evaluate_document(
        self,
        document_id: str,
        provider_name: str,
        extracted_fields: Dict[str, Any],
        ground_truth_fields: Dict[str, Any],
        context: str = ""
    ) -> DocumentEvaluation:
        """
        Evaluate extracted fields against ground truth using RAGAS metrics.
        
        Args:
            document_id: Document identifier
            provider_name: Name of the OCR provider
            extracted_fields: Fields extracted by OCR
            ground_truth_fields: Expected field values
            context: Document text context (from OCR raw text)
        
        Returns:
            DocumentEvaluation with per-field RAGAS metrics
        """
        logger.info(f"Evaluating document {document_id} from {provider_name} using RAGAS")
        
        # Prepare dataset
        dataset, field_names_order = self._prepare_ragas_dataset(
            extracted_fields, ground_truth_fields, context
        )
        
        if len(dataset) == 0:
            logger.warning(f"No fields to evaluate for {document_id}")
            return DocumentEvaluation(
                document_id=document_id,
                provider_name=provider_name,
                field_evaluations=[],
            )
        
        try:
            # Run RAGAS evaluation
            logger.debug(f"Running RAGAS with metrics: {self.metric_names}")
            result = evaluate(
                dataset=dataset,
                metrics=self.metrics,
            )
            
            # Convert to DataFrame for easier processing
            result_df = result.to_pandas()
            logger.debug(f"RAGAS result columns: {list(result_df.columns)}")
            
            # Build field evaluations with RAGAS scores
            field_evaluations = []
            
            for i, field_name in enumerate(field_names_order):
                if i >= len(result_df):
                    break
                
                row = result_df.iloc[i]
                
                # Extract RAGAS scores
                metrics_dict = {}
                factual_correctness_score = None
                semantic_similarity_score = None
                answer_correctness_score = None
                answer_relevancy_score = None
                faithfulness_score = None
                
                # Map column names to our metrics
                for col in result_df.columns:
                    col_lower = col.lower()
                    value = row[col]
                    
                    # Handle NaN values
                    if value != value:  # NaN check
                        value = None
                    elif isinstance(value, (int, float)):
                        value = float(value)
                    else:
                        continue
                    
                    if value is not None:
                        metrics_dict[col] = value
                        
                        if 'factual' in col_lower:
                            factual_correctness_score = value
                        elif 'semantic' in col_lower:
                            semantic_similarity_score = value
                        elif 'answer_correctness' in col_lower or col_lower == 'answer_correctness':
                            answer_correctness_score = value
                        elif 'relevancy' in col_lower:
                            answer_relevancy_score = value
                        elif 'faithful' in col_lower:
                            faithfulness_score = value
                
                field_eval = FieldEvaluation(
                    field_name=field_name,
                    question=row.get('user_input', f"What is the {field_name.replace('_', ' ')}?"),
                    expected_value=ground_truth_fields[field_name],
                    extracted_value=extracted_fields.get(field_name),
                    context=context[:500] if context else "",
                    factual_correctness=factual_correctness_score,
                    semantic_similarity=semantic_similarity_score,
                    answer_correctness=answer_correctness_score,
                    answer_relevancy=answer_relevancy_score,
                    faithfulness=faithfulness_score,
                    metrics=metrics_dict,
                )
                field_evaluations.append(field_eval)
            
            # Calculate aggregate metrics
            aggregate_metrics = {}
            for metric_name in self.metric_names:
                scores = []
                for fe in field_evaluations:
                    score = getattr(fe, metric_name, None)
                    if score is not None:
                        scores.append(score)
                if scores:
                    aggregate_metrics[f"avg_{metric_name}"] = sum(scores) / len(scores)
            
            # Log summary
            summary_parts = []
            for key, value in aggregate_metrics.items():
                summary_parts.append(f"{key}={value:.2%}")
            logger.info(f"RAGAS evaluation complete for {document_id}: {', '.join(summary_parts)}")
            
            return DocumentEvaluation(
                document_id=document_id,
                provider_name=provider_name,
                field_evaluations=field_evaluations,
                aggregate_metrics=aggregate_metrics,
                raw_ragas_result=result_df.to_dict(),
            )
            
        except Exception as e:
            logger.error(f"RAGAS evaluation failed for {document_id}: {e}", exc_info=True)
            
            # Return evaluation with error flag
            field_evaluations = []
            for field_name, expected in ground_truth_fields.items():
                extracted = extracted_fields.get(field_name)
                field_evaluations.append(FieldEvaluation(
                    field_name=field_name,
                    question=f"What is the {field_name.replace('_', ' ')}?",
                    expected_value=expected,
                    extracted_value=extracted,
                    context=context[:500] if context else "",
                    metrics={'error': str(e)},
                ))
            
            return DocumentEvaluation(
                document_id=document_id,
                provider_name=provider_name,
                field_evaluations=field_evaluations,
                aggregate_metrics={'error': str(e)},
            )
    
    def evaluate_batch(
        self,
        evaluations: List[Dict[str, Any]]
    ) -> List[DocumentEvaluation]:
        """
        Evaluate multiple documents.
        
        Args:
            evaluations: List of dicts with:
                - document_id
                - provider_name
                - extracted_fields
                - ground_truth_fields
                - context (optional)
        
        Returns:
            List of DocumentEvaluation results
        """
        results = []
        
        for item in evaluations:
            result = self.evaluate_document(
                document_id=item['document_id'],
                provider_name=item['provider_name'],
                extracted_fields=item['extracted_fields'],
                ground_truth_fields=item['ground_truth_fields'],
                context=item.get('context', ''),
            )
            results.append(result)
        
        return results
    
    @classmethod
    def get_available_metrics(cls) -> List[str]:
        """Get list of available metric names."""
        return [
            'factual_correctness',
            'semantic_similarity',
            'answer_correctness',
            'answer_relevancy',
            'faithfulness',
        ]
