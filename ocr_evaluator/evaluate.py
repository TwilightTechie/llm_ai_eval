#!/usr/bin/env python3
"""
OCR Evaluation CLI

Command-line interface for running OCR evaluations.

Usage:
    python evaluate.py --providers openai --dataset datasets/sample --schema prescription
    python evaluate.py --providers openai,textract --dataset datasets/medical --schema claim --output reports/
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import yaml

from src.utils.logging_config import setup_logging, get_logger
from src.providers import ProviderRegistry
from src.dataset import DatasetLoader, SchemaManager
from src.evaluation import EvaluationEngine
from src.reporting import ReportGenerator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="OCR Evaluation Framework - Evaluate OCR providers using RAGAS metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run evaluation with OpenAI provider
  python evaluate.py --name "prescription_test_v1" --providers openai --dataset datasets/sample --schema prescription

  # Run with multiple providers
  python evaluate.py --name "multi_provider_comparison" --providers openai,textract --dataset datasets/medical --schema claim

  # Specify output directory
  python evaluate.py --name "prod_eval" --providers openai --dataset datasets/sample --schema prescription --output reports/

  # Use custom config
  python evaluate.py --name "config_test" --config config/evaluation.yaml --providers openai --dataset datasets/sample
        """
    )
    
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Evaluation name/identifier (used as prefix for logs and reports, e.g., 'prescription_test_v1')"
    )
    
    parser.add_argument(
        "--providers",
        type=str,
        required=True,
        help="Comma-separated list of providers to evaluate (e.g., openai,textract)"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset directory"
    )
    
    parser.add_argument(
        "--schema",
        type=str,
        required=True,
        help="Schema name to use for extraction (e.g., prescription, medical_claim)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="reports",
        help="Output directory for reports (default: reports)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/evaluation.yaml",
        help="Path to evaluation configuration file"
    )
    
    parser.add_argument(
        "--providers-config",
        type=str,
        default="config/providers.yaml",
        help="Path to providers configuration file"
    )
    
    parser.add_argument(
        "--schemas-dir",
        type=str,
        default="config/schemas",
        help="Directory containing schema YAML files"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (overrides config)"
    )
    
    parser.add_argument(
        "--no-multiprocessing",
        action="store_true",
        help="Disable multiprocessing (run sequentially)"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    parser.add_argument(
        "--json-output",
        action="store_true",
        help="Also output results as JSON"
    )
    
    parser.add_argument(
        "--description",
        type=str,
        default="",
        help="Optional description for this evaluation run"
    )
    
    return parser.parse_args()


def sanitize_name(name: str) -> str:
    """Sanitize evaluation name for use in filenames."""
    # Replace spaces and special chars with underscores
    import re
    sanitized = re.sub(r'[^\w\-]', '_', name)
    # Remove consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    # Remove leading/trailing underscores
    return sanitized.strip('_')


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def load_provider_configs(config_path: Path) -> dict:
    """Load provider configurations."""
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
            return config.get('providers', {})
    return {}


def main():
    """Main entry point."""
    args = parse_args()
    
    # Resolve paths relative to script location
    script_dir = Path(__file__).parent
    
    # Sanitize evaluation name
    eval_name = sanitize_name(args.name)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    eval_id = f"{eval_name}_{timestamp}"
    
    config_path = script_dir / args.config
    providers_config_path = script_dir / args.providers_config
    schemas_dir = script_dir / args.schemas_dir
    dataset_path = Path(args.dataset)
    output_dir = Path(args.output)
    
    # Create evaluation-specific log directory
    log_dir = script_dir / "logs" / eval_name
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configurations
    config = load_config(config_path)
    provider_configs = load_provider_configs(providers_config_path)
    
    # Override config with CLI args
    if args.workers:
        config.setdefault('multiprocessing', {})['max_workers'] = args.workers
    
    if args.no_multiprocessing:
        config.setdefault('multiprocessing', {})['enabled'] = False
    
    config.setdefault('logging', {})['level'] = args.log_level
    
    # Set evaluation-specific log file path
    config.setdefault('logging', {}).setdefault('file', {})['path'] = str(log_dir / f"{eval_id}.log")
    config['logging']['file']['enabled'] = True
    
    # Setup logging with evaluation name prefix
    setup_logging(config, log_dir=str(log_dir))
    logger = get_logger("cli")
    
    logger.info("=" * 60)
    logger.info("OCR Evaluation Framework")
    logger.info("=" * 60)
    logger.info(f"Evaluation Name: {args.name}")
    logger.info(f"Evaluation ID: {eval_id}")
    if args.description:
        logger.info(f"Description: {args.description}")
    logger.info(f"Log Directory: {log_dir}")
    
    # Parse providers
    providers = [p.strip() for p in args.providers.split(",")]
    logger.info(f"Providers: {providers}")
    logger.info(f"Dataset: {dataset_path}")
    logger.info(f"Schema: {args.schema}")
    
    # Validate dataset exists
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        sys.exit(1)
    
    # Load dataset
    try:
        dataset_loader = DatasetLoader(dataset_path)
        dataset = dataset_loader.load()
        logger.info(f"Loaded {len(dataset)} documents")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        sys.exit(1)
    
    # Initialize evaluation engine
    engine = EvaluationEngine(config=config)
    
    # Load schemas
    if schemas_dir.exists():
        engine.load_schemas(schemas_dir)
    else:
        logger.warning(f"Schemas directory not found: {schemas_dir}")
    
    # Filter provider configs for requested providers
    filtered_configs = {}
    for provider in providers:
        if provider in provider_configs:
            filtered_configs[provider] = provider_configs[provider]
        else:
            logger.warning(f"No config found for provider: {provider}, using defaults")
            filtered_configs[provider] = {}
    
    # Run evaluation
    try:
        logger.info("Starting evaluation...")
        result = engine.run_evaluation(
            dataset=dataset,
            providers=providers,
            provider_configs=filtered_configs,
            schema_name=args.schema,
            evaluation_name=args.name,
            evaluation_id=eval_id,
        )
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)
    
    # Generate reports in evaluation-specific directory
    eval_output_dir = output_dir / eval_name
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    
    # HTML report
    report_generator = ReportGenerator()
    html_path = eval_output_dir / f"{eval_id}_report.html"
    
    try:
        report_generator.generate_html(result, html_path)
        logger.info(f"HTML report generated: {html_path}")
    except Exception as e:
        logger.error(f"Failed to generate HTML report: {e}")
    
    # JSON output (always save for reference)
    json_path = eval_output_dir / f"{eval_id}_results.json"
    try:
        result.save(json_path)
        logger.info(f"JSON results saved: {json_path}")
    except Exception as e:
        logger.error(f"Failed to save JSON results: {e}")
    
    # Print summary
    logger.info("=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Evaluation Name: {args.name}")
    logger.info(f"Evaluation ID: {eval_id}")
    logger.info(f"Total Documents: {result.summary.get('total_documents', 0)}")
    logger.info(f"Total Providers: {result.summary.get('total_providers', 0)}")
    logger.info(f"Best Provider: {result.summary.get('best_provider', 'N/A')}")
    
    if result.summary.get('provider_rankings'):
        logger.info("\nProvider Rankings:")
        for i, ranking in enumerate(result.summary['provider_rankings'], 1):
            # Build metrics string from available RAGAS scores
            metrics_parts = []
            if ranking.get('avg_factual_correctness') is not None:
                metrics_parts.append(f"FC={ranking['avg_factual_correctness']:.1%}")
            if ranking.get('avg_semantic_similarity') is not None:
                metrics_parts.append(f"SS={ranking['avg_semantic_similarity']:.1%}")
            if ranking.get('avg_answer_correctness') is not None:
                metrics_parts.append(f"AC={ranking['avg_answer_correctness']:.1%}")
            
            metrics_str = ", ".join(metrics_parts) if metrics_parts else "No metrics"
            logger.info(f"  {i}. {ranking['provider']}: {metrics_str}")
    
    logger.info("=" * 60)
    logger.info("OUTPUT FILES:")
    logger.info(f"  📊 Report: {html_path}")
    logger.info(f"  📄 Results: {json_path}")
    logger.info(f"  📝 Logs: {log_dir / f'{eval_id}.log'}")
    logger.info("=" * 60)
    
    # Print to console as well (in case logging is file-only)
    print(f"\n✅ Evaluation '{args.name}' completed!")
    print(f"   📊 Report: {html_path}")
    print(f"   📄 Results: {json_path}")
    print(f"   📝 Logs: {log_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

