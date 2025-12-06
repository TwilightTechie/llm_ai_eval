"""
HTML report generator for OCR evaluation results.
Creates beautiful, shareable reports with detailed RAGAS metrics analysis.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..evaluation.engine import EvaluationResult, ProviderResult
from ..evaluation.ragas_evaluator import DocumentEvaluation, FieldEvaluation
from ..utils.logging_config import get_logger, PerformanceLogger
from .report_summarizer import ReportSummarizer

logger = get_logger(__name__)


class ReportGenerator:
    """
    Generates HTML reports from evaluation results with detailed RAGAS metrics.
    """
    
    def __init__(self, templates_dir: Optional[Path] = None):
        """
        Initialize report generator.
        
        Args:
            templates_dir: Directory containing Jinja2 templates
        """
        if templates_dir is None:
            templates_dir = Path(__file__).parent / "templates"
        
        self.templates_dir = Path(templates_dir)
        
        # Create templates directory if it doesn't exist
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure default template exists
        self._ensure_default_template()
        
        # Initialize Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            autoescape=select_autoescape(['html', 'xml']),
        )
        
        # Add custom filters
        self.env.filters['format_percent'] = self._format_percent
        self.env.filters['format_score'] = self._format_score
        self.env.filters['format_ms'] = lambda x: f"{x:.0f}ms" if isinstance(x, (int, float)) else x
        self.env.filters['format_number'] = lambda x: f"{x:,.2f}" if isinstance(x, float) else f"{x:,}" if isinstance(x, int) else x
        self.env.filters['truncate_text'] = lambda x, n=50: (str(x)[:n] + '...' if len(str(x)) > n else str(x)) if x else ''
        
        logger.info(f"Report generator initialized with templates from: {self.templates_dir}")
    
    @staticmethod
    def _format_percent(x):
        """Format as percentage."""
        if x is None:
            return 'N/A'
        if isinstance(x, (int, float)):
            return f"{x * 100:.1f}%"
        return str(x)
    
    @staticmethod
    def _format_score(x):
        """Format RAGAS score (0-1) as readable score."""
        if x is None:
            return '-'
        if isinstance(x, (int, float)):
            return f"{x:.3f}"
        return str(x)
    
    def _ensure_default_template(self):
        """Create default HTML template if it doesn't exist."""
        template_path = self.templates_dir / "report.html"
        
        if not template_path.exists():
            template_path.write_text(DEFAULT_HTML_TEMPLATE)
            logger.info(f"Created default template: {template_path}")
    
    @PerformanceLogger(get_logger(__name__), "Generating HTML report")
    def generate_html(
        self,
        result: EvaluationResult,
        output_path: Path,
        include_details: bool = True,
        max_samples: int = 20,
        include_ai_summary: bool = True,
    ) -> Path:
        """
        Generate HTML report from evaluation results.
        
        Args:
            result: EvaluationResult to report on
            output_path: Path for output HTML file
            include_details: Include detailed per-document results
            max_samples: Maximum number of sample documents to show
            include_ai_summary: Generate AI summary using OpenAI
        
        Returns:
            Path to generated report
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare template data
        template_data = self._prepare_template_data(
            result, include_details, max_samples
        )
        
        # Generate AI summary if requested
        ai_summary = None
        if include_ai_summary:
            try:
                summarizer = ReportSummarizer()
                ai_summary = summarizer.generate_summary(template_data)
                logger.info(f"AI summary generated successfully")
            except Exception as e:
                logger.warning(f"Failed to generate AI summary: {e}")
                ai_summary = None
        
        template_data['ai_summary'] = ai_summary
        
        # Render template
        template = self.env.get_template("report.html")
        html_content = template.render(**template_data)
        
        # Write output
        output_path.write_text(html_content)
        
        logger.info(f"Generated HTML report: {output_path}")
        
        return output_path
    
    def _prepare_template_data(
        self,
        result: EvaluationResult,
        include_details: bool,
        max_samples: int,
    ) -> Dict[str, Any]:
        """Prepare data for template rendering."""
        
        # Provider comparison data
        providers_data = []
        for provider_name, provider_result in result.provider_results.items():
            providers_data.append({
                'name': provider_name,
                'metrics': provider_result.aggregate_metrics,
                'document_count': len(provider_result.document_evaluations),
                'error_count': len(provider_result.errors),
                'total_time_ms': provider_result.total_processing_time_ms,
                'total_tokens': provider_result.total_tokens,
            })
        
        # Detailed evaluation results table
        detailed_results = self._get_detailed_results(result, max_samples)
        
        # Field-level analysis with RAGAS metrics
        field_analysis = self._analyze_fields_ragas(result)
        
        # Get RAGAS metrics used
        ragas_metrics = result.summary.get('ragas_metrics_used', [])
        
        return {
            'evaluation_id': result.evaluation_id,
            'evaluation_name': result.evaluation_name,
            'dataset_name': result.dataset_name,
            'timestamp': result.timestamp,
            'generated_at': datetime.utcnow(),
            'summary': result.summary,
            'providers': providers_data,
            'ragas_metrics': ragas_metrics,
            'field_analysis': field_analysis,
            'detailed_results': detailed_results,
            'config': result.config,
        }
    
    def _analyze_fields_ragas(self, result: EvaluationResult) -> List[Dict[str, Any]]:
        """Analyze performance by field across providers using RAGAS metrics."""
        field_data: Dict[str, Dict[str, Any]] = {}
        
        for provider_name, provider_result in result.provider_results.items():
            for doc_eval in provider_result.document_evaluations:
                for field_eval in doc_eval.field_evaluations:
                    field_name = field_eval.field_name
                    
                    if field_name not in field_data:
                        field_data[field_name] = {
                            'field_name': field_name,
                            'providers': {},
                        }
                    
                    if provider_name not in field_data[field_name]['providers']:
                        field_data[field_name]['providers'][provider_name] = {
                            'factual_correctness': [],
                            'semantic_similarity': [],
                            'answer_correctness': [],
                            'answer_relevancy': [],
                            'faithfulness': [],
                            'count': 0,
                        }
                    
                    stats = field_data[field_name]['providers'][provider_name]
                    stats['count'] += 1
                    
                    # Collect RAGAS scores
                    if field_eval.factual_correctness is not None:
                        stats['factual_correctness'].append(field_eval.factual_correctness)
                    if field_eval.semantic_similarity is not None:
                        stats['semantic_similarity'].append(field_eval.semantic_similarity)
                    if field_eval.answer_correctness is not None:
                        stats['answer_correctness'].append(field_eval.answer_correctness)
                    if field_eval.answer_relevancy is not None:
                        stats['answer_relevancy'].append(field_eval.answer_relevancy)
                    if field_eval.faithfulness is not None:
                        stats['faithfulness'].append(field_eval.faithfulness)
        
        # Calculate averages
        result_list = []
        for field_name, data in field_data.items():
            field_entry = {
                'field_name': field_name,
                'providers': {},
            }
            
            for provider_name, stats in data['providers'].items():
                provider_stats = {'count': stats['count']}
                
                for metric in ['factual_correctness', 'semantic_similarity', 'answer_correctness', 
                               'answer_relevancy', 'faithfulness']:
                    values = stats[metric]
                    if values:
                        provider_stats[f'avg_{metric}'] = sum(values) / len(values)
                    else:
                        provider_stats[f'avg_{metric}'] = None
                
                field_entry['providers'][provider_name] = provider_stats
            
            result_list.append(field_entry)
        
        return result_list
    
    def _get_detailed_results(
        self,
        result: EvaluationResult,
        max_samples: int
    ) -> List[Dict[str, Any]]:
        """Get detailed per-document, per-field results with all RAGAS metrics."""
        detailed = []
        
        for provider_name, provider_result in result.provider_results.items():
            for doc_eval in provider_result.document_evaluations[:max_samples]:
                for field_eval in doc_eval.field_evaluations:
                    detailed.append({
                        'document_id': doc_eval.document_id,
                        'provider': provider_name,
                        'field_name': field_eval.field_name,
                        'question': field_eval.question,
                        'expected_value': field_eval.expected_value,
                        'extracted_value': field_eval.extracted_value,
                        'context': field_eval.context[:200] if field_eval.context else '',
                        'factual_correctness': field_eval.factual_correctness,
                        'semantic_similarity': field_eval.semantic_similarity,
                        'answer_correctness': field_eval.answer_correctness,
                        'answer_relevancy': field_eval.answer_relevancy,
                        'faithfulness': field_eval.faithfulness,
                    })
        
        return detailed


# Default HTML template with RAGAS metrics
DEFAULT_HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ evaluation_name }} - OCR Evaluation Report</title>
    <style>
        :root {
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --bg-tertiary: #21262d;
            --text-primary: #f0f6fc;
            --text-secondary: #8b949e;
            --text-muted: #6e7681;
            --accent-green: #3fb950;
            --accent-red: #f85149;
            --accent-yellow: #d29922;
            --accent-blue: #58a6ff;
            --accent-purple: #a371f7;
            --accent-cyan: #39c5cf;
            --border-color: #30363d;
            --shadow: 0 8px 24px rgba(0,0,0,0.4);
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            min-height: 100vh;
        }
        
        .container {
            max-width: 1600px;
            margin: 0 auto;
            padding: 2rem;
        }
        
        header {
            background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-tertiary) 100%);
            border-bottom: 1px solid var(--border-color);
            padding: 2rem 0;
            margin-bottom: 2rem;
        }
        
        header .container {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 1rem;
        }
        
        h1 {
            font-size: 2rem;
            font-weight: 700;
            background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .meta {
            color: var(--text-secondary);
            font-size: 0.9rem;
        }
        
        .meta span {
            margin-right: 1.5rem;
        }
        
        .section {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: var(--shadow);
        }
        
        .section-title {
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .section-title::before {
            content: '';
            width: 4px;
            height: 1.25rem;
            background: var(--accent-blue);
            border-radius: 2px;
        }
        
        /* Summary Cards */
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1rem;
            margin-bottom: 1.5rem;
        }
        
        .summary-card {
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1.25rem;
            text-align: center;
        }
        
        .summary-card .value {
            font-size: 1.75rem;
            font-weight: 700;
            color: var(--accent-blue);
        }
        
        .summary-card .label {
            font-size: 0.8rem;
            color: var(--text-secondary);
            margin-top: 0.25rem;
        }
        
        .summary-card.success .value { color: var(--accent-green); }
        .summary-card.warning .value { color: var(--accent-yellow); }
        .summary-card.error .value { color: var(--accent-red); }
        .summary-card.info .value { color: var(--accent-cyan); }
        
        /* Tables */
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85rem;
        }
        
        th, td {
            padding: 0.75rem 0.5rem;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }
        
        th {
            background: var(--bg-tertiary);
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            font-size: 0.7rem;
            letter-spacing: 0.05em;
            position: sticky;
            top: 0;
        }
        
        tr:hover {
            background: var(--bg-tertiary);
        }
        
        /* Score colors */
        .score-high { color: var(--accent-green); font-weight: 600; }
        .score-medium { color: var(--accent-yellow); font-weight: 600; }
        .score-low { color: var(--accent-red); font-weight: 600; }
        .score-na { color: var(--text-muted); }
        
        /* Badges */
        .badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
        }
        
        .badge-success { background: rgba(63, 185, 80, 0.15); color: var(--accent-green); }
        .badge-warning { background: rgba(210, 153, 34, 0.15); color: var(--accent-yellow); }
        .badge-error { background: rgba(248, 81, 73, 0.15); color: var(--accent-red); }
        .badge-info { background: rgba(88, 166, 255, 0.15); color: var(--accent-blue); }
        
        /* Progress Bar */
        .score-bar {
            height: 6px;
            background: var(--bg-tertiary);
            border-radius: 3px;
            overflow: hidden;
            width: 60px;
            display: inline-block;
            vertical-align: middle;
            margin-right: 0.5rem;
        }
        
        .score-fill {
            height: 100%;
            border-radius: 3px;
            transition: width 0.3s ease;
        }
        
        .score-fill.high { background: var(--accent-green); }
        .score-fill.medium { background: var(--accent-yellow); }
        .score-fill.low { background: var(--accent-red); }
        
        /* Rankings */
        .ranking-item {
            display: flex;
            align-items: center;
            padding: 1rem;
            background: var(--bg-tertiary);
            border-radius: 8px;
            margin-bottom: 0.75rem;
        }
        
        .ranking-position {
            width: 2.5rem;
            height: 2.5rem;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 50%;
            font-weight: 700;
            margin-right: 1rem;
            flex-shrink: 0;
        }
        
        .ranking-position.gold { background: linear-gradient(135deg, #ffd700, #ffb700); color: #000; }
        .ranking-position.silver { background: linear-gradient(135deg, #c0c0c0, #a0a0a0); color: #000; }
        .ranking-position.bronze { background: linear-gradient(135deg, #cd7f32, #a0522d); color: #fff; }
        .ranking-position.other { background: var(--bg-secondary); color: var(--text-secondary); }
        
        .ranking-details {
            flex: 1;
        }
        
        .ranking-name {
            font-weight: 600;
            font-size: 1.1rem;
        }
        
        .ranking-metrics {
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
            margin-top: 0.5rem;
            color: var(--text-secondary);
            font-size: 0.8rem;
        }
        
        .ranking-metrics .metric-item {
            display: flex;
            align-items: center;
            gap: 0.25rem;
        }
        
        .ranking-score {
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--accent-green);
        }
        
        /* Detailed Results Table */
        .results-table-container {
            overflow-x: auto;
            max-height: 600px;
            overflow-y: auto;
        }
        
        .results-table {
            min-width: 1200px;
        }
        
        .results-table td {
            max-width: 200px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        
        .results-table td.expandable {
            cursor: pointer;
        }
        
        .results-table td.expandable:hover {
            white-space: normal;
            word-break: break-word;
        }
        
        /* Field Analysis */
        .field-card {
            background: var(--bg-tertiary);
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 0.75rem;
        }
        
        .field-name {
            font-weight: 600;
            margin-bottom: 0.75rem;
            font-size: 1rem;
        }
        
        .field-metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 0.5rem;
        }
        
        .field-provider-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.5rem;
            background: var(--bg-secondary);
            border-radius: 4px;
            font-size: 0.85rem;
        }
        
        .field-provider-name {
            font-weight: 500;
            color: var(--accent-blue);
        }
        
        .field-scores {
            display: flex;
            gap: 0.75rem;
        }
        
        .field-score-item {
            display: flex;
            align-items: center;
            gap: 0.25rem;
        }
        
        .field-score-label {
            color: var(--text-muted);
            font-size: 0.7rem;
        }
        
        /* RAGAS Metrics Legend */
        .metrics-legend {
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
            padding: 1rem;
            background: var(--bg-tertiary);
            border-radius: 8px;
            margin-bottom: 1rem;
            font-size: 0.85rem;
        }
        
        .legend-item {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .legend-abbr {
            font-weight: 600;
            color: var(--accent-cyan);
        }
        
        /* AI Summary */
        .ai-summary-section {
            background: linear-gradient(135deg, rgba(163, 113, 247, 0.1) 0%, rgba(88, 166, 255, 0.1) 100%);
            border: 1px solid var(--accent-purple);
        }
        
        .ai-summary-section .section-title::before {
            background: linear-gradient(135deg, var(--accent-purple), var(--accent-blue));
        }
        
        .ai-summary-content {
            font-size: 1.05rem;
            line-height: 1.8;
            color: var(--text-primary);
            padding: 1rem;
            background: var(--bg-tertiary);
            border-radius: 8px;
            border-left: 4px solid var(--accent-purple);
        }
        
        /* Footer */
        footer {
            text-align: center;
            padding: 2rem;
            color: var(--text-muted);
            font-size: 0.85rem;
        }
        
        /* Responsive */
        @media (max-width: 768px) {
            .container { padding: 1rem; }
            h1 { font-size: 1.5rem; }
            .summary-grid { grid-template-columns: repeat(2, 1fr); }
            .ranking-metrics { flex-direction: column; gap: 0.25rem; }
        }
    </style>
</head>
<body>
    <header>
        <div class="container">
            <div>
                <h1>🔍 {{ evaluation_name }}</h1>
                <div class="meta">
                    <span>🏷️ ID: {{ evaluation_id }}</span>
                    <span>📁 Dataset: {{ dataset_name }}</span>
                    <span>🕐 {{ timestamp.strftime('%Y-%m-%d %H:%M UTC') }}</span>
                </div>
            </div>
        </div>
    </header>
    
    <main class="container">
        <!-- AI Summary Section -->
        {% if ai_summary %}
        <div class="section ai-summary-section">
            <h2 class="section-title">🤖 AI Summary</h2>
            <div class="ai-summary-content">
                {{ ai_summary }}
            </div>
        </div>
        {% endif %}
        
        <!-- RAGAS Metrics Legend -->
        <div class="metrics-legend">
            <strong>RAGAS Metrics:</strong>
            <div class="legend-item"><span class="legend-abbr">FC</span> Factual Correctness</div>
            <div class="legend-item"><span class="legend-abbr">SS</span> Semantic Similarity</div>
            <div class="legend-item"><span class="legend-abbr">AC</span> Answer Correctness</div>
            <div class="legend-item"><span class="legend-abbr">AR</span> Answer Relevancy</div>
            <div class="legend-item"><span class="legend-abbr">F</span> Faithfulness</div>
            <div class="legend-item" style="margin-left: auto;">
                <span class="score-high">■</span> ≥0.8
                <span class="score-medium">■</span> 0.5-0.8
                <span class="score-low">■</span> &lt;0.5
            </div>
        </div>
        
        <!-- Summary Section -->
        <div class="section">
            <h2 class="section-title">Executive Summary</h2>
            <div class="summary-grid">
                <div class="summary-card">
                    <div class="value">{{ summary.total_documents }}</div>
                    <div class="label">Documents</div>
                </div>
                <div class="summary-card">
                    <div class="value">{{ summary.total_providers }}</div>
                    <div class="label">Providers</div>
                </div>
                <div class="summary-card success">
                    <div class="value">{{ summary.best_provider or 'N/A' }}</div>
                    <div class="label">Best Provider</div>
                </div>
                {% if summary.provider_rankings and summary.provider_rankings[0].avg_factual_correctness is not none %}
                <div class="summary-card info">
                    <div class="value">{{ summary.provider_rankings[0].avg_factual_correctness|format_percent }}</div>
                    <div class="label">Best Factual Correctness</div>
                </div>
                {% endif %}
                {% if summary.provider_rankings and summary.provider_rankings[0].avg_semantic_similarity is not none %}
                <div class="summary-card info">
                    <div class="value">{{ summary.provider_rankings[0].avg_semantic_similarity|format_percent }}</div>
                    <div class="label">Best Semantic Similarity</div>
                </div>
                {% endif %}
                {% if summary.provider_rankings and summary.provider_rankings[0].avg_answer_correctness is not none %}
                <div class="summary-card info">
                    <div class="value">{{ summary.provider_rankings[0].avg_answer_correctness|format_percent }}</div>
                    <div class="label">Best Answer Correctness</div>
                </div>
                {% endif %}
            </div>
        </div>
        
        <!-- Provider Rankings -->
        <div class="section">
            <h2 class="section-title">Provider Rankings (by RAGAS Scores)</h2>
            {% for ranking in summary.provider_rankings %}
            <div class="ranking-item">
                <div class="ranking-position {% if loop.index == 1 %}gold{% elif loop.index == 2 %}silver{% elif loop.index == 3 %}bronze{% else %}other{% endif %}">
                    {{ loop.index }}
                </div>
                <div class="ranking-details">
                    <div class="ranking-name">{{ ranking.provider }}</div>
                    <div class="ranking-metrics">
                        {% if ranking.avg_factual_correctness is not none %}
                        <span class="metric-item">📊 FC: {{ ranking.avg_factual_correctness|format_percent }}</span>
                        {% endif %}
                        {% if ranking.avg_semantic_similarity is not none %}
                        <span class="metric-item">🎯 SS: {{ ranking.avg_semantic_similarity|format_percent }}</span>
                        {% endif %}
                        {% if ranking.avg_answer_correctness is not none %}
                        <span class="metric-item">✅ AC: {{ ranking.avg_answer_correctness|format_percent }}</span>
                        {% endif %}
                        {% if ranking.avg_answer_relevancy is not none %}
                        <span class="metric-item">🔗 AR: {{ ranking.avg_answer_relevancy|format_percent }}</span>
                        {% endif %}
                        {% if ranking.avg_faithfulness is not none %}
                        <span class="metric-item">📝 F: {{ ranking.avg_faithfulness|format_percent }}</span>
                        {% endif %}
                        <span class="metric-item">⏱️ {{ ranking.avg_processing_time_ms|format_ms }}</span>
                        <span class="metric-item">📄 {{ ranking.document_count }} docs</span>
                    </div>
                </div>
                <div class="ranking-score">{{ ranking.primary_score|format_percent }}</div>
            </div>
            {% endfor %}
        </div>
        
        <!-- Detailed Evaluation Results -->
        <div class="section">
            <h2 class="section-title">Detailed Evaluation Results (RAGAS Metrics per Field)</h2>
            <div class="results-table-container">
                <table class="results-table">
                    <thead>
                        <tr>
                            <th>Document</th>
                            <th>Provider</th>
                            <th>Field</th>
                            <th>Question</th>
                            <th>Ground Truth</th>
                            <th>Extracted</th>
                            <th>FC</th>
                            <th>SS</th>
                            <th>AC</th>
                            <th>AR</th>
                            <th>F</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for row in detailed_results %}
                        <tr>
                            <td>{{ row.document_id }}</td>
                            <td><span class="badge badge-info">{{ row.provider }}</span></td>
                            <td><strong>{{ row.field_name }}</strong></td>
                            <td class="expandable" title="{{ row.question }}">{{ row.question|truncate_text(40) }}</td>
                            <td class="expandable" title="{{ row.expected_value }}">{{ row.expected_value|truncate_text(30) }}</td>
                            <td class="expandable" title="{{ row.extracted_value }}">{{ row.extracted_value|truncate_text(30) }}</td>
                            <td class="{% if row.factual_correctness is none %}score-na{% elif row.factual_correctness >= 0.8 %}score-high{% elif row.factual_correctness >= 0.5 %}score-medium{% else %}score-low{% endif %}">
                                {{ row.factual_correctness|format_score }}
                            </td>
                            <td class="{% if row.semantic_similarity is none %}score-na{% elif row.semantic_similarity >= 0.8 %}score-high{% elif row.semantic_similarity >= 0.5 %}score-medium{% else %}score-low{% endif %}">
                                {{ row.semantic_similarity|format_score }}
                            </td>
                            <td class="{% if row.answer_correctness is none %}score-na{% elif row.answer_correctness >= 0.8 %}score-high{% elif row.answer_correctness >= 0.5 %}score-medium{% else %}score-low{% endif %}">
                                {{ row.answer_correctness|format_score }}
                            </td>
                            <td class="{% if row.answer_relevancy is none %}score-na{% elif row.answer_relevancy >= 0.8 %}score-high{% elif row.answer_relevancy >= 0.5 %}score-medium{% else %}score-low{% endif %}">
                                {{ row.answer_relevancy|format_score }}
                            </td>
                            <td class="{% if row.faithfulness is none %}score-na{% elif row.faithfulness >= 0.8 %}score-high{% elif row.faithfulness >= 0.5 %}score-medium{% else %}score-low{% endif %}">
                                {{ row.faithfulness|format_score }}
                            </td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>
        
        <!-- Field-Level Analysis -->
        <div class="section">
            <h2 class="section-title">Field-Level Analysis (Average RAGAS Scores)</h2>
            {% for field in field_analysis %}
            <div class="field-card">
                <div class="field-name">📝 {{ field.field_name }}</div>
                <div class="field-metrics">
                    {% for provider_name, stats in field.providers.items() %}
                    <div class="field-provider-row">
                        <span class="field-provider-name">{{ provider_name }}</span>
                        <div class="field-scores">
                            {% if stats.avg_factual_correctness is not none %}
                            <span class="field-score-item">
                                <span class="field-score-label">FC:</span>
                                <span class="{% if stats.avg_factual_correctness >= 0.8 %}score-high{% elif stats.avg_factual_correctness >= 0.5 %}score-medium{% else %}score-low{% endif %}">
                                    {{ stats.avg_factual_correctness|format_percent }}
                                </span>
                            </span>
                            {% endif %}
                            {% if stats.avg_semantic_similarity is not none %}
                            <span class="field-score-item">
                                <span class="field-score-label">SS:</span>
                                <span class="{% if stats.avg_semantic_similarity >= 0.8 %}score-high{% elif stats.avg_semantic_similarity >= 0.5 %}score-medium{% else %}score-low{% endif %}">
                                    {{ stats.avg_semantic_similarity|format_percent }}
                                </span>
                            </span>
                            {% endif %}
                            {% if stats.avg_answer_correctness is not none %}
                            <span class="field-score-item">
                                <span class="field-score-label">AC:</span>
                                <span class="{% if stats.avg_answer_correctness >= 0.8 %}score-high{% elif stats.avg_answer_correctness >= 0.5 %}score-medium{% else %}score-low{% endif %}">
                                    {{ stats.avg_answer_correctness|format_percent }}
                                </span>
                            </span>
                            {% endif %}
                        </div>
                    </div>
                    {% endfor %}
                </div>
            </div>
            {% endfor %}
        </div>
        
        <!-- Provider Comparison Table -->
        <div class="section">
            <h2 class="section-title">Provider Comparison Summary</h2>
            <table>
                <thead>
                    <tr>
                        <th>Provider</th>
                        <th>Documents</th>
                        <th>Avg Factual Correctness</th>
                        <th>Avg Semantic Similarity</th>
                        <th>Avg Answer Correctness</th>
                        <th>OCR Success Rate</th>
                        <th>Total Time</th>
                        <th>Errors</th>
                    </tr>
                </thead>
                <tbody>
                    {% for provider in providers %}
                    <tr>
                        <td><strong>{{ provider.name }}</strong></td>
                        <td>{{ provider.document_count }}</td>
                        <td class="{% if provider.metrics.avg_factual_correctness is not defined or provider.metrics.avg_factual_correctness is none %}score-na{% elif provider.metrics.avg_factual_correctness >= 0.8 %}score-high{% elif provider.metrics.avg_factual_correctness >= 0.5 %}score-medium{% else %}score-low{% endif %}">
                            {{ provider.metrics.avg_factual_correctness|format_percent if provider.metrics.avg_factual_correctness is defined else 'N/A' }}
                        </td>
                        <td class="{% if provider.metrics.avg_semantic_similarity is not defined or provider.metrics.avg_semantic_similarity is none %}score-na{% elif provider.metrics.avg_semantic_similarity >= 0.8 %}score-high{% elif provider.metrics.avg_semantic_similarity >= 0.5 %}score-medium{% else %}score-low{% endif %}">
                            {{ provider.metrics.avg_semantic_similarity|format_percent if provider.metrics.avg_semantic_similarity is defined else 'N/A' }}
                        </td>
                        <td class="{% if provider.metrics.avg_answer_correctness is not defined or provider.metrics.avg_answer_correctness is none %}score-na{% elif provider.metrics.avg_answer_correctness >= 0.8 %}score-high{% elif provider.metrics.avg_answer_correctness >= 0.5 %}score-medium{% else %}score-low{% endif %}">
                            {{ provider.metrics.avg_answer_correctness|format_percent if provider.metrics.avg_answer_correctness is defined else 'N/A' }}
                        </td>
                        <td>
                            <span class="badge {% if provider.metrics.ocr_success_rate is defined and provider.metrics.ocr_success_rate >= 0.9 %}badge-success{% elif provider.metrics.ocr_success_rate is defined and provider.metrics.ocr_success_rate >= 0.7 %}badge-warning{% else %}badge-error{% endif %}">
                                {{ provider.metrics.ocr_success_rate|format_percent if provider.metrics.ocr_success_rate is defined else 'N/A' }}
                            </span>
                        </td>
                        <td>{{ provider.total_time_ms|format_ms }}</td>
                        <td>
                            {% if provider.error_count > 0 %}
                            <span class="badge badge-error">{{ provider.error_count }}</span>
                            {% else %}
                            <span class="badge badge-success">0</span>
                            {% endif %}
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
    </main>
    
    <footer>
        <p>Generated by OCR Evaluator using RAGAS | {{ generated_at.strftime('%Y-%m-%d %H:%M:%S UTC') }}</p>
        <p style="margin-top: 0.5rem; font-size: 0.75rem;">
            Metrics: Factual Correctness (FC), Semantic Similarity (SS), Answer Correctness (AC), Answer Relevancy (AR), Faithfulness (F)
        </p>
    </footer>
</body>
</html>
'''
