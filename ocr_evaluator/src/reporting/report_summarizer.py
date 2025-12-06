"""
AI-powered report summarizer.
Generates a 2-4 line executive summary of the evaluation report using OpenAI.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional

from openai import OpenAI

from ..utils.logging_config import get_logger, PerformanceLogger

logger = get_logger(__name__)


class ReportSummarizer:
    """
    Generates AI summaries for OCR evaluation reports.
    """
    
    def __init__(self, model: str = "gpt-4o-mini"):
        """
        Initialize the summarizer.
        
        Args:
            model: OpenAI model to use for summarization
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        logger.info(f"Report summarizer initialized with model: {model}")
    
    @PerformanceLogger(get_logger(__name__), "Generating AI summary")
    def generate_summary(self, report_data: Dict[str, Any]) -> str:
        """
        Generate a 2-4 line AI summary of the evaluation report.
        
        Args:
            report_data: Dictionary containing evaluation results
            
        Returns:
            AI-generated summary string
        """
        # Extract key metrics for the prompt
        summary_info = self._extract_summary_info(report_data)
        
        prompt = f"""You are an OCR evaluation analyst. Generate a concise 2-4 line executive summary for this OCR evaluation report.

EVALUATION DATA:
- Evaluation Name: {summary_info['evaluation_name']}
- Documents Evaluated: {summary_info['total_documents']}
- Best Provider: {summary_info['best_provider']}
- Factual Correctness: {summary_info['factual_correctness']}
- Semantic Similarity: {summary_info['semantic_similarity']}
- Answer Correctness: {summary_info['answer_correctness']}
- Faithfulness: {summary_info['faithfulness']}
- Fields with Lowest Scores: {summary_info['weak_fields']}
- Fields with Highest Scores: {summary_info['strong_fields']}

INSTRUCTIONS:
1. Start with overall assessment (good/acceptable/needs improvement)
2. Highlight the key strength or weakness
3. Mention any concerning metrics (below 70%)
4. End with a brief recommendation if needed

Keep it to 2-4 lines maximum. Be direct and actionable."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a concise technical report analyst. Always respond in 2-4 lines."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            summary = response.choices[0].message.content.strip()
            logger.info(f"Generated AI summary: {summary[:100]}...")
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate AI summary: {e}")
            return self._generate_fallback_summary(summary_info)
    
    def _extract_summary_info(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key information from report data for summarization."""
        summary = report_data.get('summary', {})
        provider_rankings = summary.get('provider_rankings', [])
        
        # Get best provider metrics
        best_metrics = provider_rankings[0] if provider_rankings else {}
        
        # Get field analysis to find weak/strong fields
        field_analysis = report_data.get('field_analysis', [])
        weak_fields = []
        strong_fields = []
        
        for field in field_analysis:
            for provider_name, stats in field.get('providers', {}).items():
                avg_fc = stats.get('avg_factual_correctness')
                if avg_fc is not None:
                    if avg_fc < 0.7:
                        weak_fields.append(f"{field['field_name']} ({avg_fc:.0%})")
                    elif avg_fc >= 0.9:
                        strong_fields.append(f"{field['field_name']} ({avg_fc:.0%})")
        
        return {
            'evaluation_name': report_data.get('evaluation_name', 'Unknown'),
            'total_documents': summary.get('total_documents', 0),
            'best_provider': summary.get('best_provider', 'N/A'),
            'factual_correctness': self._format_metric(best_metrics.get('avg_factual_correctness')),
            'semantic_similarity': self._format_metric(best_metrics.get('avg_semantic_similarity')),
            'answer_correctness': self._format_metric(best_metrics.get('avg_answer_correctness')),
            'faithfulness': self._format_metric(best_metrics.get('avg_faithfulness')),
            'weak_fields': ', '.join(weak_fields[:3]) if weak_fields else 'None',
            'strong_fields': ', '.join(strong_fields[:3]) if strong_fields else 'None',
        }
    
    @staticmethod
    def _format_metric(value: Optional[float]) -> str:
        """Format metric value as percentage string."""
        if value is None:
            return 'N/A'
        return f"{value * 100:.1f}%"
    
    def _generate_fallback_summary(self, summary_info: Dict[str, Any]) -> str:
        """Generate a simple fallback summary if AI fails."""
        fc = summary_info['factual_correctness']
        ss = summary_info['semantic_similarity']
        
        return (
            f"Evaluation '{summary_info['evaluation_name']}' completed on {summary_info['total_documents']} documents. "
            f"Best provider: {summary_info['best_provider']} with Factual Correctness: {fc}, Semantic Similarity: {ss}. "
            f"{'Review weak fields: ' + summary_info['weak_fields'] if summary_info['weak_fields'] != 'None' else 'All fields performing adequately.'}"
        )


def summarize_report_file(json_path: str, model: str = "gpt-4o-mini") -> str:
    """
    Convenience function to summarize a report from a JSON file.
    
    Args:
        json_path: Path to the evaluation results JSON file
        model: OpenAI model to use
        
    Returns:
        AI-generated summary string
    """
    json_path = Path(json_path)
    
    if not json_path.exists():
        raise FileNotFoundError(f"Report file not found: {json_path}")
    
    with open(json_path, 'r') as f:
        report_data = json.load(f)
    
    summarizer = ReportSummarizer(model=model)
    return summarizer.generate_summary(report_data)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m src.reporting.report_summarizer <path_to_results.json>")
        sys.exit(1)
    
    json_path = sys.argv[1]
    
    try:
        summary = summarize_report_file(json_path)
        print("\n" + "="*60)
        print("AI SUMMARY")
        print("="*60)
        print(summary)
        print("="*60 + "\n")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

