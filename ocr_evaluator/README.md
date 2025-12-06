# OCR Evaluation Framework

A modular, extensible framework for evaluating OCR services using **RAGAS metrics**. Built to benchmark multiple OCR providers (OpenAI Vision, AWS Textract, Google Document AI, etc.) against ground truth data.

## 🎯 Features

- **RAGAS-Only Evaluation**: All metrics computed using RAGAS library (no simple string matching)
- **Multi-Provider Support**: Easily add and compare different OCR engines
- **Comprehensive Metrics**: Factual Correctness, Semantic Similarity, Answer Correctness, Answer Relevancy, Faithfulness
- **Multiprocessing**: Parallel document processing with configurable workers
- **Rate Limiting**: Built-in rate limiter with token bucket and sliding window strategies
- **Comprehensive Logging**: Detailed logs with performance metrics and API call tracking
- **Beautiful HTML Reports**: Detailed reports with per-field RAGAS scores
- **Configurable Schemas**: YAML-based field extraction schemas

---

## 📊 Evaluation Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         OCR EVALUATION FLOW                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 1. INPUT                                                             │   │
│  │    ┌──────────────────┐     ┌──────────────────┐                    │   │
│  │    │   PDF Documents   │     │  Ground Truth    │                    │   │
│  │    │   (documents/)    │     │  JSON Files      │                    │   │
│  │    │                   │     │  (ground_truth/) │                    │   │
│  │    │  prescription.pdf │     │  prescription.json                   │   │
│  │    └────────┬──────────┘     └────────┬─────────┘                    │   │
│  │             │                         │                              │   │
│  │             └───────────┬─────────────┘                              │   │
│  └─────────────────────────┼───────────────────────────────────────────┘   │
│                            ▼                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 2. OCR EXTRACTION (OpenAI Vision / Other Providers)                  │   │
│  │                                                                      │   │
│  │    PDF ──► Images ──► GPT-4o Vision ──► Extracted Fields (JSON)     │   │
│  │                                                                      │   │
│  │    Schema defines what fields to extract:                            │   │
│  │    • patient_name, doctor_name, diagnosis, medications, etc.        │   │
│  │                                                                      │   │
│  │    Output: { "patient_name": "Rajesh Kumar", "diagnosis": "..." }   │   │
│  └─────────────────────────┬───────────────────────────────────────────┘   │
│                            ▼                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 3. RAGAS EVALUATION (Per Field)                                      │   │
│  │                                                                      │   │
│  │    For each extracted field, RAGAS evaluates:                        │   │
│  │                                                                      │   │
│  │    ┌─────────────────────────────────────────────────────────────┐  │   │
│  │    │  user_input:        "What is the patient name?"             │  │   │
│  │    │  response:          "Rajesh Kumar" (extracted)              │  │   │
│  │    │  reference:         "Rajesh Kumar" (ground truth)           │  │   │
│  │    │  retrieved_contexts: ["Full document text..."]              │  │   │
│  │    └─────────────────────────────────────────────────────────────┘  │   │
│  │                                                                      │   │
│  │    RAGAS Metrics Computed (0.0 to 1.0 scale):                       │   │
│  │    ┌────────────────────────┬───────────────────────────────────┐  │   │
│  │    │ Factual Correctness    │ Is extracted value factually      │  │   │
│  │    │ (FC)                   │ correct vs ground truth?          │  │   │
│  │    ├────────────────────────┼───────────────────────────────────┤  │   │
│  │    │ Semantic Similarity    │ How semantically similar is       │  │   │
│  │    │ (SS)                   │ extraction to ground truth?       │  │   │
│  │    ├────────────────────────┼───────────────────────────────────┤  │   │
│  │    │ Answer Correctness     │ Combined correctness score        │  │   │
│  │    │ (AC)                   │ (factual + semantic)              │  │   │
│  │    ├────────────────────────┼───────────────────────────────────┤  │   │
│  │    │ Answer Relevancy       │ Is extraction relevant to         │  │   │
│  │    │ (AR)                   │ the question asked?               │  │   │
│  │    ├────────────────────────┼───────────────────────────────────┤  │   │
│  │    │ Faithfulness           │ Does extraction stay faithful     │  │   │
│  │    │ (F)                    │ to source document context?       │  │   │
│  │    └────────────────────────┴───────────────────────────────────┘  │   │
│  └─────────────────────────┬───────────────────────────────────────────┘   │
│                            ▼                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 4. REPORT GENERATION                                                 │   │
│  │                                                                      │   │
│  │    HTML Report includes:                                             │   │
│  │    • Executive Summary (best provider, aggregate scores)            │   │
│  │    • Provider Rankings (sorted by RAGAS scores)                     │   │
│  │    • Detailed Results Table:                                         │   │
│  │      ┌──────────┬─────────┬───────────┬──────┬──────┬──────┐        │   │
│  │      │ Document │ Field   │ Question  │  FC  │  SS  │  AC  │        │   │
│  │      ├──────────┼─────────┼───────────┼──────┼──────┼──────┤        │   │
│  │      │ presc_01 │ patient │ What is...│ 0.95 │ 0.98 │ 0.96 │        │   │
│  │      │ presc_01 │ doctor  │ What is...│ 0.88 │ 0.92 │ 0.90 │        │   │
│  │      └──────────┴─────────┴───────────┴──────┴──────┴──────┘        │   │
│  │    • Field-Level Analysis (per-field averages)                      │   │
│  │    • Provider Comparison Summary                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
ocr_evaluator/
├── config/
│   ├── providers.yaml          # Provider configurations
│   ├── evaluation.yaml         # Evaluation settings (metrics, multiprocessing, logging)
│   └── schemas/                # Field extraction schemas
│       ├── prescription.yaml
│       └── medical_claim.yaml
├── src/
│   ├── providers/              # OCR provider implementations
│   │   ├── base.py             # Abstract base class
│   │   ├── openai_provider.py  # OpenAI Vision provider
│   │   └── registry.py         # Provider registry
│   ├── dataset/                # Dataset management
│   │   ├── loader.py           # Load documents + ground truth
│   │   └── schema.py           # Schema management
│   ├── evaluation/             # Evaluation engine
│   │   ├── engine.py           # Main orchestrator with multiprocessing
│   │   └── ragas_evaluator.py  # RAGAS library integration (v0.3.9+)
│   ├── reporting/              # Report generation
│   │   └── generator.py        # HTML report generator
│   └── utils/                  # Utilities
│       ├── logging_config.py   # Comprehensive logging
│       ├── pdf.py              # PDF to image conversion
│       └── rate_limiter.py     # Rate limiting
├── datasets/
│   └── sample/                 # Sample dataset structure
│       ├── documents/          # PDF files go here
│       ├── ground_truth/       # JSON ground truth files
│       └── metadata.json       # Dataset metadata
├── reports/                    # Generated evaluation reports
├── logs/                       # Log files (organized by evaluation name)
├── evaluate.py                 # CLI entry point
├── requirements.txt            # Python dependencies
└── README.md
```

---

## 🚀 Quick Start

### 1. Installation

```bash
cd ocr_evaluator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

### 3. Prepare Your Dataset

```
datasets/my_dataset/
├── documents/
│   ├── doc_001.pdf
│   └── doc_002.pdf
├── ground_truth/
│   ├── doc_001.json      # Must match PDF filename
│   └── doc_002.json
└── metadata.json
```

**Ground Truth JSON Format:**

```json
{
  "document_id": "doc_001",
  "schema": "prescription",
  "fields": {
    "patient_name": "Rajesh Kumar",
    "doctor_name": "Dr. Priya Sharma",
    "diagnosis": "Type 2 Diabetes",
    "medications": [
      {
        "medicine_name": "Metformin",
        "dosage": "500mg",
        "frequency": "Twice daily"
      }
    ],
    "prescription_date": "2024-11-15"
  }
}
```

### 4. Run Evaluation

```bash
python evaluate.py \
  --name "prescription_test_v1" \
  --providers openai \
  --dataset datasets/sample \
  --schema prescription \
  --output reports/
```

### 5. View Report

```bash
open reports/prescription_test_v1/prescription_test_v1_*_report.html
```

---

## 📊 RAGAS Metrics

All evaluation is performed using RAGAS library metrics (no simple string matching):

| Metric | Abbreviation | Description | Range |
|--------|--------------|-------------|-------|
| **Factual Correctness** | FC | Is the extracted value factually correct compared to ground truth? | 0.0 - 1.0 |
| **Semantic Similarity** | SS | How semantically similar is the extraction to ground truth? | 0.0 - 1.0 |
| **Answer Correctness** | AC | Combined correctness score (factual + semantic) | 0.0 - 1.0 |
| **Answer Relevancy** | AR | Is the extraction relevant to the question asked? | 0.0 - 1.0 |
| **Faithfulness** | F | Does extraction stay faithful to the source document? | 0.0 - 1.0 |

**Score Interpretation:**
- 🟢 **≥ 0.8**: High quality extraction
- 🟡 **0.5 - 0.8**: Acceptable, may need review
- 🔴 **< 0.5**: Poor extraction, needs attention

---

## ⚙️ Configuration

### Evaluation Settings (`config/evaluation.yaml`)

```yaml
# RAGAS Metrics to use
metrics:
  enabled:
    - factual_correctness
    - semantic_similarity
    - answer_correctness
    - answer_relevancy
    - faithfulness
  llm:
    model: "gpt-4o-mini"  # Model for RAGAS evaluation
    temperature: 0.0

# Multiprocessing
multiprocessing:
  enabled: true
  max_workers: 4
  resources:
    max_memory_mb: 2048
  batch_size: 10
  timeout_seconds: 300

# Rate Limiting
rate_limiter:
  enabled: true
  strategy: "token_bucket"
  requests_per_minute: 60
  tokens_per_minute: 150000

# Logging
logging:
  level: "INFO"
  format: "detailed"
```

### Provider Settings (`config/providers.yaml`)

```yaml
providers:
  openai:
    enabled: true
    model: "gpt-4o"           # Model for OCR extraction
    max_tokens: 4096
    temperature: 0.0
    rate_limit:
      requests_per_minute: 60
      tokens_per_minute: 150000
```

---

## 🔧 CLI Options

```bash
python evaluate.py --help

Required:
  --name NAME           Evaluation name (used for logs/reports prefix)
  --providers PROVIDERS Comma-separated providers (e.g., openai)
  --dataset PATH        Path to dataset directory
  --schema TEXT         Schema name for extraction

Optional:
  --output PATH         Output directory (default: reports)
  --config PATH         Evaluation config file
  --workers INT         Number of parallel workers
  --no-multiprocessing  Run sequentially
  --log-level           DEBUG, INFO, WARNING, ERROR
  --description TEXT    Description for this run
```

**Example:**

```bash
python evaluate.py \
  --name "prod_evaluation_v2" \
  --description "Testing with updated prescription schema" \
  --providers openai \
  --dataset datasets/medical_claims \
  --schema medical_claim \
  --output reports/ \
  --workers 8 \
  --log-level DEBUG
```

---

## 📁 Output Structure

With `--name "prescription_test_v1"`:

```
ocr_evaluator/
├── logs/
│   └── prescription_test_v1/
│       └── prescription_test_v1_20251129_193000.log
│
├── reports/
│   └── prescription_test_v1/
│       ├── prescription_test_v1_20251129_193000_report.html
│       └── prescription_test_v1_20251129_193000_results.json
```

---

## 🔌 Adding New Providers

1. Create provider class:

```python
# src/providers/textract_provider.py
from .base import OCRProvider, OCRResult
from .registry import ProviderRegistry

@ProviderRegistry.register("textract")
class TextractProvider(OCRProvider):
    def extract(self, document_path, schema, document_id=None):
        # Your extraction logic
        return OCRResult(
            document_id=document_id,
            provider_name=self.name,
            fields=extracted_fields,
            raw_text=full_text,  # Important for RAGAS context
            success=True,
        )
    
    def extract_text(self, document_path):
        return extracted_text
```

2. Add to `config/providers.yaml`:

```yaml
providers:
  textract:
    enabled: true
    region: "us-east-1"
```

3. Import in `src/providers/__init__.py`:

```python
from .textract_provider import TextractProvider
```

---

## 📈 Report Sections

The HTML report includes:

1. **RAGAS Metrics Legend** - Quick reference for metric abbreviations
2. **Executive Summary** - Best provider, aggregate scores
3. **Provider Rankings** - Sorted by primary RAGAS score
4. **Detailed Results Table** - Per-document, per-field RAGAS scores
5. **Field-Level Analysis** - Average scores per field across providers
6. **Provider Comparison** - Side-by-side metrics comparison

---

## 🛠️ Development

### Running Tests

```bash
pip install pytest pytest-cov
pytest tests/ -v
```

### Syntax Check

```bash
python -m py_compile evaluate.py src/**/*.py
```

---

## 📝 License

MIT License

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes
4. Submit a pull request
