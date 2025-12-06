# Limitations

## RAGAS Library
- `LLM returned 1 generations instead of requested 3` warning - RAGAS requests multiple generations but OpenAI returns 1. Results still valid.
- Evaluation is slow (~5s per field) due to multiple LLM calls per metric
- Requires OpenAI API for both OCR extraction and RAGAS evaluation
- **Cost estimate**: ~$0.10-0.20 per document with 10 fields (5 metrics × 10 fields × ~2-3 LLM calls each)

## Document Support
- PDF only (no image files like PNG/JPG)
- Single provider implemented (OpenAI Vision)

## Metrics
- No cost tracking implemented yet
- No latency benchmarking separate from evaluation time

## Scalability
- Large datasets may hit API rate limits
- No database storage (JSON files only)
- No incremental/resumable evaluation runs

## Ground Truth
- Manual ground truth creation required
- No annotation tool provided
- Complex nested fields (e.g., medications list) may have lower accuracy scores

