# Understanding the OCR Evaluation Report

## What This Tool Does

This tool evaluates how well an OCR system extracts fields from documents (like prescriptions) by comparing extracted values against ground truth (expected values).

---

## RAGAS Metrics Explained

### 1. Factual Correctness (FC)
**What it measures**: Are the facts in the extracted text correct?

| Score | Meaning | OCR Interpretation |
|-------|---------|-------------------|
| 1.0 | All facts match | OCR extracted correct information |
| 0.5 | Some facts wrong | OCR got partial information or made errors |
| 0.0 | Completely wrong | OCR extracted wrong data |

**Example**:
- Ground truth: `"Dr. John Smith"`
- Extracted: `"Dr. John Smith"` → FC = 1.0 ✅
- Extracted: `"Dr. John Smyth"` → FC = 0.7 (name misspelled)
- Extracted: `"Dr. Jane Doe"` → FC = 0.0 ❌

---

### 2. Semantic Similarity (SS)
**What it measures**: Do the extracted and expected values mean the same thing?

| Score | Meaning | OCR Interpretation |
|-------|---------|-------------------|
| 1.0 | Identical meaning | Perfect extraction |
| 0.7+ | Very similar meaning | Minor wording differences, acceptable |
| 0.5 | Somewhat similar | Partial match, needs review |
| <0.5 | Different meaning | OCR likely failed |

**Example**:
- Ground truth: `"Take twice daily"`
- Extracted: `"Take 2 times per day"` → SS = 0.95 ✅ (same meaning)
- Extracted: `"Take once daily"` → SS = 0.4 ❌ (different meaning)

---

### 3. Answer Correctness (AC)
**What it measures**: Overall correctness combining factual accuracy and semantic match.

| Score | Meaning | OCR Interpretation |
|-------|---------|-------------------|
| 0.8+ | Excellent | OCR is reliable for this field |
| 0.6-0.8 | Good | Minor issues, generally usable |
| 0.4-0.6 | Fair | Needs manual verification |
| <0.4 | Poor | OCR unreliable for this field |

**This is your primary metric** - it gives the overall quality score.

---

### 4. Answer Relevancy (AR)
**What it measures**: Is the extracted value relevant to what was asked?

| Score | Meaning | OCR Interpretation |
|-------|---------|-------------------|
| 1.0 | Highly relevant | Extracted the right type of information |
| 0.5 | Partially relevant | Extracted related but not exact info |
| 0.0 | Not relevant | Extracted wrong field entirely |

**Example** (extracting "patient_name"):
- Extracted: `"John Doe"` → AR = 1.0 ✅ (it's a name)
- Extracted: `"123 Main St"` → AR = 0.1 ❌ (that's an address, not a name)

---

### 5. Faithfulness
**What it measures**: Is the extracted value supported by the document content?

| Score | Meaning | OCR Interpretation |
|-------|---------|-------------------|
| 1.0 | Fully supported | Extraction came from the document |
| 0.5 | Partially supported | Some info may be inferred/hallucinated |
| 0.0 | Not supported | OCR hallucinated information |

**Important for detecting hallucinations** - when OCR invents data not in the document.

---

## Quick Reference: What Good Scores Look Like

| Metric | Target | Acceptable | Concerning |
|--------|--------|------------|------------|
| Factual Correctness | >0.9 | 0.7-0.9 | <0.7 |
| Semantic Similarity | >0.9 | 0.8-0.9 | <0.8 |
| Answer Correctness | >0.8 | 0.6-0.8 | <0.6 |
| Answer Relevancy | >0.8 | 0.6-0.8 | <0.6 |
| Faithfulness | 1.0 | 0.8-1.0 | <0.8 |

---

## Reading the Report

### Summary Section
- **Aggregate scores**: Average of all fields across all documents
- Use this to compare different OCR providers

### Field Analysis
- Shows which fields perform well/poorly
- Low scores on specific fields = OCR struggles with that field type

### Detailed Results Table
- Per-document, per-field breakdown
- **Question**: The field being extracted
- **Context**: The document text (truncated)
- **Response**: What OCR extracted
- **Reference**: Ground truth (expected value)
- **Scores**: Individual metric scores

---

## Common Patterns

| Pattern | Likely Cause |
|---------|--------------|
| High SS, Low FC | Meaning correct but details wrong (typos, formatting) |
| Low SS, High FC | Facts correct but phrased differently |
| Low AR | OCR extracting wrong field type |
| Low Faithfulness | OCR hallucinating data not in document |
| All metrics low | OCR failed to read document properly |

---

## For Our OCR Use Case

**Priority order of metrics**:
1. **Factual Correctness** - Most critical. Wrong facts = wrong data in system
2. **Answer Correctness** - Overall reliability indicator
3. **Faithfulness** - Detect hallucinations (critical for medical docs)
4. **Semantic Similarity** - Catch meaning drift
5. **Answer Relevancy** - Ensure right fields extracted

**Red flags to watch**:
- Faithfulness < 0.8 → OCR may be making up data
- Factual Correctness < 0.7 → Data quality issues
- Large variance between fields → Some fields need schema adjustment

