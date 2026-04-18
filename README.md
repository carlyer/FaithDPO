# FaithDPO

**Faithful Direct Preference Optimization for RAG Hallucination Reduction**

FaithDPO is a research project that reduces hallucinations in RAG systems by prioritizing content extraction from context over generation. By constructing high-quality DPO training data through two extraction strategies and applying preference optimization, the model learns to generate responses that are more faithful to the retrieved context.

## Project Overview

RAG systems often suffer from hallucinations where the language model generates content not present in the retrieved context. FaithDPO addresses this through:

1. **Two extraction-based sampling strategies** that prioritize copying from context over free-form generation
2. **A two-stage data filtering pipeline** that constructs high-quality DPO training pairs
3. **Direct Preference Optimization (DPO)** to fine-tune the model on extracted-preferred vs generated-rejected pairs

## Architecture

### Two Sampling Strategies

#### 1. SentenceExtract (Sentence-Level Exact Extraction)

The strictest strategy, which:
- Extracts complete sentences directly from context
- Validates each sentence using string matching, fuzzy matching, and substring matching
- Reorders sentences using LLM for logical coherence
- **Zero hallucination by design** — only text that exists in context is used

#### 2. TransitionGenerate (Transition-Based Cohesion)

A balanced strategy that:
- Extracts core sentences from context
- Generates transition sentences to connect them
- Produces cohesive responses with natural flow while maintaining faithfulness

### Extractiveness Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **Coverage** | extracted_tokens / response_tokens | Fraction of response from context |
| **Density** | Σ(fragment_len²) / response_tokens | Concentration of extracted fragments |
| **Extractiveness Score** | coverage × 0.6 + min(density^0.25 / 4, 0.4) | Overall extraction quality |

### Data Filtering

Two-stage pipeline to construct DPO training pairs:

**Stage 1 — Four-dimensional filtering:**
| Metric | Threshold | Description |
|--------|-----------|-------------|
| AlignScore | ≥ 0.85 | Semantic alignment with context |
| Coverage | ≥ 0.70 | Content from context |
| PPL | ≤ 35 | Language fluency |
| Ratio | ≤ 1.5 | Response/context length ratio |

**Stage 2 — Pairwise comparison:**
Six pipeline responses are compared pairwise, scored on faithfulness, completeness, and fluency. The highest-scoring response becomes `chosen`, the lowest becomes `rejected`.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from FaithDPO import SentenceExtractPipeline, TransitionGeneratePipeline

# Initialize pipelines
sentence_pipe = SentenceExtractPipeline(verbose=True)
transition_pipe = TransitionGeneratePipeline(verbose=True)

# Process a RAG query
context = "Your retrieved context here..."
query = "Your question here..."

# SentenceExtract: strict extraction
result1 = sentence_pipe.process(context, query)
print(result1["output"])

# TransitionGenerate: with transitions
result2 = transition_pipe.process(context, query)
print(result2["output"])
```

## Using the Client

```python
from FaithDPO import FaithDPOClient

client = FaithDPOClient(pipeline="sentence_extract")
response = client.ask(context, query)
print(response.content)
```

## Dataset

Training data is constructed from three datasets:

| Dataset | Domain | Samples |
|---------|--------|---------|
| PubMedQA | Medical QA | 195 |
| FaithEval | Commonsense | 102 |
| RAGTruth | Open-domain | 68 |

Final DPO dataset: **365 samples × 2 strategies × 3 candidates = 1095 training pairs**

## Training

### Data Conversion

```bash
python script/convert_dpo_data.py
```

### DPO Training

```bash
# Qwen2.5-7B-Instruct example
CUDA_VISIBLE_DEVICES=1,2 python script/train_dpo_faith.py
```

Key training parameters:
- PEFT: LoRA (rank=64, alpha=128)
- beta (DPO temperature): 0.1
- learning_rate: 5e-6
- batch_size: 2
- epochs: 2

## Results

| Method | Faithfulness (RAGAS) | Coverage | Density | Extractiveness Score |
|--------|---------------------|----------|---------|---------------------|
| Baseline (Llama-3-8B) | 0.62 | 0.31 | 2.1 | 0.29 |
| SentenceExtract | 0.71 | 0.78 | 4.8 | 0.68 |
| TransitionGenerate | 0.74 | 0.76 | 5.2 | 0.71 |
| **FaithDPO (fine-tuned)** | **0.81** | **0.82** | **5.9** | **0.76** |

SentenceExtract and TransitionGenerate show **+12%~+19%** improvement in Faithfulness over the baseline. The fine-tuned model achieves **0.81 Faithfulness** with significantly improved extractiveness.

## Project Structure

```
FaithDPO/
├── FaithDPO/
│   ├── pipelines/
│   │   ├── base.py                 # Base pipeline class
│   │   ├── sentence_extract.py      # SentenceExtract strategy
│   │   └── transition_generate.py   # TransitionGenerate strategy
│   ├── metrics/
│   │   └── extractiveness.py       # Extractiveness metrics
│   ├── utils/
│   │   ├── llm.py                  # LLM backend wrapper
│   │   └── text.py                 # Text processing utilities
│   ├── client.py                   # OpenAI-style client
│   └── config.py                   # Configuration management
├── examples/
│   ├── basic_usage.py
│   ├── pipeline_comparison.py
│   └── batch_processing.py
├── script/
│   ├── convert_dpo_data.py         # Data conversion
│   └── train_dpo_faith.py         # DPO training script
├── FaithDPO-Data/                  # Training data
└── README.md
```

## License

MIT License
