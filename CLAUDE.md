# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FaithDPO implements two extractive RAG pipelines that reduce hallucinations by prioritizing content extraction from context over generation. The paper demonstrates that extraction-based approaches significantly reduce hallucination while maintaining response quality.

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .

# Run examples
python examples/basic_usage.py
python examples/pipeline_comparison.py

# Convert data to DPO format
python script/convert_dpo_data.py

# Run DPO training
python script/train_dpo_faith.py
```

## Architecture

### Entry Point: FaithDPOClient (`FaithDPO/client.py`)
- OpenAI-style client interface (`client.responses.create()`)
- Routes requests to the appropriate pipeline
- Returns `FaithDPOResponse` with content and extractiveness metrics

### Two Pipelines (`FaithDPO/pipelines/`)
All pipelines inherit from `PipelineBase` and implement `process(context, query)`:

1. **SentenceExtract** (`sentence_extract.py`) - Strictest extraction
   - Extracts complete sentences verbatim from context
   - Validates each sentence using string/fuzzy/substring matching
   - Uses LLM to intelligently order sentences
   - Zero hallucination by design

2. **TransitionGenerate** (`transition_generate.py`) - Transition-based cohesion
   - Extracts core sentences from context
   - Generates transition sentences between them
   - Produces cohesive responses with natural flow

### LLM Backend (`FaithDPO/utils/llm.py`)
- Wraps OpenAI API (or compatible endpoints like Azure, SiliconFlow)
- Supports `enable_thinking` parameter for reasoning models
- `call()` and `call_with_system()` methods

### Extractiveness Metrics (`FaithDPO/metrics/extractiveness.py`)
- Calculates Coverage (token overlap) and Density (fragment length)
- Overall score: `coverage * 0.6 + min(density**0.25 / 4, 0.4)`
- Supports Chinese (via jieba) and English
- Visualization: `render_extractiveness_heatmap()` (terminal ANSI)

### Configuration (`FaithDPO/config.py`)
- Loads from `.env` file (searches cwd, package dir, parent dir)
- Required: `OPENAI_API_KEY`
- Optional: `OPENAI_BASE_URL`, `DEFAULT_MODEL`, `DEFAULT_PIPELINE`, `DEFAULT_TEMPERATURE`, `DEFAULT_TIMEOUT`, `VERBOSE`

## Key Design Patterns

- **Pipeline pattern**: Each pipeline extends `PipelineBase` with shared LLM calling and extractiveness tracking
- **Response dataclass**: `FaithDPOResponse` with `render_heatmap()` and `get_fragments()` helpers
- **Standardized output**: All pipelines return dict via `create_standard_response_structure()` with keys: `output`, `extractiveness_coverage`, `extractiveness_density`, `extractiveness_score`, `extra`
