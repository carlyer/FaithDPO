"""
FaithDPO - Faithful Extraction for RAG Hallucination Reduction

This package provides two extraction-based RAG pipelines that prioritize extracting
content from context rather than generating new information, significantly reducing
hallucinations in RAG systems.

## Pipelines

- **SentenceExtract**: Ultra-strict extraction with intelligent sentence ordering
- **TransitionGenerate**: Cohesive responses with transition generation

## Quick Start

```python
from FaithDPO import FaithDPOClient

client = FaithDPOClient(
    model="gpt-4o-mini",
    default_pipeline="sentence_extract"
)

response = client.responses.create(
    context="Your context document...",
    query="Your question here?"
)

print(response.content)
print(f"Extractiveness: {response.extractiveness_score:.3f}")
```

## Citation

If you use this package, please cite:

```bibtex
@article{faithdpo,
  title={FaithDPO: Faithful Extraction for RAG Hallucination Reduction via Direct Preference Optimization},
  year={2024}
}
```
"""

__version__ = "0.1.0"

from FaithDPO.client import FaithDPOClient, FaithDPOResponse
from FaithDPO.pipelines import (
    PipelineBase,
    SentenceExtractPipeline,
    TransitionGeneratePipeline,
)
from FaithDPO.metrics import ExtractivenessMetrics
from FaithDPO.utils import LLMBackend, TextProcessor, create_standard_response_structure

__all__ = [
    # Version
    '__version__',

    # Client
    'FaithDPOClient',
    'FaithDPOResponse',

    # Pipelines
    'PipelineBase',
    'SentenceExtractPipeline',
    'TransitionGeneratePipeline',

    # Metrics
    'ExtractivenessMetrics',

    # Utilities
    'LLMBackend',
    'TextProcessor',
    'create_standard_response_structure',
]
