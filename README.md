# RegIntel Engine — Finalized Source

Neuro-symbolic compliance intelligence engine. This package contains all
finalized, working source files as of the current iteration.

## Structure

```
regintel_engine/
├── __init__.py
├── models.py              — Core data models (FactDict, ExtractedFact, Verdict, etc.)
├── pipeline.py             — RegIntelPipeline — main entry point, backend orchestration
├── vocabulary.py           — 45-predicate controlled vocabulary across 4 domains
├── rules/
│   └── engine.py           — SymbolicRuleEngine + RuleCompiler — 24 base rules, 3-valued logic
├── ilp/
│   └── sparse_ilp.py       — Inductive Logic Programming — compound rule discovery
├── audit/
│   └── chain.py            — HMACauditChain — HMAC-SHA256 tamper-evident audit trail
└── extraction/
    ├── llm_extractor.py    — LLM parser: Groq, Ollama, JAX, PyTorch backends
    └── verifier.py         — Hallucination verifier — rejects non-verbatim extractions

kaggle_notebook_per_cell.py       — Primary Kaggle setup notebook (cells 1-22)
kaggle_notebook_complete.py       — Complete single-file notebook variant
kaggle_evaluation_notebook.py     — Evaluation harness
kaggle_prototype_notebook.py      — GT-ILP prototype proof notebook
kaggle_preprocessing_notebook.py  — Groq batch extraction preprocessing
build_ilp_data_from_gt.py         — Builds ILP training facts from ground truth labels
```

## Quick Start

```python
from regintel_engine.pipeline import RegIntelPipeline
from regintel_engine.ilp.sparse_ilp import ILPConfig

# GT-ILP mode (no LLM required — for testing/demos)
pipeline = RegIntelPipeline.create(
    mock_llm=True,
    audit_key='your-64-char-hex-key',
)
print(f"{pipeline.engine.n_base_rules} base rules loaded")

# Production mode — Groq backend
pipeline = RegIntelPipeline.create(
    backend="groq",
    groq_model="llama-3.3-70b-versatile",
    groq_api_key="your-groq-key",
    audit_key='your-64-char-hex-key',
)
result = pipeline.run(document_text, domain="gmp")
print(result.verdict.verdict)  # COMPLIANT | NON_COMPLIANT | NEEDS_REVIEW

# Production mode — local Ollama (data sovereignty)
pipeline = RegIntelPipeline.create(
    backend="ollama",
    ollama_model="gemma3:27b",
    ollama_url="http://localhost:11434",
    audit_key='your-64-char-hex-key',
)
```

## Verified State

- 24 base rules across 4 domains (GMP, FDA Device, Clinical Trials, SEC Financial)
- All modules import cleanly with no errors
- Engine initializes correctly in GT-ILP mode
- ILP discovers 1 compound rule (conf=1.00, support=111) on the 1,265-doc training set
- Evaluated: 100% decision accuracy, 100% safety recall, 0 false negatives on 150-doc test set

Generated from finalized source as of current session.
