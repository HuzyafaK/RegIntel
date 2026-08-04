# RegIntel

Neuro-symbolic compliance intelligence engine. Combines LLM-based fact
extraction, a symbolic rule engine with 3-valued logic, Inductive Logic
Programming (ILP) for compound rule discovery, and an HMAC-signed
tamper-evident audit trail.

## Demo

**▶️ [Watch the demo video](https://www.youtube.com/watch?v=JA2PcbBO2ek)**

[![▶ RegIntel demo video](https://img.youtube.com/vi/JA2PcbBO2ek/hqdefault.jpg)](https://www.youtube.com/watch?v=JA2PcbBO2ek)

## Repository structure

```
RegIntel/
├── regintel_engine/            Core Python package
│   ├── models.py                  Data models — FactDict, ExtractedFact, Verdict
│   ├── pipeline.py                RegIntelPipeline — main entry point, backend orchestration
│   ├── vocabulary.py              45-predicate controlled vocabulary across 4 domains
│   ├── rules/
│   │   └── engine.py              SymbolicRuleEngine + RuleCompiler — 24 base rules, 3-valued logic
│   ├── ilp/
│   │   └── sparse_ilp.py          Inductive Logic Programming — compound rule discovery
│   ├── audit/
│   │   └── chain.py               HMACAuditChain — HMAC-SHA256 tamper-evident audit trail
│   └── extraction/
│       ├── llm_extractor.py       LLM parser — Groq, Ollama, JAX, PyTorch backends
│       └── verifier.py            Hallucination verifier — rejects non-verbatim extractions
│
├── notebooks/                  Kaggle notebook scripts (see notebooks/README.md for run order)
│   ├── kaggle_prototype_notebook.py
│   ├── kaggle_preprocessing_notebook.py
│   ├── kaggle_notebook_per_cell.py
│   ├── kaggle_notebook_complete.py
│   └── kaggle_evaluation_notebook.py
│
├── scripts/                    Standalone data/training utilities
│   └── build_ilp_data_from_gt.py  Builds ILP training facts from ground-truth labels
│
├── requirements.txt
├── pyproject.toml
└── LICENSE
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

## Installation

```bash
pip install -r requirements.txt
```

The core engine has no third-party dependencies in `mock_llm=True` (GT-ILP)
mode. Install the extras in `requirements.txt` only for the backend you
intend to use (Groq, JAX/Keras, or PyTorch/HuggingFace).

## Verified State

- 24 base rules across 4 domains (GMP, FDA Device, Clinical Trials, SEC Financial)
- All modules import cleanly with no errors
- Engine initializes correctly in GT-ILP mode
- ILP discovers 1 compound rule (conf=1.00, support=111) on the 1,265-doc training set
- Evaluated: 100% decision accuracy, 100% safety recall, 0 false negatives on 150-doc test set

## License

See [LICENSE](LICENSE).
