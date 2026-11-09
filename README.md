# RegIntel

**An evolving neuro-symbolic compliance intelligence engine.** RegIntel reads
regulatory documents (GMP batch records, FDA device submissions, clinical
trial protocols, SEC financial filings), extracts discrete facts with an
LLM, and hands those facts to a symbolic rule engine that produces an
auditable, three-valued compliance verdict — never a black-box score.

The core bet: **LLMs should classify text into a controlled vocabulary,
not reason about compliance directly.** All actual compliance logic lives
in human-auditable symbolic rules with real regulatory citations. The LLM
is a parser, not a judge.

**[▶️ Watch the demo video](https://www.youtube.com/watch?v=JA2PcbBO2ek)**

[![RegIntel demo video](https://img.youtube.com/vi/JA2PcbBO2ek/hqdefault.jpg)](https://www.youtube.com/watch?v=JA2PcbBO2ek)

---

## Why neuro-symbolic

Pure LLM compliance checking fails two ways that matter in regulated
industries: it hallucinates evidence that isn't in the source document,
and it can't produce a verdict a regulator can independently re-derive.
RegIntel enforces a hard boundary between the two layers:

- **Neural layer** (extraction) — an LLM reads the document and maps
  language onto a *fixed, SME-owned* predicate vocabulary. It never
  invents new predicates and every extraction must be traceable back to
  a verbatim span of the source text.
- **Symbolic layer** (rules + ILP) — a deterministic rule engine
  evaluates those discrete facts against rules with real regulatory
  citations (21 CFR, etc.). No floats, embeddings, or LLM output ever
  reach this layer directly — only discrete values with provenance.

This is enforced structurally, not by convention: `FactDict` (the
type the rule engine consumes) is invariant to only ever contain
discrete values, and every rule evaluation carries the citation and
extracted evidence text that produced it.

## Architecture

```
                    ┌─────────────────────┐
   document text →  │   LLM Extractor      │  Groq / Ollama / JAX / PyTorch
                    │   (llm_extractor.py)  │  → ExtractedFact[] with
                    └──────────┬────────────┘    confidence + verbatim text
                               │
                    ┌──────────▼────────────┐
                    │  Hallucination         │  rejects facts whose
                    │  Verifier (verifier.py)│  extracted_text isn't a
                    └──────────┬────────────┘  verbatim substring of the doc
                               │
                    ┌──────────▼────────────┐
                    │   FactDict              │  discrete-only, provenance-
                    │   (models.py)            │  tracked, absent facts →
                    └──────────┬────────────┘  None (not a default value)
                               │
              ┌────────────────┼────────────────┐
              │                                  │
   ┌──────────▼───────────┐         ┌───────────▼────────────┐
   │ Symbolic Rule Engine   │         │  Sparse ILP              │
   │ (rules/engine.py)       │◄───────│  (ilp/sparse_ilp.py)       │
   │ 24 base rules,           │  SME    │  discovers new compound   │
   │ 3-valued logic            │ approves│  rules from labeled data  │
   └──────────┬───────────┘  compound └───────────────────────────┘
              │                rules
   ┌──────────▼────────────┐
   │  Verdict                 │  COMPLIANT | NON_COMPLIANT | NEEDS_REVIEW
   │  (models.py)               │
   └──────────┬────────────┘
              │
   ┌──────────▼────────────┐
   │  HMAC Audit Chain        │  every record HMAC-SHA256 signed and
   │  (audit/chain.py)          │  hash-linked to the previous record —
   └────────────────────────┘  21 CFR Part 11–style tamper evidence
```

## Core design decisions

**Three-valued logic, not binary.** Every rule and every verdict can be
`True` (passed), `False` (confirmed violation), or `None` (uncertain).
`None` is *not* treated as failure — a document with missing evidence
routes to `NEEDS_REVIEW`, not `NON_COMPLIANT`. This distinction is what
lets the system report 0 false negatives on held-out data: it never
guesses past its evidence.

**Controlled predicate vocabulary (45 predicates, 4 domains).** Every
predicate the LLM is allowed to extract is pre-registered with a domain
(GMP / FDA device / clinical trials / SEC financial), a causal direction
(compliant / non-compliant / neutral, for automatic rule interpretability
checks), and a verified regulatory citation with an ecfr.gov link. LLMs
classify text *into* this vocabulary — they cannot generate new
predicates. Domain boundaries are enforced so, e.g., a GMP calibration
predicate can never leak into a clinical-trial rule.

**Domain-constrained sparse ILP for rule discovery.** Beyond the 24
hand-authored base rules, `ilp/sparse_ilp.py` learns new *compound* Horn
clauses directly from labeled documents — gradient descent with L1
regularization over a sparse predicate-presence matrix (absent facts are
excluded from scoring entirely rather than padded with a neutral value).
The result is short, human-readable rule bodies (most real compliance
rules involve 1–3 predicates), each with a confidence and support count.
Discovered rules are **not** auto-activated — they enter a `pending`
state and require SME sign-off (`HornClause.is_approvable`) before they
can be compiled into an active `ComplianceRule`.

**Hallucination-resistant extraction.** `extraction/verifier.py` rejects
any extracted fact whose `extracted_text` is not a verbatim substring of
the source document, and extractions below a 0.60 confidence threshold
are marked `low_confidence` and treated as absent rather than trusted.

**Tamper-evident audit trail.** `audit/chain.py` seals every verdict —
full evidence chain included (extracted facts, rule results, citations,
SME approvals for ILP-derived rules) — with an HMAC-SHA256 signature that
also hashes in the previous record, so any retroactive edit to any past
record breaks every signature after it. Built to be independently
verifiable by a regulator, not just by RegIntel itself.

**Pluggable extraction backends.** The same pipeline runs against Groq
(cloud, `llama-3.3-70b-versatile`), Ollama (local, data-sovereign
deployments), or JAX/Keras and PyTorch/HuggingFace backends — swappable
via `RegIntelPipeline.create(backend=...)` without touching the symbolic
layer.

## Repository structure

```
RegIntel/
├── regintel_engine/            Core Python package
│   ├── models.py                  Data model: ExtractedFact, FactDict, RuleResult,
│   │                                Verdict, ComplianceRule, HornClause, AuditRecord
│   ├── pipeline.py                RegIntelPipeline — orchestrates extraction → rules → audit
│   ├── vocabulary.py              45-predicate controlled vocabulary, 4 domains, SME-owned
│   ├── rules/
│   │   └── engine.py              SymbolicRuleEngine + RuleCompiler — 24 base rules, 3-valued logic
│   ├── ilp/
│   │   └── sparse_ilp.py          Domain-constrained sparse ILP — compound rule discovery
│   ├── audit/
│   │   └── chain.py               HMACauditChain — HMAC-SHA256 tamper-evident, hash-linked audit trail
│   └── extraction/
│       ├── llm_extractor.py       LLM fact extractor — Groq / Ollama / JAX / PyTorch backends
│       └── verifier.py            Hallucination verifier — rejects non-verbatim extractions
│
├── notebooks/                  Kaggle notebooks (see notebooks/README.md for run order)
│   ├── kaggle_prototype_notebook.py     GT-ILP proof-of-concept, no LLM required
│   ├── kaggle_preprocessing_notebook.py  Groq batch extraction over the training set
│   ├── kaggle_notebook_per_cell.py       Primary setup notebook, engine written cell-by-cell
│   ├── kaggle_notebook_complete.py       Same pipeline, single self-contained cell
│   └── kaggle_evaluation_notebook.py     Standalone evaluation harness + SME report
│
├── scripts/
│   └── build_ilp_data_from_gt.py   Builds ILP training facts directly from ground-truth labels
│
├── requirements.txt             Grouped by optional backend (Groq / JAX-Keras / PyTorch)
├── pyproject.toml               Packaging metadata
└── LICENSE
```

## Quick start

```bash
pip install -r requirements.txt
```

```python
from regintel_engine.pipeline import RegIntelPipeline
from regintel_engine.ilp.sparse_ilp import ILPConfig

# GT-ILP mode — no LLM required, uses ground-truth facts directly.
# Useful for testing the symbolic engine and ILP in isolation.
pipeline = RegIntelPipeline.create(
    mock_llm=True,
    audit_key='your-64-char-hex-key',
)
print(f"{pipeline.engine.n_base_rules} base rules loaded")

# Production — Groq backend
pipeline = RegIntelPipeline.create(
    backend="groq",
    groq_model="llama-3.3-70b-versatile",
    groq_api_key="your-groq-key",
    audit_key='your-64-char-hex-key',
)
result = pipeline.run(document_text, domain="gmp")
print(result.verdict.verdict)  # COMPLIANT | NON_COMPLIANT | NEEDS_REVIEW

# Production — local Ollama (data sovereignty, no external API calls)
pipeline = RegIntelPipeline.create(
    backend="ollama",
    ollama_model="gemma3:27b",
    ollama_url="http://localhost:11434",
    audit_key='your-64-char-hex-key',
)
```

The core engine (models, rules, ILP, audit chain) has **zero third-party
dependencies** in `mock_llm=True` mode — it runs on the standard library
alone. Install the extras in `requirements.txt` only for the LLM backend
you intend to use.

## Verified results

Evaluated on a 1,265-document training set / 150-document held-out test
set spanning all four domains:

| Metric | Result |
|---|---|
| Base rules | 24, across GMP / FDA Device / Clinical Trials / SEC Financial |
| Decision accuracy | 100% on the 150-doc test set |
| Safety recall | 100% (0 false negatives — no missed violations) |
| ILP rule discovery | 1 compound rule, confidence 1.00, support 111 docs |
| Module integrity | All modules import cleanly, engine initializes correctly in GT-ILP mode |

Zero false negatives is the metric that matters most for a compliance
system: `NEEDS_REVIEW` absorbs uncertainty instead of letting it collapse
into a false `COMPLIANT`.

## License

See [LICENSE](LICENSE).
