"""
regintel_engine.pipeline
=========================
Main RegIntel compliance pipeline.

The complete neuro-symbolic pipeline:

TEST TIME (every document):
  Document text
      ↓ LLM extraction (only AI at test time)
  ExtractedFacts
      ↓ Hallucination verification
  Verified facts
      ↓ Normalization + text classification
  FactDict (discrete only — no embeddings)
      ↓ Symbolic rule engine (zero AI)
  Verdict
      ↓ Audit chain (HMAC signed)
  AuditRecord

TRAINING TIME (once, offline):
  Labeled documents
      ↓ LLM extraction (same pipeline)
  (facts, label) pairs
      ↓ Domain-constrained ILP
  HornClauses
      ↓ DreamCoder abstraction
  Primitives + refined clauses
      ↓ SME review (human approves)
  Approved clauses
      ↓ Rule compiler
  ComplianceRules → symbolic engine

Usage
-----
    pipeline = RegIntelPipeline.create()

    # Test time
    result = pipeline.run(document_text, domain="gmp")
    print(result.verdict.verdict)

    # Training time
    pipeline.training.train(labeled_examples)
    pipeline.training.approve_clause(clause, "sme@company.com", ...)
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from .models import (
    ExtractedFact, FactDict, HornClause, Verdict, AuditRecord,
)
from .vocabulary import PredicateVocabulary, EncodingTables
from .extraction.llm_extractor import LLMFactExtractor
from .extraction.verifier import (
    HallucinationVerifier, FactNormalizer, FactDictBuilder,
)
from .rules.engine import SymbolicRuleEngine, RuleCompiler
from .audit.chain import HMACauditChain
from .ilp.sparse_ilp import SparseILP, DreamCoder, ILPConfig

log = logging.getLogger("regintel.pipeline")


# ─────────────────────────────────────────────────────────────────────────────
#  PIPELINE RESULT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """Complete output from one document evaluation."""
    verdict:      Verdict
    audit_record: AuditRecord
    facts:        FactDict

    @property
    def label(self) -> str:
        return self.verdict.verdict

    @property
    def is_compliant(self) -> bool:
        return self.label == "COMPLIANT"

    def explain(self) -> str:
        """Human-readable explanation of the verdict."""
        lines = [
            f"Verdict: {self.verdict.verdict}",
            f"Domain:  {self.verdict.domain}",
            "",
        ]

        failed = self.verdict.failed_rules
        if failed:
            lines.append("FINDINGS:")
            for r in failed:
                lines.append(f"  [{r.severity.upper()}] {r.rule_id}: {r.message}")
                for pred, ev in r.evidence.items():
                    if ev.get("value") is not None:
                        lines.append(
                            f"    {pred} = {ev['value']}"
                            f" ← \"{ev.get('extracted_text', '')[:80]}\""
                        )
            lines.append("")

        uncertain = self.verdict.uncertain_rules
        if uncertain:
            lines.append("UNCERTAIN (insufficient evidence):")
            for r in uncertain:
                lines.append(f"  [{r.severity.upper()}] {r.rule_id}: {r.message}")
            lines.append("")

        passed = [r for r in self.verdict.rule_results if r.passed is True]
        if passed:
            lines.append(f"PASSED: {len(passed)} rules")

        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  TRAINING ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

class TrainingOrchestrator:
    """
    Orchestrates the training-time pipeline.

    Runs ILP on extracted facts, manages DreamCoder abstraction,
    handles SME approval workflow, compiles approved rules.
    """

    def __init__(
        self,
        vocab:    PredicateVocabulary,
        engine:   SymbolicRuleEngine,
        ilp:      SparseILP,
        compiler: RuleCompiler,
        dreamcoder: DreamCoder,
    ) -> None:
        self.vocab      = vocab
        self.engine     = engine
        self.ilp        = ilp
        self.compiler   = compiler
        self.dreamcoder = dreamcoder
        self._pending:  List[HornClause] = []

    def train(
        self,
        examples:           List[Dict[str, Any]],
        run_dreamcoder:     bool = True,
    ) -> List[HornClause]:
        """
        Run ILP on labeled examples to discover rules.

        Parameters
        ----------
        examples     List of {"facts": sparse_dict, "label": "COMPLIANT"|"NON_COMPLIANT",
                              "domain": "gmp"|"fda_device"|"clinical"|"sec_financial"}
        run_dreamcoder  Whether to run DreamCoder abstraction after ILP

        Returns
        -------
        Discovered HornClauses (pending SME approval)
        """
        # Group examples by domain
        by_domain: Dict[str, List[Dict]] = {}
        for ex in examples:
            domain = ex.get("domain", "gmp")
            by_domain.setdefault(domain, []).append(ex)

        all_clauses = []
        for domain, domain_examples in by_domain.items():
            log.info(f"[Training] ILP for domain={domain} "
                     f"n={len(domain_examples)}")
            clauses = self.ilp.induce(domain_examples, domain)
            all_clauses.extend(clauses)
            log.info(f"[Training] domain={domain}: "
                     f"{len(clauses)} clauses discovered")

        if run_dreamcoder and all_clauses:
            new_primitives = self.dreamcoder.wake(all_clauses)
            if new_primitives:
                all_clauses = self.dreamcoder.sleep(all_clauses)
                log.info(f"[DreamCoder] {len(new_primitives)} primitives, "
                         f"{len(all_clauses)} compressed clauses")

        self._pending.extend(all_clauses)

        log.info(f"[Training] Complete: {len(all_clauses)} total clauses "
                 f"({len(self._pending)} pending approval)")
        return all_clauses

    def pending_clauses(self) -> List[HornClause]:
        """Return clauses awaiting SME review."""
        return [c for c in self._pending if c.status == "pending"]

    def approve_clause(
        self,
        clause_id:    str,
        approved_by:  str,
        sme_statement: str,
        regulation:   Optional[str] = None,
    ) -> bool:
        """
        SME approves a discovered clause.
        Compiles it to a ComplianceRule and adds to the engine.
        """
        clause = next((c for c in self._pending if c.clause_id == clause_id), None)
        if not clause:
            log.warning(f"[Training] Clause not found: {clause_id}")
            return False

        if not clause.is_approvable:
            log.warning(f"[Training] Clause {clause_id} not approvable: "
                        f"conf={clause.confidence:.2f}, "
                        f"body={clause.body}")
            return False

        clause.status = "approved"

        approved_on = datetime.utcnow().strftime("%Y-%m-%d")

        rule = self.compiler.compile(
            clause        = clause,
            approved_by   = approved_by,
            approved_on   = approved_on,
            sme_statement = sme_statement,
        )

        self.engine.add_rule(rule)
        log.info(f"[Training] Approved and compiled: {rule.rule_id}")
        return True

    def reject_clause(self, clause_id: str, reason: str) -> None:
        clause = next((c for c in self._pending if c.clause_id == clause_id), None)
        if clause:
            clause.status = "rejected"
            log.info(f"[Training] Rejected clause {clause_id}: {reason}")

    def sme_review_summary(self) -> str:
        """Print summary for SME review."""
        pending = self.pending_clauses()
        if not pending:
            return "No clauses pending review."

        lines = [
            f"{'='*70}",
            f"SME REVIEW REQUIRED — {len(pending)} clauses pending",
            f"{'='*70}",
        ]
        for c in pending:
            approvable = "✓ APPROVABLE" if c.is_approvable else "✗ NOT APPROVABLE"
            lines.append(f"\n[{c.clause_id}] {approvable}")
            lines.append(f"  Notation:   {c.notation}")
            lines.append(f"  Domain:     {c.domain}")
            lines.append(f"  Confidence: {c.confidence:.2f}")
            lines.append(f"  Coverage:   {c.coverage:.2f}")
            lines.append(f"  Support:    {c.support} training docs")
            lines.append(f"  Predicates: {', '.join(c.body)}")

            # Show citations from vocabulary
            for pred in c.body:
                entry = self.vocab.get(pred)
                if entry:
                    lines.append(f"  Citation [{pred}]: {entry.regulation}")

        lines.append(f"\n{'='*70}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

class RegIntelPipeline:
    """
    The complete RegIntel neuro-symbolic compliance pipeline.

    At test time:
      1. LLM extracts facts (only AI — runs locally, no external calls)
      2. Verifier checks grounding (no AI — string operations)
      3. Normalizer maps to standard vocabulary (no AI)
      4. FactDictBuilder creates discrete FactDict (no AI)
      5. SymbolicRuleEngine evaluates rules (no AI — Python functions)
      6. AuditChain seals HMAC record (no AI)

    At training time:
      1. Same extraction pipeline on labeled documents
      2. ILP discovers rules from (facts, label) pairs
      3. DreamCoder abstracts patterns to primitives
      4. SME reviews and approves clauses
      5. RuleCompiler creates compiled Python functions
      6. Engine updated with new rules
    """

    def __init__(
        self,
        extractor:  LLMFactExtractor,
        verifier:   HallucinationVerifier,
        normalizer: FactNormalizer,
        builder:    FactDictBuilder,
        engine:     SymbolicRuleEngine,
        audit:      HMACauditChain,
        vocab:      PredicateVocabulary,
        training:   TrainingOrchestrator,
    ) -> None:
        self.extractor  = extractor
        self.verifier   = verifier
        self.normalizer = normalizer
        self.builder    = builder
        self.engine     = engine
        self.audit      = audit
        self.vocab      = vocab
        self.training   = training

    @classmethod
    def create(
        cls,
        llm_model:        str            = "Qwen/Qwen2.5-7B-Instruct",
        mock_llm:         bool           = False,
        backend:          str            = "pytorch",   # "groq"|"ollama"|"pytorch"|"jax"|"mock"
        tpu_preset:       Optional[str]  = None,
        tpu_dtype:        str            = "bfloat16",
        load_in_8bit:     bool           = False,
        load_in_4bit:     bool           = False,
        max_input_length: int            = 8000,
        # Groq-specific
        groq_model:       str            = "llama-3.3-70b-versatile",
        groq_api_key:     Optional[str]  = None,
        # Ollama-specific (own hardware)
        ollama_model:     str            = "gemma3:27b",
        ollama_url:       str            = "http://localhost:11434",
        audit_key:        Optional[str]  = None,
        ilp_config:       Optional[ILPConfig] = None,
    ) -> "RegIntelPipeline":
        """
        Create a complete pipeline.

        Recommended backends
        --------------------
        backend="groq"   (cloud API, no GPU needed — MVP/demo)
            groq_model options:
              "llama-3.3-70b-versatile"  — best quality, 6k tok/min free
              "openai/gpt-oss-120b"      — higher rate limits than llama 70b
              "llama-3.1-8b-instant"     — fastest, good quality

        backend="ollama"  (own hardware — Gemma 4, Llama 70B locally)
            ollama_model options:
              "gemma3:27b"   — best quality/size for regulatory extraction
              "gemma4:27b"   — newer, install: ollama pull gemma4:27b
              "llama3.3:70b" — highest quality (needs ~80 GB RAM)
            Setup: curl -fsSL https://ollama.com/install.sh | sh
                   ollama pull gemma3:27b && ollama serve

        backend="jax"    (Kaggle TPU v5e-8)
            tpu_preset="gemma3_instruct_27b"  — recommended

        backend="mock"   (testing)
        """
        vocab    = PredicateVocabulary.default()
        encoding = EncodingTables()
        all_vocab_names = set(vocab.get_names())

        # ── Build extractor ───────────────────────────────────────────────────
        if mock_llm or backend == "mock":
            extractor = LLMFactExtractor.mock()

        elif backend == "groq":
            extractor = LLMFactExtractor.from_groq(
                model            = groq_model,
                api_key          = groq_api_key,
                max_input_length = max_input_length,
            )

        elif backend == "ollama":
            extractor = LLMFactExtractor.from_ollama(
                model            = ollama_model,
                base_url         = ollama_url,
                max_input_length = max_input_length,
            )

        elif backend == "jax":
            preset = tpu_preset or "gemma3_instruct_27b"
            extractor = LLMFactExtractor.from_tpu(
                model_preset     = preset,
                dtype            = tpu_dtype,
                max_input_length = max_input_length,
            )

        else:  # pytorch
            extractor = LLMFactExtractor.from_pytorch(
                model_name       = llm_model,
                load_in_8bit     = load_in_8bit,
                load_in_4bit     = load_in_4bit,
                max_input_length = max_input_length,
            )
        """
        Create a complete pipeline.

        Backend options
        ---------------
        backend="groq"  (recommended for MVP)
            Calls Groq Cloud API — no GPU/TPU needed.
            Best model: "llama-3.3-70b-versatile" (128k context, very fast)
            Requires: pip install groq  +  GROQ_API_KEY env var

        backend="mock"  (testing / CI)
            No model needed. Deterministic keyword extraction.

        backend="jax"   (Kaggle TPU v5e-8)
            keras_hub + JAX. Set tpu_preset="gemma3_instruct_27b".

        backend="pytorch"  (GPU)
            HuggingFace transformers. Use load_in_8bit=True for 70B.

        Quick start (Groq)
        ------------------
            import os
            os.environ["GROQ_API_KEY"] = "your_key"

            pipeline = RegIntelPipeline.create(
                backend   = "groq",
                groq_model = "llama-3.3-70b-versatile",
            )
        """
        vocab    = PredicateVocabulary.default()
        encoding = EncodingTables()
        all_vocab_names = set(vocab.get_names())

        # ── Build extractor ───────────────────────────────────────────────────
        if mock_llm or backend == "mock":
            extractor = LLMFactExtractor.mock()

        elif backend == "groq":
            extractor = LLMFactExtractor.from_groq(
                model            = groq_model,
                api_key          = groq_api_key,
                max_input_length = max_input_length,
            )

        elif backend == "jax":
            preset = tpu_preset or "gemma3_instruct_27b"
            extractor = LLMFactExtractor.from_tpu(
                model_preset     = preset,
                dtype            = tpu_dtype,
                max_input_length = max_input_length,
            )

        else:  # pytorch
            extractor = LLMFactExtractor.from_pytorch(
                model_name       = llm_model,
                load_in_8bit     = load_in_8bit,
                load_in_4bit     = load_in_4bit,
                max_input_length = max_input_length,
            )

        verifier   = HallucinationVerifier()
        normalizer = FactNormalizer()
        builder    = FactDictBuilder(encoding, vocab_names=all_vocab_names)
        engine     = SymbolicRuleEngine(vocab)
        audit      = HMACauditChain(key_hex=audit_key)
        ilp        = SparseILP(vocab, ilp_config or ILPConfig())
        compiler   = RuleCompiler(vocab)
        dreamcoder = DreamCoder()

        training = TrainingOrchestrator(
            vocab=vocab, engine=engine, ilp=ilp,
            compiler=compiler, dreamcoder=dreamcoder,
        )

        log.info(
            f"[Pipeline] backend={extractor._backend}  "
            f"max_input={max_input_length}  "
            f"base_rules={engine.n_rules}"
        )

        return cls(
            extractor  = extractor,
            verifier   = verifier,
            normalizer = normalizer,
            builder    = builder,
            engine     = engine,
            audit      = audit,
            vocab      = vocab,
            training   = training,
        )

    def run(
        self,
        document: str,
        domain:   str = "gmp",
    ) -> PipelineResult:
        """
        Run the complete pipeline on one document.

        STAGE 1 — LLM extraction (only AI)
        STAGE 2 — Verification + normalization (no AI)
        STAGE 3 — Symbolic evaluation (no AI)
        STAGE 4 — Audit sealing (no AI)
        """
        doc_hash = hashlib.sha256(document.encode()).hexdigest()

        # ── Stage 1: LLM extracts facts ───────────────────────────────────────
        structured_raw = self.extractor.extract_structured(document, domain)

        # Classify text facts (deviation descriptions, summaries, etc.)
        text_preds = []
        vocab_str  = self.vocab.get_classification_prompt_vocab(domain)
        known_text_fields = [
            "deviation_description", "violation_description",
            "reason_for_recall", "clinical_summary",
            "why_stopped", "audit_findings",
            # SEC: filings use FILING CONTENT + REGULATORY CONTEXT sections
            "filing_content", "regulatory_context",
            # Device: GMP/QSR violation details
            "device_quality_findings",
        ]
        for field in known_text_fields:
            # Look for this field in the document
            field_text = self._extract_field_text(document, field)
            if field_text:
                preds = self.extractor.classify_text(
                    text         = field_text,
                    field_name   = field,
                    vocabulary   = vocab_str,
                    domain       = domain,
                )
                text_preds.extend(preds)

        # ── Stage 2: Verify + normalize + build FactDict ──────────────────────
        verified_struct, _ = self.verifier.verify(structured_raw, document)
        verified_text,  _ = self.verifier.verify(text_preds, document)

        verified_struct = self.normalizer.normalize_all(verified_struct)
        verified_text   = self.normalizer.normalize_all(verified_text)

        fact_dict = self.builder.build(
            structured_facts     = verified_struct,
            text_predicate_facts = verified_text,
            domain               = domain,
        )

        # ── Stage 3: Symbolic rule engine ─────────────────────────────────────
        verdict = self.engine.build_verdict(fact_dict, domain, doc_hash)

        # ── Stage 4: Seal audit record ────────────────────────────────────────
        audit_record = self.audit.seal(verdict)

        return PipelineResult(
            verdict      = verdict,
            audit_record = audit_record,
            facts        = fact_dict,
        )

    def prepare_training_example(
        self,
        document: str,
        label:    str,
        domain:   str,
    ) -> Optional[Dict[str, Any]]:
        """
        Prepare one labeled document for ILP training.
        Returns sparse facts dict ready for SparseILP.induce().
        """
        # Run extraction + verification
        structured_raw = self.extractor.extract_structured(document, domain)
        verified, _    = self.verifier.verify(structured_raw, document)
        verified       = self.normalizer.normalize_all(verified)

        vocab_str   = self.vocab.get_classification_prompt_vocab(domain)
        text_preds  = []
        for field in ["deviation_description", "reason_for_recall",
                      "violation_description", "why_stopped"]:
            text = self._extract_field_text(document, field)
            if text:
                preds = self.extractor.classify_text(
                    text=text, field_name=field,
                    vocabulary=vocab_str, domain=domain)
                verified_preds, _ = self.verifier.verify(preds, document)
                text_preds.extend(verified_preds)

        verified = self.normalizer.normalize_all(verified)
        text_preds = self.normalizer.normalize_all(text_preds)

        fact_dict = self.builder.build(verified, text_preds, domain)
        sparse    = self.builder.build_ilp_sparse_dict(fact_dict)

        if not sparse:
            return None

        return {
            "facts":  sparse,
            "label":  label,
            "domain": domain,
        }

    def prepare_training_batch(
        self,
        documents: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Prepare a batch of labeled documents for ILP training.
        documents: List of {"document": str, "label": str, "domain": str}
        """
        examples = []
        for i, doc in enumerate(documents):
            ex = self.prepare_training_example(
                doc["document"], doc["label"], doc.get("domain", "gmp"))
            if ex:
                examples.append(ex)
            if (i + 1) % 50 == 0:
                log.info(f"[Training prep] {i+1}/{len(documents)} documents processed")

        log.info(f"[Training prep] {len(examples)}/{len(documents)} "
                 f"examples ready for ILP")
        return examples

    # ── Domain-aware field extraction patterns ────────────────────────────────
    # Maps logical field names to regex patterns that match the actual section
    # headers found in real FDA/SEC/ClinicalTrials documents.
    # Each pattern captures the text following the header.

    _FIELD_PATTERNS: Dict[str, List[str]] = {
        # FDA drug/device enforcement records
        "reason_for_recall": [
            r"REASON FOR RECALL[^:\n]*[:\n]\s*",
            r"REASON FOR VIOLATION[^:\n]*[:\n]\s*",
            r"Reason for Recall[^:\n]*[:\n]\s*",
        ],
        "violation_description": [
            r"VIOLATION DESCRIPTION[^:\n]*[:\n]\s*",
            r"VIOLATION[^:\n]*[:\n]\s*",
            r"FINDINGS?[^:\n]*[:\n]\s*",
            r"OBSERVATION[^:\n]*[:\n]\s*",
        ],
        "deviation_description": [
            r"DEVIATION[^:\n]*[:\n]\s*",
            r"DISCREPANCY[^:\n]*[:\n]\s*",
            r"OUT.OF.SPECIFICATION[^:\n]*[:\n]\s*",
            r"OOS INVESTIGATION[^:\n]*[:\n]\s*",
        ],
        # Clinical trial records
        "why_stopped": [
            r"Reason Stopped[^:\n]*[:\n]\s*",
            r"WHY STOPPED[^:\n]*[:\n]\s*",
            r"Why the Study Was Stopped[^:\n]*[:\n]\s*",
            r"REASON FOR TERMINATION[^:\n]*[:\n]\s*",
        ],
        "clinical_summary": [
            r"BRIEF SUMMARY[^:\n]*[:\n]\s*",
            r"Brief Summary[^:\n]*[:\n]\s*",
            r"STUDY OVERVIEW[^:\n]*[:\n]\s*",
        ],
        "primary_outcome": [
            r"PRIMARY OUTCOME[^:\n]*[:\n]\s*",
            r"Primary Outcome[^:\n]*[:\n]\s*",
            r"PRIMARY ENDPOINT[^:\n]*[:\n]\s*",
        ],
        # SEC filings — key sections where violations appear
        "audit_findings": [
            r"MATERIAL WEAKNESS[^:\n]*[:\n]\s*",
            r"Material Weakness[^:\n]*[:\n]\s*",
            r"INTERNAL CONTROL[^:\n]*[:\n]\s*",
            r"MANAGEMENT.S ASSESSMENT[^:\n]*[:\n]\s*",
        ],
        # SEC 10-K/A — "FILING CONTENT" contains the restatement description
        # "REGULATORY CONTEXT" contains the disclosure failure narrative
        # Neither has a "VIOLATION" header — must match these exact section names
        "filing_content": [
            r"FILING CONTENT[^:\n]*[:\n]\s*",
            r"Filing Content[^:\n]*[:\n]\s*",
        ],
        "regulatory_context": [
            r"REGULATORY CONTEXT[^:\n]*[:\n]\s*",
            r"Regulatory Context[^:\n]*[:\n]\s*",
            r"BACKGROUND[^:\n]*[:\n]\s*",
        ],
        # FDA device enforcement — device QSR / 21 CFR 820 violations
        "device_quality_findings": [
            r"DEVICE DEFECT[^:\n]*[:\n]\s*",
            r"QUALITY SYSTEM[^:\n]*[:\n]\s*",
            r"MANUFACTURING DEFICIENCY[^:\n]*[:\n]\s*",
            r"QSR VIOLATION[^:\n]*[:\n]\s*",
        ],
        # Generic fallback — field name as header (last resort)
        "_generic": [
            r"{field}[^:\n]*[:\n]\s*",
        ],
    }

    def _extract_field_text(
        self,
        document: str,
        field:    str,
    ) -> Optional[str]:
        """
        Extract text for a specific semantic field from a document.

        Uses domain-aware regex patterns that match real document headers
        (e.g. "REASON FOR RECALL / VIOLATION DESCRIPTION:", "Reason Stopped:").
        Falls back to generic field-name matching.

        Returns up to 600 chars of the section text, or None if not found.
        """
        patterns = self._FIELD_PATTERNS.get(field, [])

        # Always try the generic fallback last
        generic_pattern = field.replace("_", r"[\s_]")
        patterns = patterns + [rf"(?i){generic_pattern}[^:\n]*[:\n]\s*"]

        for pattern in patterns:
            try:
                match = re.search(pattern, document, re.IGNORECASE)
                if match:
                    start = match.end()
                    # Extract up to next all-caps section header or 600 chars
                    remainder = document[start:start + 800]
                    # Stop at next section header (all caps line or labelled field)
                    stop = re.search(
                        r"\n[A-Z][A-Z\s/]{5,}[:\n]", remainder)
                    if stop:
                        text = remainder[:stop.start()].strip()
                    else:
                        text = remainder[:600].strip()
                    if text and len(text) >= 15:
                        return text
            except re.error:
                continue

        return None

    def rules_summary(self) -> str:
        return self.engine.rules_summary()

    # ── Rule persistence ──────────────────────────────────────────────────────

    def save_rules(self, path: str) -> None:
        """
        Save all ILP-discovered (learned) rules to disk.
        Allows ILP training results to persist between Kaggle sessions.

        Saves: clause notation, predicates, domain, confidence, provenance.
        Does NOT save compiled_check (regenerated on load from the clause data).
        """
        import json, os
        from dataclasses import asdict

        learned = [
            r for r in self.engine._rules.values()
            if r.rule_source == "learned_ilp"
        ]
        records = []
        for rule in learned:
            records.append({
                "rule_id":         rule.rule_id,
                "name":            rule.name,
                "domain":          rule.domain,
                "severity":        rule.severity,
                "regulation":      rule.regulation,
                "regulation_text": rule.regulation_text,
                "regulation_url":  rule.regulation_url,
                "pass_message":    rule.pass_message,
                "fail_message":    rule.fail_message,
                "predicates":      rule.predicates,
                "confidence":      rule.confidence,
                "approved_by":     rule.approved_by,
                "approved_on":     rule.approved_on,
                "sme_statement":   rule.sme_statement,
                "training_support":rule.training_support,
                "horn_notation":   rule.horn_notation,
            })

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(records, f, indent=2)

        log.info(f"[Pipeline] Saved {len(records)} learned rules → {path}")

    def load_rules(self, path: str) -> int:
        """
        Load previously saved ILP rules back into the engine.
        Recompiles compiled_check from the saved predicate list.
        Returns number of rules loaded.
        """
        import json
        from .models import ComplianceRule
        from .rules.engine import RuleCompiler

        if not __import__("os").path.exists(path):
            log.warning(f"[Pipeline] Rules file not found: {path}")
            return 0

        with open(path) as f:
            records = json.load(f)

        compiler = RuleCompiler(self.vocab)
        n_loaded = 0

        for rec in records:
            # Reconstruct HornClause-like data to recompile check function
            from .models import HornClause
            predicates = rec["predicates"]
            head = "non_compliant" if "non_compliant" in rec["name"] else "compliant"

            is_noncompliant = (head == "non_compliant")
            check_fn = RuleCompiler._make_check(predicates, is_noncompliant)

            rule = ComplianceRule(
                rule_id         = rec["rule_id"],
                name            = rec["name"],
                domain          = rec["domain"],
                severity        = rec["severity"],
                rule_source     = "learned_ilp",
                regulation      = rec["regulation"],
                regulation_text = rec["regulation_text"],
                regulation_url  = rec["regulation_url"],
                pass_message    = rec["pass_message"],
                fail_message    = rec["fail_message"],
                predicates      = predicates,
                confidence      = rec["confidence"],
                approved_by     = rec.get("approved_by", ""),
                approved_on     = rec.get("approved_on", ""),
                sme_statement   = rec.get("sme_statement", ""),
                training_support= rec.get("training_support", 0),
                horn_notation   = rec.get("horn_notation", ""),
            )
            rule.compiled_check = check_fn
            self.engine.add_rule(rule)
            n_loaded += 1

        log.info(f"[Pipeline] Loaded {n_loaded} learned rules from {path}")
        return n_loaded
