"""
regintel_engine.models
======================
Core data structures for the RegIntel compliance engine.

Design principles
-----------------
1. Strict boundary between neural and symbolic layers
   - ExtractedFact carries both a discrete value AND its provenance
   - FactDict is what the symbolic engine sees — discrete only
   - No floats, embeddings, or probabilities cross into FactDict

2. Every decision is traceable
   - ExtractedFact → document line (extracted_text)
   - RuleResult → rule_id + evidence predicates
   - Verdict → rule_results + audit record

3. Three-valued logic for symbolic evaluation
   - True  = passed (positive evidence present and correct)
   - False = confirmed violation (evidence present, condition failed)
   - None  = uncertain (fact absent or low confidence)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, ClassVar, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
#  EXTRACTED FACT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExtractedFact:
    """
    One fact extracted from a document by the LLM.

    Fields
    ------
    name           Standard vocabulary name (e.g. 'calibration_current')
    value          Discrete value: bool | str | int | float | None
    extracted_text Exact sentence from document supporting this fact
    confidence     0.0–1.0 extraction confidence
    reasoning      One-sentence LLM explanation
    source_type    'structured' | 'text_predicate' | 'numeric'
    document_line  Approximate line number in source document
    low_confidence True if confidence < 0.60 (flagged, treated as absent)
    """
    name:           str
    value:          Any                   # bool | str | int | float | None
    extracted_text: str
    confidence:     float
    reasoning:      str                   = ""
    source_type:    str                   = "structured"
    document_line:  int                   = 0
    low_confidence: bool                  = False

    def __post_init__(self) -> None:
        self.low_confidence = self.confidence < 0.60


@dataclass
class FactDict:
    """
    The discrete facts dict that the symbolic rule engine receives.

    INVARIANT: no neural floats, no embeddings, no probabilities.
    Only discrete values with provenance.

    absent_facts contains names of facts not found or below confidence
    threshold — these return None from get(), causing rules to return
    None (NEEDS_REVIEW) rather than False (NON_COMPLIANT).
    """
    facts:          Dict[str, Any]             # name → discrete value
    provenance:     Dict[str, ExtractedFact]   # name → original fact
    absent_facts:   List[str]                  = field(default_factory=list)
    domain:         str                        = "gmp"

    def get(self, name: str, default: Any = None) -> Any:
        """Get fact value. Returns None (not default) for absent facts."""
        if name in self.absent_facts:
            return None    # uncertain — drives NEEDS_REVIEW not NON_COMPLIANT
        return self.facts.get(name, default)

    def get_evidence(self, name: str) -> Optional[str]:
        """Get the extracted_text that supports this fact."""
        prov = self.provenance.get(name)
        return prov.extracted_text if prov else None

    def get_confidence(self, name: str) -> float:
        prov = self.provenance.get(name)
        return prov.confidence if prov else 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  RULE RESULT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RuleResult:
    """
    Result of evaluating one compliance rule against a FactDict.

    passed values (three-valued logic):
      True  = rule passed (all conditions met)
      False = rule failed (confirmed violation)
      None  = uncertain (required facts absent or low confidence)
    """
    rule_id:          str
    rule_name:        str
    domain:           str
    severity:         str              # 'critical' | 'major' | 'minor'
    passed:           Optional[bool]   # True | False | None
    message:          str              # human-readable explanation
    regulation:       str              # e.g. "21 CFR 211.84"
    regulation_text:  str              # actual regulatory text
    regulation_url:   str              # ecfr.gov link
    evidence:         Dict[str, Any]   # predicate → {value, extracted_text}
    rule_source:      str              # 'base' | 'learned_ilp'
    confidence:       float            = 1.0   # rule confidence (ILP rules)
    approved_by:      str              = ""    # SME who approved (ILP rules)
    approved_on:      str              = ""


# ─────────────────────────────────────────────────────────────────────────────
#  VERDICT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Verdict:
    """
    Final compliance verdict for one document.

    Three possible values:
      COMPLIANT      all critical rules passed
      NON_COMPLIANT  at least one critical rule returned False
      NEEDS_REVIEW   at least one critical rule returned None
                     (uncertainty — escalate to human)
    """
    verdict:        str                   # COMPLIANT | NON_COMPLIANT | NEEDS_REVIEW
    domain:         str
    rule_results:   List[RuleResult]
    facts:          FactDict
    doc_hash:       str
    timestamp:      str
    system_version: str                   = "regintel-1.0.0"

    @property
    def is_compliant(self) -> bool:
        return self.verdict == "COMPLIANT"

    @property
    def failed_rules(self) -> List[RuleResult]:
        return [r for r in self.rule_results if r.passed is False]

    @property
    def uncertain_rules(self) -> List[RuleResult]:
        return [r for r in self.rule_results if r.passed is None]

    @property
    def primary_violation(self) -> Optional[RuleResult]:
        """The first critical rule that failed."""
        for r in self.rule_results:
            if r.passed is False and r.severity == "critical":
                return r
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  COMPLIANCE RULE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ComplianceRule:
    """
    One compliance rule in the symbolic engine.

    compiled_check: pure Python function stored as a regular attribute
      (not a dataclass field) so the rule is fully serialisable with asdict().
      Input:  FactDict
      Output: Optional[bool]  (True=pass, False=fail, None=uncertain)

    Citations come from the predicate vocabulary — never LLM generated.
    """
    rule_id:          str
    name:             str
    domain:           str
    severity:         str              # 'critical' | 'major' | 'minor'
    rule_source:      str              # 'base' | 'learned_ilp'
    regulation:       str
    regulation_text:  str
    regulation_url:   str
    pass_message:     str
    fail_message:     str
    predicates:       List[str]        # predicate names this rule checks
    confidence:       float            = 1.0
    approved_by:      str              = ""
    approved_on:      str              = ""
    sme_statement:    str              = ""
    training_support: int              = 0
    horn_notation:    str              = ""

    # Not a dataclass field — excluded from asdict() and __init__ signature.
    # Set immediately after construction via rule.compiled_check = fn.
    compiled_check: ClassVar[Optional[Callable]] = None  # placeholder for type checkers

    def __post_init__(self) -> None:
        # compiled_check is set by the constructor helper or rule compiler,
        # not via dataclass __init__. Initialise to a safe default.
        if not hasattr(self, "_compiled_check"):
            object.__setattr__(self, "_compiled_check", None)

    @property
    def compiled_check(self) -> Optional[Callable]:  # type: ignore[override]
        return object.__getattribute__(self, "_compiled_check")

    @compiled_check.setter
    def compiled_check(self, fn: Callable) -> None:
        object.__setattr__(self, "_compiled_check", fn)

    def evaluate(self, facts: FactDict) -> RuleResult:
        """Evaluate this rule against a FactDict."""
        passed = self.compiled_check(facts)
        # Three-valued: True=pass, False=fail, None=uncertain
        # None is falsy in Python — must check explicitly to avoid
        # showing fail_message when evidence is simply absent.
        if passed is True:
            message = self.pass_message
        elif passed is False:
            message = self.fail_message
        else:
            message = (f"UNCERTAIN — required evidence for '{self.name}' "
                       f"not found in document")

        # Collect evidence for predicates this rule checked
        evidence = {}
        for pred in self.predicates:
            evidence[pred] = {
                "value":          facts.get(pred),
                "extracted_text": facts.get_evidence(pred) or "",
                "confidence":     facts.get_confidence(pred),
            }

        return RuleResult(
            rule_id         = self.rule_id,
            rule_name       = self.name,
            domain          = self.domain,
            severity        = self.severity,
            passed          = passed,
            message         = message,
            regulation      = self.regulation,
            regulation_text = self.regulation_text,
            regulation_url  = self.regulation_url,
            evidence        = evidence,
            rule_source     = self.rule_source,
            confidence      = self.confidence,
            approved_by     = self.approved_by,
            approved_on     = self.approved_on,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  ILP HORN CLAUSE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HornClause:
    """
    A Horn clause discovered by the ILP system.
    Before SME approval — not yet compiled to ComplianceRule.

    head:    'compliant' | 'non_compliant'
    body:    list of predicate names from controlled vocabulary
    status:  'pending' | 'approved' | 'rejected' | 'modified'
    """
    clause_id:  str
    head:       str               # 'compliant' | 'non_compliant'
    body:       List[str]         # predicate names
    confidence: float
    coverage:   float
    support:    int               # N training documents
    domain:     str
    status:     str               = "pending"
    notation:   str               = ""

    def __post_init__(self) -> None:
        if not self.notation:
            body_str    = " AND ".join(f"{p} = True" for p in self.body)
            self.notation = f"{self.head} :- {body_str}"

    @property
    def is_approvable(self) -> bool:
        """Only pending clauses with high confidence and named predicates."""
        if self.status != "pending":          # already approved/rejected → no
            return False
        if self.confidence < 0.80:
            return False
        if any(p.startswith("pred_") for p in self.body):
            return False
        return True


# ─────────────────────────────────────────────────────────────────────────────
#  AUDIT RECORD
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AuditRecord:
    """
    One tamper-evident audit record.
    Every field is included in the HMAC signature.
    """
    record_id:       str
    doc_hash:        str           # SHA-256 of original document
    facts_hash:      str           # SHA-256 of serialised FactDict
    verdict:         str
    domain:          str
    timestamp:       str
    system_version:  str

    # Full evidence chain
    extracted_facts: List[Dict]    # serialised ExtractedFacts
    rule_results:    List[Dict]    # serialised RuleResults

    # Chain integrity
    prev_hash:       str           # hash of previous AuditRecord
    record_hash:     str           # SHA-256 of this record (without signature)
    signature:       str           # HMAC-SHA256

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)
