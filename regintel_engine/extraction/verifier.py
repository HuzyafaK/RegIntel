"""
regintel_engine.extraction.verifier
=====================================
Hallucination verification, normalization, and fact building.

Three components:
  HallucinationVerifier  checks every extracted fact is grounded in document
  FactNormalizer         maps document language to standard vocabulary names
  FactDictBuilder        assembles verified facts into FactDict for engine
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from ..models import ExtractedFact, FactDict
from ..vocabulary import EncodingTables, PredicateVocabulary

log = logging.getLogger("regintel.verifier")


# ─────────────────────────────────────────────────────────────────────────────
#  HALLUCINATION VERIFIER
# ─────────────────────────────────────────────────────────────────────────────

class HallucinationVerifier:
    """
    Verifies every extracted fact is grounded in the source document.

    Three checks:
    1. extracted_text exists verbatim in document
    2. numeric value appears in extracted_text
    3. boolean facts have explicit PASS/FAIL signal in extracted_text

    Result buckets:
      confidence >= 0.90, no flags → accept
      confidence 0.60-0.90          → accept with flag
      confidence < 0.60             → reject (treat as absent)
      not grounded in document      → reject (hallucinated)

    CRITICAL: unverified facts never reach the symbolic engine.
    A hallucinated fact could cause a wrong verdict.
    """

    # Explicit signals for boolean pass/fail
    PASS_SIGNALS = {"pass", "compliant", "within specification", "within spec",
                    "cleared", "approved", "current", "valid", "satisfactory",
                    "adequate", "effective", "met", "achieved"}

    FAIL_SIGNALS = {"fail", "non-compliant", "out of specification", "oos",
                    "overdue", "expired", "absent", "missing", "not found",
                    "not received", "not on file", "not documented",
                    "violation", "recalled", "terminated", "inadequate"}

    CONFIDENCE_ACCEPT  = 0.90
    CONFIDENCE_FLAG    = 0.60

    def verify(
        self,
        facts:    List[ExtractedFact],
        document: str,
    ) -> Tuple[List[ExtractedFact], List[str]]:
        """
        Verify facts against document.
        Returns (verified_facts, rejection_log).
        verified_facts may have reduced confidence if checks fail.
        """
        doc_lower  = document.lower()
        accepted   = []
        rejected   = []

        for fact in facts:
            result, reason = self._check(fact, document, doc_lower)
            if result == "accept":
                accepted.append(fact)
            elif result == "flag":
                fact.confidence = max(fact.confidence * 0.7, self.CONFIDENCE_FLAG)
                accepted.append(fact)
            else:
                rejected.append(f"{fact.name}: {reason}")
                log.debug(f"[VERIFIER] Rejected {fact.name}: {reason}")

        return accepted, rejected

    def _check(
        self,
        fact:      ExtractedFact,
        document:  str,
        doc_lower: str,
    ) -> Tuple[str, str]:  # (accept|flag|reject, reason)
        """Run all verification checks on one fact."""
        extracted = fact.extracted_text.strip()

        # Check 1: extracted_text must exist in document
        if extracted and extracted.lower() not in doc_lower:
            # Try partial match (first 50 chars)
            if len(extracted) > 50:
                partial = extracted[:50].lower()
                if partial not in doc_lower:
                    return "reject", "extracted_text not found in document (hallucinated)"
            else:
                return "reject", "extracted_text not found in document (hallucinated)"

        # Check 2: numeric values must appear in extracted_text
        if isinstance(fact.value, (int, float)):
            val_str = str(fact.value)
            if val_str not in extracted and val_str not in document[:5000]:
                return "flag", f"numeric value {val_str} not found in extracted_text"

        # Check 3: boolean facts need explicit signal
        if isinstance(fact.value, bool) and extracted:
            text_lower = extracted.lower()
            has_pass   = any(sig in text_lower for sig in self.PASS_SIGNALS)
            has_fail   = any(sig in text_lower for sig in self.FAIL_SIGNALS)

            if fact.value is True and not has_pass:
                return "flag", "boolean True but no explicit PASS signal in text"
            if fact.value is False and not has_fail:
                return "flag", "boolean False but no explicit FAIL signal in text"

        # Check 4: confidence threshold
        if fact.confidence < self.CONFIDENCE_FLAG:
            return "reject", f"confidence {fact.confidence:.2f} below threshold"

        return "accept", "ok"


# ─────────────────────────────────────────────────────────────────────────────
#  FACT NORMALIZER
# ─────────────────────────────────────────────────────────────────────────────

class FactNormalizer:
    """
    Maps document language variations to standard vocabulary names,
    including positive/negative framing inversion.

    The LLM extraction prompt returns positively-framed booleans
    (e.g. calibration_current=True/False) while the controlled vocabulary
    uses violation-signal names (calibration_expired=True/False).

    Two types of mapping are handled:
    1. ALIASES: name-only rename (same polarity)
       coa_reviewed → api_coa_reviewed
    2. INVERTED: name rename + value inversion (polarity flip)
       calibration_current=True  → calibration_expired=False
       calibration_current=False → calibration_expired=True
    """

    # Direct aliases: document term → standard vocabulary name (same polarity)
    ALIASES: Dict[str, str] = {
        "coa_reviewed":                       "api_coa_reviewed",
        "certificate_of_analysis_reviewed":   "api_coa_reviewed",
        "raw_material_released":              "api_coa_reviewed",
        "api_released":                       "api_coa_reviewed",
        "equipment_calibration_current":      "calibration_current",
        "calibration_valid":                  "calibration_current",
        "instrument_calibration_current":     "calibration_current",
        "process_parameters_in_spec":         "process_in_spec",
        "manufacturing_in_spec":              "process_in_spec",
        "critical_process_parameters_met":    "process_in_spec",
        "sample_size":                        "clinical_n",
        "enrollment_count":                   "clinical_n",
        "subjects_enrolled":                  "clinical_n",
        "n_subjects":                         "clinical_n",
        "participants":                       "clinical_n",
        "predicate_k_number":                 "predicate_device",
        "substantially_equivalent_predicate": "predicate_device",
        "legally_marketed_predicate":         "predicate_device",
        "iso_14971_applied":                  "risk_management_standard",
        "risk_management_documented":         "risk_management_standard",
        "risk_file_present":                  "risk_management_standard",
    }

    # Inverted aliases: document name → (vocabulary name, invert_value)
    # When invert_value=True, the boolean is flipped before storing.
    # Used when LLM uses positive assertions but vocab uses violation signals.
    INVERTED: Dict[str, tuple] = {
        # calibration_current=False → calibration_expired=True
        "calibration_current":      ("calibration_expired", True),
        # api_coa_reviewed=True → api_coa_missing=False
        "api_coa_reviewed":         ("api_coa_missing",     True),
        # process_in_spec=False → oos_result=True
        "process_in_spec":          ("oos_result",          True),
        # dissolution_pass=False → oos_result=True
        "dissolution_pass":         ("oos_result",          True),
        # sterility_tested=True → sterility_assurance_failure=False
        "sterility_tested":         ("sterility_assurance_failure", True),
    }

    def normalize_name(self, name: str) -> str:
        """Map fact name to standard vocabulary name (aliases only)."""
        # Direct alias lookup
        if name in self.ALIASES:
            return self.ALIASES[name]
        # Snake-case normalise then lookup
        norm = re.sub(r"[^a-z0-9_]", "_", name.lower())
        norm = re.sub(r"_+", "_", norm).strip("_")
        return self.ALIASES.get(norm, norm)

    def normalize_all(self, facts: List[ExtractedFact]) -> List[ExtractedFact]:
        """
        Normalize all fact names and apply value inversions where needed.

        After inversion, `calibration_current=False` becomes
        `calibration_expired=True` — the vocabulary predicate name
        with the correct polarity.
        """
        result = []
        for fact in facts:
            # Check for inversion mapping first
            if fact.name in self.INVERTED:
                vocab_name, should_invert = self.INVERTED[fact.name]
                if should_invert and isinstance(fact.value, bool):
                    fact.value = not fact.value
                fact.name = vocab_name
            else:
                # Plain alias
                fact.name = self.normalize_name(fact.name)
            result.append(fact)
        return result


# ─────────────────────────────────────────────────────────────────────────────
#  FACT DICT BUILDER
# ─────────────────────────────────────────────────────────────────────────────

class FactDictBuilder:
    """
    Assembles verified, normalized ExtractedFacts into a FactDict
    that the symbolic rule engine can evaluate.

    INVARIANT: no neural floats, no embeddings, no probabilities in FactDict.
    Only discrete values with full provenance.

    vocab_names: set of controlled vocabulary predicate names.
    Only predicates in this set are included in the ILP sparse dict.
    This prevents noise facts (brand_name, sponsor, etc.) from
    contaminating ILP training with spurious correlations.
    """

    def __init__(
        self,
        encoding_tables: Optional[EncodingTables] = None,
        vocab_names:     Optional[set] = None,
    ) -> None:
        self.encoding    = encoding_tables or EncodingTables()
        self.vocab_names = vocab_names     # set of controlled predicate names

    def build(
        self,
        structured_facts:    List[ExtractedFact],
        text_predicate_facts: List[ExtractedFact],
        domain:              str,
    ) -> FactDict:
        """
        Build a FactDict from verified structured and text predicate facts.

        Structured facts: keep original discrete values (bool, str, int)
        Text predicate facts: already boolean from classifier
        Both: low-confidence facts go to absent_facts list
        """
        facts_dict = {}
        provenance = {}
        absent     = []

        all_facts = structured_facts + text_predicate_facts

        for fact in all_facts:
            if fact.low_confidence:
                # Uncertain — treat as absent (→ None → NEEDS_REVIEW not NON_COMPLIANT)
                absent.append(fact.name)
                continue

            # Avoid overwriting higher-confidence facts
            if fact.name in facts_dict:
                existing_conf = provenance[fact.name].confidence
                if fact.confidence <= existing_conf:
                    continue

            facts_dict[fact.name] = fact.value
            provenance[fact.name] = fact

        return FactDict(
            facts        = facts_dict,
            provenance   = provenance,
            absent_facts = absent,
            domain       = domain,
        )

    def build_ilp_sparse_dict(
        self,
        fact_dict: FactDict,
    ) -> Dict[str, float]:
        """
        Convert FactDict to sparse float dict for ILP training.

        Only vocabulary predicates are included — this prevents noise
        facts (brand_name, sponsor, application_number, etc.) from
        appearing in ILP and creating spurious rules.

        Absent facts NOT included (not padded with 0.5).
        ILP mask: present = scored, absent = excluded from this document.
        """
        sparse: Dict[str, float] = {}

        for name, value in fact_dict.facts.items():
            # Skip any fact that is not in the controlled vocabulary
            if self.vocab_names is not None and name not in self.vocab_names:
                continue

            encoded = self.encoding.encode(name, value)
            if encoded is not None:
                sparse[name] = encoded

        return sparse
