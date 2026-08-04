"""
regintel_engine.rules
======================
Base rules, rule compiler, and symbolic rule engine.

Three components:

1. BASE_RULES
   Hand-written by SME. Cover known regulatory requirements.
   Always present regardless of training data.
   Never discovered by ILP — written by regulatory experts.

2. RuleCompiler
   Converts HornClause → ComplianceRule (compiled Python function).
   The compiled_check function is pure Python:
     Input:  FactDict
     Output: Optional[bool]  (True=pass, False=fail, None=uncertain)
   Zero neural involvement.

3. SymbolicRuleEngine
   Evaluates all rules against a FactDict.
   Three-valued logic:
     True  = passed
     False = confirmed violation → may trigger NON_COMPLIANT
     None  = uncertain → NEEDS_REVIEW
   Verdict logic:
     any critical rule = False → NON_COMPLIANT
     any critical rule = None  → NEEDS_REVIEW
     all rules = True          → COMPLIANT
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..models import (
    ComplianceRule, FactDict, HornClause, RuleResult, Verdict,
)
from ..vocabulary import PredicateVocabulary

log = logging.getLogger("regintel.rules")


# ─────────────────────────────────────────────────────────────────────────────
#  BASE RULES
# ─────────────────────────────────────────────────────────────────────────────

def _make_gmp_rules(vocab: PredicateVocabulary) -> List[ComplianceRule]:
    """Hand-written GMP compliance rules based on 21 CFR Part 211."""

    rules = []

    def r(rule_id, name, severity, predicates, check_fn,
          pass_msg, fail_msg, reg, reg_text, reg_url):
        rule = ComplianceRule(
            rule_id         = rule_id,
            name            = name,
            domain          = "gmp",
            severity        = severity,
            rule_source     = "base",
            regulation      = reg,
            regulation_text = reg_text,
            regulation_url  = reg_url,
            pass_message    = pass_msg,
            fail_message    = fail_msg,
            predicates      = predicates,
        )
        rule.compiled_check = check_fn
        return rule

    # GMP-001: CoA must not be missing
    rule = ComplianceRule(
        rule_id         = "GMP-001",
        name            = "Raw material certificate of analysis verification",
        domain          = "gmp",
        severity        = "critical",
        rule_source     = "base",
        regulation      = "21 CFR 211.84(d)(1)",
        regulation_text = "Each lot of components shall be withheld from use until sampled, tested and released by the quality control unit.",
        regulation_url  = "https://www.ecfr.gov/current/title-21/section-211.84",
        pass_message    = "API CoA reviewed and approved prior to manufacturing",
        fail_message    = "API Certificate of Analysis missing — batch cannot be released",
        predicates      = ["api_coa_missing"],
    )
    # api_coa_missing=False → CoA was present → pass
    # api_coa_missing=True  → CoA was missing → fail
    # api_coa_missing=None  → unknown         → uncertain
    rule.compiled_check = lambda f: (
        None if f.get("api_coa_missing") is None
        else not bool(f.get("api_coa_missing"))
    )
    rules.append(rule)

    # GMP-002: Equipment calibration must not be expired
    rule = ComplianceRule(
        rule_id         = "GMP-002",
        name            = "Equipment calibration current",
        domain          = "gmp",
        severity        = "major",
        rule_source     = "base",
        regulation      = "21 CFR 211.68(a)",
        regulation_text = "Equipment shall be routinely calibrated, inspected, or checked according to a written program.",
        regulation_url  = "https://www.ecfr.gov/current/title-21/section-211.68",
        pass_message    = "All equipment calibration is current",
        fail_message    = "Equipment calibration overdue — process data may be unreliable",
        predicates      = ["calibration_expired"],
    )
    rule.compiled_check = lambda f: (
        None if f.get("calibration_expired") is None
        else not bool(f.get("calibration_expired"))
    )
    rules.append(rule)

    # GMP-003: No out-of-specification results
    rule = ComplianceRule(
        rule_id         = "GMP-003",
        name            = "Critical process parameters within specification",
        domain          = "gmp",
        severity        = "critical",
        rule_source     = "base",
        regulation      = "21 CFR 211.100(a)",
        regulation_text = "Written procedures for production and process controls shall be established.",
        regulation_url  = "https://www.ecfr.gov/current/title-21/section-211.100",
        pass_message    = "All critical process parameters within validated specification",
        fail_message    = "Out-of-specification result detected — batch integrity compromised",
        predicates      = ["oos_result"],
    )
    rule.compiled_check = lambda f: (
        None if f.get("oos_result") is None
        else not bool(f.get("oos_result"))
    )
    rules.append(rule)

    # GMP-004: Deviation without CAPA
    def check_deviation(f: FactDict) -> Optional[bool]:
        capa_missing = f.get("deviation_without_capa")
        if capa_missing is None:
            return None
        return not bool(capa_missing)   # passes if NO deviation-without-capa

    rule = ComplianceRule(
        rule_id         = "GMP-004",
        name            = "All deviations have corrective action",
        domain          = "gmp",
        severity        = "critical",
        rule_source     = "base",
        regulation      = "21 CFR 211.192",
        regulation_text = "Any unexplained discrepancy shall be thoroughly investigated.",
        regulation_url  = "https://www.ecfr.gov/current/title-21/section-211.192",
        pass_message    = "All manufacturing deviations have documented corrective actions",
        fail_message    = "Deviation recorded without CAPA — unresolved quality failure",
        predicates      = ["deviation_without_capa"],
    )
    rule.compiled_check = check_deviation
    rules.append(rule)

    # GMP-005: No contamination
    def check_contamination(f: FactDict) -> Optional[bool]:
        contamination = f.get("contamination_detected")
        if contamination is None:
            return None
        return not bool(contamination)

    rule = ComplianceRule(
        rule_id         = "GMP-005",
        name            = "No contamination detected in batch",
        domain          = "gmp",
        severity        = "critical",
        rule_source     = "base",
        regulation      = "21 CFR 211.42",
        regulation_text = "There shall be separate or defined areas to prevent contamination.",
        regulation_url  = "https://www.ecfr.gov/current/title-21/section-211.42",
        pass_message    = "No contamination detected in product",
        fail_message    = "Contamination detected — batch must be quarantined and investigated",
        predicates      = ["contamination_detected"],
    )
    rule.compiled_check = check_contamination
    rules.append(rule)

    # GMP-006: Sterility assurance
    # "Microbial Contamination of Non-Sterile Products" → 21 CFR 211.113(b)
    # This is the most common recall reason for sterile/non-sterile products.
    def check_sterility(f: FactDict) -> Optional[bool]:
        failure = f.get("sterility_assurance_failure")
        if failure is True:
            return False
        if failure is False:
            return True
        return None

    rule = ComplianceRule(
        rule_id         = "GMP-006",
        name            = "Sterility assurance — no microbiological contamination",
        domain          = "gmp",
        severity        = "critical",
        rule_source     = "base",
        regulation      = "21 CFR 211.113(b)",
        regulation_text = "Appropriate written procedures shall be established and followed that are designed to prevent microbiological contamination of drug products purporting to be sterile.",
        regulation_url  = "https://www.ecfr.gov/current/title-21/section-211.113",
        pass_message    = "Sterility assurance procedures met — no microbiological contamination",
        fail_message    = "Sterility assurance failure — microbial contamination of non-sterile product",
        predicates      = ["sterility_assurance_failure"],
    )
    rule.compiled_check = check_sterility
    rules.append(rule)

    # GMP-007: Labeling compliance
    # Real dataset GT: labeling_noncompliant (11 training docs)
    rule = ComplianceRule(
        rule_id         = "GMP-007",
        name            = "Drug labeling accurate and complete",
        domain          = "gmp",
        severity        = "critical",
        rule_source     = "base",
        regulation      = "21 CFR 211.125",
        regulation_text = "Labeling issuance shall include careful examination for identity and conformity to the labeling specified in the master or batch production records.",
        regulation_url  = "https://www.ecfr.gov/current/title-21/section-211.125",
        pass_message    = "Labeling meets all regulatory requirements",
        fail_message    = "Labeling non-compliant — incorrect, missing, or mislabeled product",
        predicates      = ["labeling_noncompliant"],
    )
    rule.compiled_check = lambda f: (
        None if f.get("labeling_noncompliant") is None
        else not bool(f.get("labeling_noncompliant"))
    )
    rules.append(rule)

    # GMP-008: Stability testing
    # Real dataset GT: stability_not_confirmed (11 training docs)
    rule = ComplianceRule(
        rule_id         = "GMP-008",
        name            = "Drug stability testing completed and confirmed",
        domain          = "gmp",
        severity        = "major",
        rule_source     = "base",
        regulation      = "21 CFR 211.166(a)",
        regulation_text = "There shall be a written testing program designed to assess the stability characteristics of drug products.",
        regulation_url  = "https://www.ecfr.gov/current/title-21/section-211.166",
        pass_message    = "Stability testing completed and expiration date confirmed",
        fail_message    = "Stability not confirmed — product expiry or storage conditions questionable",
        predicates      = ["stability_not_confirmed"],
    )
    rule.compiled_check = lambda f: (
        None if f.get("stability_not_confirmed") is None
        else not bool(f.get("stability_not_confirmed"))
    )
    rules.append(rule)

    # GMP-009: General GMP violation (catch-all)
    # Real dataset GT: gmp_violation (111 training docs — most common GMP NC predicate)
    # FDA enforcement reports often cite "21 CFR Part 211" without a specific subsection.
    # This rule catches those general violations AND uses gmp_compliant/drug_approved
    # as positive compliance signals for COMPLIANT docs.
    def check_gmp_violation(f: FactDict) -> Optional[bool]:
        violation = f.get("gmp_violation")
        if violation is True:
            return False
        # Positive compliance signals
        if f.get("gmp_compliant") is True or f.get("drug_approved") is True:
            return True
        if violation is False:
            return True
        return None

    rule = ComplianceRule(
        rule_id         = "GMP-009",
        name            = "No general GMP violation (21 CFR Part 211)",
        domain          = "gmp",
        severity        = "critical",
        rule_source     = "base",
        regulation      = "21 CFR Part 211",
        regulation_text = "The Current Good Manufacturing Practice regulations contain the minimum current good manufacturing practice for the manufacture, processing, packing, or holding of a drug product.",
        regulation_url  = "https://www.ecfr.gov/current/title-21/part-211",
        pass_message    = "No GMP violations — drug product meets 21 CFR Part 211",
        fail_message    = "GMP violation identified — 21 CFR Part 211 non-compliance",
        predicates      = ["gmp_violation", "gmp_compliant", "drug_approved"],
    )
    rule.compiled_check = check_gmp_violation
    rules.append(rule)

    # GMP-010: Dissolution testing
    # Real dataset GT: dissolution_failed (15 training docs)
    rule = ComplianceRule(
        rule_id         = "GMP-010",
        name            = "Dissolution testing meets specification",
        domain          = "gmp",
        severity        = "critical",
        rule_source     = "base",
        regulation      = "21 CFR 211.165(b)",
        regulation_text = "Drug products shall be tested to ensure conformance with specifications for identity, strength, quality, and purity.",
        regulation_url  = "https://www.ecfr.gov/current/title-21/section-211.165",
        pass_message    = "Dissolution testing within specification",
        fail_message    = "Dissolution test failed — drug product does not meet release specification",
        predicates      = ["dissolution_failed"],
    )
    rule.compiled_check = lambda f: (
        None if f.get("dissolution_failed") is None
        else not bool(f.get("dissolution_failed"))
    )
    rules.append(rule)

    # GMP-011: Production and process controls
    # Real dataset GT: production_process_controls_absent (8 training docs)
    rule = ComplianceRule(
        rule_id         = "GMP-011",
        name            = "Production and process controls adequate",
        domain          = "gmp",
        severity        = "critical",
        rule_source     = "base",
        regulation      = "21 CFR 211.100(b)",
        regulation_text = "Written production and process control procedures shall be followed in the execution of the various production and process control functions.",
        regulation_url  = "https://www.ecfr.gov/current/title-21/section-211.100",
        pass_message    = "Production and process controls documented and followed",
        fail_message    = "Production or process controls absent — critical GMP failure",
        predicates      = ["production_process_controls_absent"],
    )
    rule.compiled_check = lambda f: (
        None if f.get("production_process_controls_absent") is None
        else not bool(f.get("production_process_controls_absent"))
    )
    rules.append(rule)

    return rules


def _make_device_rules(vocab: PredicateVocabulary) -> List[ComplianceRule]:
    """Hand-written FDA 510k rules based on 21 CFR 807 and ISO standards."""

    rules = []

    # DEV-001: Predicate device required for 510k
    def check_predicate(f: FactDict) -> Optional[bool]:
        pred = f.get("predicate_device")
        if pred is None:
            return None
        return bool(pred)

    rule = ComplianceRule(
        rule_id         = "DEV-001",
        name            = "Predicate device referenced in 510k",
        domain          = "fda_device",
        severity        = "critical",
        rule_source     = "base",
        regulation      = "21 CFR 807.87(f)",
        regulation_text = "A 510(k) shall contain a description of how the device compares to the predicate device.",
        regulation_url  = "https://www.ecfr.gov/current/title-21/section-807.87",
        pass_message    = "Predicate device identified",
        fail_message    = "No predicate device — substantial equivalence cannot be demonstrated",
        predicates      = ["predicate_device"],
    )
    rule.compiled_check = check_predicate
    rules.append(rule)

    # DEV-002: Risk management required
    def check_risk_mgmt(f: FactDict) -> Optional[bool]:
        risk = f.get("risk_management_standard")
        absent = f.get("risk_management_absent")

        if absent is True:
            return False
        if risk is not None and "14971" in str(risk):
            return True
        if risk is None and absent is None:
            return None
        return None

    rule = ComplianceRule(
        rule_id         = "DEV-002",
        name            = "Risk management per ISO 14971",
        domain          = "fda_device",
        severity        = "major",
        rule_source     = "base",
        regulation      = "ISO 14971:2019",
        regulation_text = "The manufacturer shall establish a process for risk management.",
        regulation_url  = "https://www.iso.org/standard/72704.html",
        pass_message    = "ISO 14971 risk management documented",
        fail_message    = "Risk management absent or not per ISO 14971",
        predicates      = ["risk_management_standard", "risk_management_absent"],
    )
    rule.compiled_check = check_risk_mgmt
    rules.append(rule)

    # DEV-003: Clinical evidence for Class III
    def check_clinical(f: FactDict) -> Optional[bool]:
        device_class    = f.get("device_class")
        clinical_absent = f.get("clinical_evidence_absent")
        clinical_n      = f.get("clinical_n")

        if device_class is None:
            return None   # cannot assess without device class
        if device_class == "III":
            if clinical_absent is True:
                return False
            if clinical_n is not None and clinical_n < 30:
                return False
            if clinical_absent is False or (clinical_n and clinical_n >= 30):
                return True
            return None   # uncertain
        return True   # Class I/II — may not need clinical

    rule = ComplianceRule(
        rule_id         = "DEV-003",
        name            = "Clinical evidence adequate for device class",
        domain          = "fda_device",
        severity        = "critical",
        rule_source     = "base",
        regulation      = "21 CFR 860.7",
        regulation_text = "Valid scientific evidence is evidence from well-controlled investigations.",
        regulation_url  = "https://www.ecfr.gov/current/title-21/section-860.7",
        pass_message    = "Clinical evidence adequate for device classification",
        fail_message    = "Clinical evidence absent or inadequate for Class III device",
        predicates      = ["device_class", "clinical_evidence_absent", "clinical_n"],
    )
    rule.compiled_check = check_clinical
    rules.append(rule)

    # DEV-004: No device QSR / Part 820 violation
    # Dataset GT predicate: device_gmp_violation (FDA device enforcement reports)
    # Covers post-market enforcement for 21 CFR Part 820 manufacturing violations.
    def check_device_gmp(f: FactDict) -> Optional[bool]:
        violation = f.get("device_gmp_violation")
        if violation is True:
            return False
        if violation is False:
            return True
        return None

    rule = ComplianceRule(
        rule_id         = "DEV-004",
        name            = "No device Quality System Regulation (QSR) violation",
        domain          = "fda_device",
        severity        = "critical",
        rule_source     = "base",
        regulation      = "21 CFR Part 820",
        regulation_text = "Each manufacturer of a device shall establish and maintain a quality system appropriate for the specific device(s) designed or manufactured.",
        regulation_url  = "https://www.ecfr.gov/current/title-21/part-820",
        pass_message    = "Device quality system meets 21 CFR Part 820 requirements",
        fail_message    = "Device QSR violation — 21 CFR Part 820 non-compliance identified",
        predicates      = ["device_gmp_violation"],
    )
    rule.compiled_check = check_device_gmp
    rules.append(rule)

    # DEV-005: Device cleared — COMPLIANT signal
    # Real dataset GT: device_cleared (161 COMPLIANT device docs)
    # Without this rule, COMPLIANT 510k docs with device_cleared=True
    # get NEEDS_REVIEW because no rule fires True to confirm compliance.
    def check_device_cleared(f: FactDict) -> Optional[bool]:
        cleared   = f.get("device_cleared")
        not_equiv = f.get("substantial_equivalence_not_demonstrated")
        if cleared is True:
            return True
        if not_equiv is True:
            return False
        return None

    rule = ComplianceRule(
        rule_id         = "DEV-005",
        name            = "Device cleared as substantially equivalent (SESE)",
        domain          = "fda_device",
        severity        = "critical",
        rule_source     = "base",
        regulation      = "21 CFR 807.100",
        regulation_text = "Class II devices found substantially equivalent to a predicate device are cleared for commercial distribution.",
        regulation_url  = "https://www.ecfr.gov/current/title-21/section-807.100",
        pass_message    = "Device cleared — substantially equivalent to predicate device",
        fail_message    = "Device NOT substantially equivalent — cannot be commercialized under 510k",
        predicates      = ["device_cleared", "substantial_equivalence_not_demonstrated"],
    )
    rule.compiled_check = check_device_cleared
    rules.append(rule)

    return rules


def _make_clinical_rules(vocab: PredicateVocabulary) -> List[ComplianceRule]:
    """Hand-written FDAAA 801 rules for clinical trial compliance."""

    rules = []

    # CLIN-001: Results posting
    # has_results=False means results were NOT posted → violation
    # has_results=True  means results were posted     → pass
    # results_not_posted=True is an explicit flag from text classification
    def check_results(f: FactDict) -> Optional[bool]:
        not_posted = f.get("results_not_posted")
        has        = f.get("has_results")
        posted     = f.get("clinical_results_posted")

        # Explicit violation signals
        if not_posted is True:
            return False
        if has is False:          # has_results=False → results NOT posted
            return False
        # Explicit pass signals
        if posted is True:
            return True
        if has is True:
            return True
        return None

    rule = ComplianceRule(
        rule_id         = "CLIN-001",
        name            = "FDAAA 801 results posted within 12 months",
        domain          = "clinical",
        severity        = "critical",
        rule_source     = "base",
        regulation      = "FDAAA 801 (42 U.S.C. 282(j))",
        regulation_text = "Responsible parties must submit summary results within 12 months of primary completion.",
        regulation_url  = "https://www.govinfo.gov/content/pkg/PLAW-110publ85/pdf/PLAW-110publ85.pdf",
        pass_message    = "Clinical trial results posted within FDAAA 801 timeframe",
        fail_message    = "Results not posted — FDAAA 801 violation",
        predicates      = ["results_not_posted", "has_results", "clinical_results_posted"],
    )
    rule.compiled_check = check_results
    rules.append(rule)

    # CLIN-002: Safety termination
    def check_safety_termination(f: FactDict) -> Optional[bool]:
        safety = f.get("safety_termination")
        if safety is True:
            return False
        if safety is False:
            return True
        return None

    rule = ComplianceRule(
        rule_id         = "CLIN-002",
        name            = "Trial not terminated for safety reasons",
        domain          = "clinical",
        severity        = "critical",
        rule_source     = "base",
        regulation      = "21 CFR 312.44",
        regulation_text = "FDA may place a clinical hold on an investigation if there is unreasonable risk.",
        regulation_url  = "https://www.ecfr.gov/current/title-21/section-312.44",
        pass_message    = "No safety-related termination identified",
        fail_message    = "Trial terminated due to safety concerns",
        predicates      = ["safety_termination"],
    )
    rule.compiled_check = check_safety_termination
    rules.append(rule)

    # CLIN-003: Protocol / administrative violation
    # Covers admin stops, IRB violations, delay-based terminations.
    # Different from safety_termination — no safety signal, but still a violation.
    # Dataset evidence: NCT04039841 "delay in inclusion that could not be made up
    # within the regulatory timeframe" → protocol_violation=True
    def check_protocol(f: FactDict) -> Optional[bool]:
        violation = f.get("protocol_violation")
        if violation is True:
            return False
        if violation is False:
            return True
        return None

    rule = ComplianceRule(
        rule_id         = "CLIN-003",
        name            = "No protocol or regulatory timeframe violation",
        domain          = "clinical",
        severity        = "critical",
        rule_source     = "base",
        regulation      = "21 CFR 312.62 / FDAAA 801",
        regulation_text = "Investigators shall maintain adequate records including protocol deviations.",
        regulation_url  = "https://www.ecfr.gov/current/title-21/section-312.62",
        pass_message    = "No protocol deviations or regulatory timeframe violations",
        fail_message    = "Protocol violation or regulatory timeframe not met",
        predicates      = ["protocol_violation"],
    )
    rule.compiled_check = check_protocol
    rules.append(rule)

    # CLIN-004: Trial completed satisfactorily — COMPLIANT signal
    # Real dataset GT: clinical_results_posted (121 docs) and
    # trial_completed_satisfactorily (121 docs) for COMPLIANT clinical trials.
    # Without this rule, COMPLIANT clinical docs with only these positive
    # predicates get NEEDS_REVIEW.
    def check_trial_complete(f: FactDict) -> Optional[bool]:
        completed = f.get("trial_completed_satisfactorily")
        posted    = f.get("clinical_results_posted")
        if completed is True or posted is True:
            return True
        return None

    rule = ComplianceRule(
        rule_id         = "CLIN-004",
        name            = "Clinical trial completed satisfactorily",
        domain          = "clinical",
        severity        = "critical",
        rule_source     = "base",
        regulation      = "42 CFR Part 11 / FDAAA 801",
        regulation_text = "Applicable clinical trials shall be registered and results reported in accordance with the law.",
        regulation_url  = "https://www.ecfr.gov/current/title-42/part-11",
        pass_message    = "Clinical trial completed and results posted — FDAAA 801 compliant",
        fail_message    = "Clinical trial not completed satisfactorily",
        predicates      = ["trial_completed_satisfactorily", "clinical_results_posted"],
    )
    rule.compiled_check = check_trial_complete
    rules.append(rule)

    return rules


def _make_sec_rules(vocab: PredicateVocabulary) -> List[ComplianceRule]:
    """Hand-written SOX/SEC compliance rules."""

    rules = []

    # SEC-001: Material weakness
    def check_mw(f: FactDict) -> Optional[bool]:
        mw = f.get("material_weakness_not_disclosed")
        if mw is True:
            return False
        effective = f.get("sox_controls_effective")
        if effective is True:
            return True
        return None

    rule = ComplianceRule(
        rule_id         = "SEC-001",
        name            = "No undisclosed material weakness in internal controls",
        domain          = "sec_financial",
        severity        = "critical",
        rule_source     = "base",
        regulation      = "SOX Section 302/404",
        regulation_text = "CEO and CFO must certify internal controls effectiveness.",
        regulation_url  = "https://www.congress.gov/bill/107th-congress/house-bill/3763",
        pass_message    = "Internal controls effective, no material weakness",
        fail_message    = "Material weakness present — SOX 302/404 violation",
        predicates      = ["material_weakness_not_disclosed", "sox_controls_effective"],
    )
    rule.compiled_check = check_mw
    rules.append(rule)

    # SEC-002: No restatement
    def check_restatement(f: FactDict) -> Optional[bool]:
        restatement = f.get("financial_restatement")
        if restatement is True:
            return False
        if restatement is False:
            return True
        return None

    rule = ComplianceRule(
        rule_id         = "SEC-002",
        name            = "No financial restatement required",
        domain          = "sec_financial",
        severity        = "critical",
        rule_source     = "base",
        regulation      = "SEC Rule 13a-11 / ASC 250",
        regulation_text = "An entity shall report the correction of a material error.",
        regulation_url  = "https://www.ecfr.gov/current/title-17/section-240.13a-11",
        pass_message    = "No financial restatement required",
        fail_message    = "Financial restatement required — prior statements materially incorrect",
        predicates      = ["financial_restatement"],
    )
    rule.compiled_check = check_restatement
    rules.append(rule)

    # SEC-003: No disclosure failure
    # Covers 10-K/A amendments filed due to material misstatements.
    # Dataset evidence: "Form 10-K/A…material misstatements that required restatement"
    # GT predicate = disclosure_failure
    def check_disclosure(f: FactDict) -> Optional[bool]:
        failure = f.get("disclosure_failure")
        if failure is True:
            return False
        if failure is False:
            return True
        return None

    rule = ComplianceRule(
        rule_id         = "SEC-003",
        name            = "No material disclosure failure (no 10-K/A restatement)",
        domain          = "sec_financial",
        severity        = "critical",
        rule_source     = "base",
        regulation      = "SEC Rule 10b-5 / SOX Section 302",
        regulation_text = "It shall be unlawful to make any untrue statement of a material fact.",
        regulation_url  = "https://www.ecfr.gov/current/title-17/section-240.10b-5",
        pass_message    = "No material disclosure failure identified",
        fail_message    = "Material disclosure failure — 10-K/A amendment required for prior misstatement",
        predicates      = ["disclosure_failure"],
    )
    rule.compiled_check = check_disclosure
    rules.append(rule)

    # SEC-004: Clean controls — COMPLIANT signal
    # Real dataset GT: sox_controls_effective (63 docs) and
    # clean_audit_opinion (63 docs) for COMPLIANT SEC filings.
    # Without this rule, COMPLIANT 10-K docs with only these positive
    # predicates get NEEDS_REVIEW because SEC-001 only checks material_weakness.
    def check_clean(f: FactDict) -> Optional[bool]:
        effective = f.get("sox_controls_effective")
        clean_op  = f.get("clean_audit_opinion")
        if effective is True or clean_op is True:
            return True
        return None

    rule = ComplianceRule(
        rule_id         = "SEC-004",
        name            = "SOX controls effective and clean audit opinion",
        domain          = "sec_financial",
        severity        = "critical",
        rule_source     = "base",
        regulation      = "SOX Section 302/404 / PCAOB AS 3101",
        regulation_text = "Management has assessed and certified effectiveness of internal controls over financial reporting.",
        regulation_url  = "https://www.congress.gov/bill/107th-congress/house-bill/3763",
        pass_message    = "SOX controls effective, clean audit opinion — 10-K filing compliant",
        fail_message    = "SOX controls not confirmed effective",
        predicates      = ["sox_controls_effective", "clean_audit_opinion"],
    )
    rule.compiled_check = check_clean
    rules.append(rule)

    return rules


def build_base_rules(vocab: PredicateVocabulary) -> Dict[str, ComplianceRule]:
    """Build all base rules and return as rule_id → ComplianceRule dict."""
    all_rules = []
    all_rules.extend(_make_gmp_rules(vocab))
    all_rules.extend(_make_device_rules(vocab))
    all_rules.extend(_make_clinical_rules(vocab))
    all_rules.extend(_make_sec_rules(vocab))
    return {r.rule_id: r for r in all_rules}


# ─────────────────────────────────────────────────────────────────────────────
#  RULE COMPILER
# ─────────────────────────────────────────────────────────────────────────────

class RuleCompiler:
    """
    Compiles approved HornClauses into ComplianceRule objects.

    The compiled_check is a pure Python closure:
      - looks up predicate values from FactDict
      - returns True (pass), False (fail), or None (uncertain)
      - zero neural involvement

    Citations come from the predicate vocabulary.
    No LLM involvement in compilation.
    """

    def __init__(self, vocab: PredicateVocabulary) -> None:
        self.vocab = vocab

    def compile(
        self,
        clause:      HornClause,
        approved_by: str,
        approved_on: str,
        sme_statement: str,
    ) -> ComplianceRule:
        """Compile an approved HornClause to a ComplianceRule."""
        is_noncompliant = (clause.head == "non_compliant")
        check_fn = self._make_check(clause.body, is_noncompliant)

        # Derive citations from vocabulary
        regulations = []
        reg_texts   = []
        reg_urls    = []
        for pred in clause.body:
            entry = self.vocab.get(pred)
            if entry:
                if entry.regulation not in regulations:
                    regulations.append(entry.regulation)
                    reg_texts.append(entry.regulation_text)
                    reg_urls.append(entry.regulation_url)

        regulation      = "; ".join(regulations) if regulations else "See predicate vocabulary"
        regulation_text = " | ".join(reg_texts[:2]) if reg_texts else ""
        regulation_url  = reg_urls[0] if reg_urls else ""

        severity = "critical" if clause.confidence >= 0.85 else "major"

        if is_noncompliant:
            pass_msg = (f"LEARNED-{clause.clause_id}: no violation detected "
                        f"(conf={clause.confidence:.2f})")
            fail_msg = (f"LEARNED-{clause.clause_id}: violation pattern detected — "
                        f"{', '.join(clause.body)}")
        else:
            pass_msg = (f"LEARNED-{clause.clause_id}: compliance evidence found "
                        f"(conf={clause.confidence:.2f})")
            fail_msg = (f"LEARNED-{clause.clause_id}: expected compliance indicators absent — "
                        f"{', '.join(clause.body)}")

        rule = ComplianceRule(
            rule_id          = f"LEARNED-{clause.clause_id}",
            name             = clause.notation,
            domain           = clause.domain,
            severity         = severity,
            rule_source      = "learned_ilp",
            regulation       = regulation,
            regulation_text  = regulation_text,
            regulation_url   = regulation_url,
            pass_message     = pass_msg,
            fail_message     = fail_msg,
            predicates       = clause.body,
            confidence       = clause.confidence,
            approved_by      = approved_by,
            approved_on      = approved_on,
            sme_statement    = sme_statement,
            training_support = clause.support,
            horn_notation    = clause.notation,
        )
        rule.compiled_check = check_fn
        return rule

    @staticmethod
    def _make_check(body: List[str], invert: bool) -> Callable:
        """
        Create a compiled check closure for a Horn clause.

        For non_compliant head (invert=True):
          clause fires when predicates present and True
          firing means violation → return False (rule fails)
          not firing means no violation → return True (rule passes)

        For compliant head (invert=False):
          clause fires when predicates present and True
          firing means compliance evidence → return True (rule passes)
          not firing means evidence absent → return None (uncertain)
        """
        pred_list = list(body)

        def check(facts: FactDict) -> Optional[bool]:
            any_uncertain = False
            fires         = True

            for pred in pred_list:
                val = facts.get(pred)
                if val is None:
                    any_uncertain = True
                    fires         = False
                    break
                if not bool(val):
                    fires = False
                    break

            if any_uncertain:
                return None   # uncertain — NEEDS_REVIEW

            if invert:
                # non_compliant head: fires → violation → fail
                return not fires
            else:
                # compliant head: fires → evidence found → pass
                # not fires → evidence absent → uncertain
                return True if fires else None

        return check


# ─────────────────────────────────────────────────────────────────────────────
#  SYMBOLIC RULE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class SymbolicRuleEngine:
    """
    The sole decision maker in the compliance pipeline.

    Evaluates all applicable rules against a FactDict.
    Zero neural involvement — pure Python dict lookups.

    Three-valued verdict logic:
      any critical rule = False → NON_COMPLIANT (confirmed violation)
      any critical rule = None  → NEEDS_REVIEW  (insufficient evidence)
      all rules = True          → COMPLIANT
    """

    def __init__(self, vocab: PredicateVocabulary) -> None:
        self.vocab   = vocab
        self._rules: Dict[str, ComplianceRule] = {}
        self._load_base_rules()

    def _load_base_rules(self) -> None:
        base = build_base_rules(self.vocab)
        self._rules.update(base)
        log.info(f"[Engine] Loaded {len(base)} base rules")

    def add_rule(self, rule: ComplianceRule) -> None:
        self._rules[rule.rule_id] = rule
        log.info(f"[Engine] Added rule: {rule.rule_id}")

    def remove_rule(self, rule_id: str) -> None:
        self._rules.pop(rule_id, None)

    @property
    def n_rules(self) -> int:
        return len(self._rules)

    @property
    def n_base_rules(self) -> int:
        return sum(1 for r in self._rules.values() if r.rule_source == "base")

    @property
    def n_learned_rules(self) -> int:
        return sum(1 for r in self._rules.values() if r.rule_source == "learned_ilp")

    def evaluate(
        self,
        facts:  FactDict,
        domain: str,
    ) -> Tuple[str, List[RuleResult]]:
        """
        Evaluate all applicable rules against facts.
        Returns (verdict, rule_results).
        """
        # Get applicable rules for this domain
        applicable = [
            rule for rule in self._rules.values()
            if rule.domain == domain or rule.domain == "universal"
        ]

        if not applicable:
            log.warning(f"[Engine] No rules for domain: {domain}")
            return "NEEDS_REVIEW", []

        results = [rule.evaluate(facts) for rule in applicable]

        # Three-valued verdict logic — driven by CRITICAL rules only.
        # Minor/major rule results are informational and surfaced in
        # rule_results but do not change the verdict by themselves.
        #
        # Logic:
        #   any critical = False  → NON_COMPLIANT  (confirmed violation)
        #   any critical = None   → NEEDS_REVIEW   (uncertain, escalate)
        #   all criticals = True  → COMPLIANT      (all requirements met)
        #
        # Note: if there are no critical rules for this domain, fall back
        # to checking all rules so NEEDS_REVIEW is still possible.
        critical = [r for r in results if r.severity == "critical"]

        if critical:
            crit_fail    = any(r.passed is False for r in critical)
            crit_unclear = any(r.passed is None  for r in critical)
        else:
            # No critical rules — check all rules
            crit_fail    = any(r.passed is False for r in results)
            crit_unclear = any(r.passed is None  for r in results)

        if crit_fail:
            verdict = "NON_COMPLIANT"
        elif crit_unclear:
            verdict = "NEEDS_REVIEW"
        else:
            verdict = "COMPLIANT"

        return verdict, results

    def build_verdict(
        self,
        facts:          FactDict,
        domain:         str,
        doc_hash:       str,
        system_version: str = "regintel-1.0.0",
    ) -> Verdict:
        """Full evaluation returning a Verdict object."""
        verdict, results = self.evaluate(facts, domain)

        return Verdict(
            verdict        = verdict,
            domain         = domain,
            rule_results   = results,
            facts          = facts,
            doc_hash       = doc_hash,
            timestamp      = datetime.utcnow().isoformat() + "Z",
            system_version = system_version,
        )

    def rules_summary(self) -> str:
        lines = [f"Rules: {self.n_rules} total ({self.n_base_rules} base, "
                 f"{self.n_learned_rules} learned)"]
        for rid, rule in sorted(self._rules.items()):
            src = "BASE   " if rule.rule_source == "base" else "LEARNED"
            lines.append(f"  [{src}] {rid:<25} {rule.severity:<8} {rule.name[:50]}")
        return "\n".join(lines)
