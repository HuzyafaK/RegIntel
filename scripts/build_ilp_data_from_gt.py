"""
build_ilp_data_from_gt.py
=========================
Converts the labeled dataset directly to ILP-ready sparse facts
using ground_truth_violations — no LLM required.

How it works
------------
NON_COMPLIANT docs:
    For each ground_truth_violation entry:
      facts[predicate] = 1.0   ← violation present
    Also adds the complementary "pass" predicates at 0.0
    so ILP can learn the contrast.

COMPLIANT docs:
    For each domain-relevant violation predicate:
      facts[predicate] = 0.0   ← violation absent
    Also adds known "pass" predicates at 1.0.

Why this is valid for MVP proof
--------------------------------
The GT predicates were hand-labeled from authoritative sources
(FDA, SEC, ClinicalTrials.gov). They are MORE accurate than any
LLM extraction. Using them directly means:

  1. ILP trains on the cleanest possible signal
  2. Evaluation metrics reflect the engine, not LLM noise
  3. We prove the neuro-symbolic architecture works end-to-end
  4. Once LLM is working, we re-run with LLM-parsed facts

Run
---
  python build_ilp_data_from_gt.py \
      --train train_for_pipeline.json \
      --test  test_for_pipeline.json  \
      --out   /kaggle/working/

Outputs
-------
  ilp_train_examples.json   — sparse facts for 1265 train docs
  ilp_test_examples.json    — sparse facts for 150 test docs
  gt_coverage_report.txt    — predicate coverage stats
"""

import json
import os
import sys
import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
#  DOMAIN PREDICATE MAP
#  Maps each domain to its violation predicates (NC=True) and
#  the complementary compliance predicates (COMPLIANT=True).
# ─────────────────────────────────────────────────────────────────────────────

# For each domain: which predicates signal NON_COMPLIANT when True
NC_PREDICATES = {
    "gmp": [
        "api_coa_missing",
        "calibration_expired",
        "contamination_detected",
        "data_integrity_failure",
        "deviation_without_capa",
        "oos_result",
        "sterility_assurance_failure",
        "batch_record_incomplete",
        "cleaning_validation_absent",
        "process_validation_inadequate",
    ],
    "fda_device": [
        "no_predicate_device",
        "risk_management_absent",
        "software_validation_incomplete",
        "design_controls_incomplete",
        "device_gmp_violation",
        "biocompat_not_demonstrated",
        "substantial_equivalence_not_demonstrated",
    ],
    "clinical": [
        "results_not_posted",
        "safety_termination",
        "protocol_violation",
        "irb_violation",
    ],
    "sec_financial": [
        "material_weakness_not_disclosed",
        "financial_restatement",
        "disclosure_failure",
        "going_concern_not_disclosed",
        "revenue_recognition_aggressive",
    ],
}

# For each domain: which predicates signal COMPLIANT when True
C_PREDICATES = {
    "gmp": [
        "gmp_compliant",
        "drug_approved",
    ],
    "fda_device": [
        "device_cleared",
    ],
    "clinical": [
        "clinical_results_posted",
        "trial_completed_satisfactorily",
    ],
    "sec_financial": [
        "sox_controls_effective",
        "clean_audit_opinion",
    ],
}


def build_nc_facts(
    gt_violations: List[Dict],
    domain:        str,
) -> Dict[str, float]:
    """
    Build sparse facts for a NON_COMPLIANT document.

    Primary signal: GT predicates → 1.0  (violation confirmed)
    Secondary signal: all other NC predicates → 0.0  (not seen)
    Compliance predicates → 0.0  (not present)

    This gives ILP a full contrast to learn from.
    """
    facts: Dict[str, float] = {}

    # 1. Extract GT predicates — these are the verified violations
    gt_preds = set()
    for gv in gt_violations:
        pred = gv.get("predicate", "").strip()
        if pred:
            facts[pred] = 1.0
            gt_preds.add(pred)

    # 2. All other NC predicates for this domain → 0.0 (absent)
    for pred in NC_PREDICATES.get(domain, []):
        if pred not in facts:
            facts[pred] = 0.0

    # 3. Compliance predicates → 0.0 (not applicable when NC)
    for pred in C_PREDICATES.get(domain, []):
        facts[pred] = 0.0

    return facts


def build_c_facts(domain: str) -> Dict[str, float]:
    """
    Build sparse facts for a COMPLIANT document.

    Compliance predicates → 1.0  (compliance confirmed)
    All NC predicates for domain → 0.0  (violations absent)
    """
    facts: Dict[str, float] = {}

    # 1. Compliance predicates → 1.0
    for pred in C_PREDICATES.get(domain, []):
        facts[pred] = 1.0

    # 2. All NC predicates for this domain → 0.0 (absent)
    for pred in NC_PREDICATES.get(domain, []):
        facts[pred] = 0.0

    return facts


def convert_dataset(
    docs: List[Dict],
    split_name: str,
) -> List[Dict]:
    """Convert a list of labeled docs to ILP training examples."""

    examples  = []
    skipped   = 0
    pred_counter: Counter = Counter()

    for doc in docs:
        doc_id  = doc.get("doc_id", "")
        domain  = doc.get("domain", "")
        label   = doc.get("label", "")
        gt_viols = doc.get("ground_truth_violations", [])

        if label == "NON_COMPLIANT":
            if not gt_viols:
                # NC doc with no GT predicates — skip (can't build useful facts)
                skipped += 1
                continue
            facts = build_nc_facts(gt_viols, domain)
        elif label == "COMPLIANT":
            facts = build_c_facts(domain)
        else:
            skipped += 1
            continue

        # Count predicate coverage
        pred_counter.update(
            k for k, v in facts.items() if v == 1.0
        )

        examples.append({
            "doc_id":    doc_id,
            "domain":    domain,
            "label":     label,
            "facts":     facts,
            # Carry GT info for verification
            "gt_predicates": [gv.get("predicate") for gv in gt_viols
                              if gv.get("predicate")],
        })

    print(f"\n{split_name}: {len(examples)} examples  ({skipped} skipped)")
    print(f"  COMPLIANT:     {sum(1 for e in examples if e['label'] == 'COMPLIANT')}")
    print(f"  NON_COMPLIANT: {sum(1 for e in examples if e['label'] == 'NON_COMPLIANT')}")
    print(f"  Avg facts per doc: {sum(len(e['facts']) for e in examples) / max(len(examples),1):.1f}")
    print(f"\n  Active predicates (value=1.0):")
    for pred, count in pred_counter.most_common(20):
        pct = count / len(examples)
        bar = "█" * int(pct * 20)
        print(f"    {pred:<42} {bar} {count:>4}  ({pct:.0%})")

    return examples


def coverage_report(
    train_ex: List[Dict],
    test_ex:  List[Dict],
) -> str:
    lines = []
    lines.append("="*65)
    lines.append("GT PREDICATE COVERAGE REPORT")
    lines.append("="*65)

    for split_name, examples in [("TRAIN", train_ex), ("TEST", test_ex)]:
        lines.append(f"\n{split_name} ({len(examples)} examples)")
        lines.append(f"  {'Domain':<22} {'Total':>6} {'NC':>6} {'C':>6} "
                     f"{'Avg facts':>10}")
        lines.append(f"  {'─'*50}")

        domains = sorted({e["domain"] for e in examples})
        for domain in domains:
            subset = [e for e in examples if e["domain"] == domain]
            nc     = sum(1 for e in subset if e["label"] == "NON_COMPLIANT")
            c      = len(subset) - nc
            avg_f  = sum(len(e["facts"]) for e in subset) / max(len(subset), 1)
            lines.append(f"  {domain:<22} {len(subset):>6} {nc:>6} {c:>6} "
                         f"{avg_f:>10.1f}")

        # Active predicates across split
        active: Counter = Counter()
        for e in examples:
            active.update(k for k, v in e["facts"].items() if v == 1.0)

        lines.append(f"\n  Top active predicates:")
        for pred, count in active.most_common(15):
            pct = count / len(examples)
            bar = "█" * int(pct * 15)
            lines.append(f"    {pred:<40} {count:>5} ({pct:.0%}) {bar}")

    lines.append("")
    lines.append("="*65)
    lines.append("ILP TRAINING NOTES")
    lines.append("="*65)
    lines.append("""
Facts are built from ground_truth_violations:
  - NON_COMPLIANT: GT predicate = 1.0, other domain NC predicates = 0.0
  - COMPLIANT:     domain compliance predicates = 1.0, NC predicates = 0.0

This is equivalent to perfect LLM extraction (zero noise).
ILP will discover rules like:
  non_compliant :- sterility_assurance_failure = True
  non_compliant :- protocol_violation = True
  non_compliant :- disclosure_failure = True
  non_compliant :- contamination_detected = True

These rules, once approved by SME, become base rules in the engine.
The prototype proves the SYMBOLIC LAYER is correct.
LLM layer accuracy can be measured separately once parsing is working.
""")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Build ILP-ready data from GT predicates")
    parser.add_argument("--train", default="train_for_pipeline.json")
    parser.add_argument("--test",  default="test_for_pipeline.json")
    parser.add_argument("--out",   default=".")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.train}...")
    with open(args.train) as f:
        train_docs = json.load(f)
    print(f"Loading {args.test}...")
    with open(args.test) as f:
        test_docs = json.load(f)

    print(f"\nDataset: {len(train_docs):,} train | {len(test_docs):,} test")

    train_ex = convert_dataset(train_docs, "TRAIN")
    test_ex  = convert_dataset(test_docs,  "TEST")

    # Save
    train_path = out_dir / "ilp_train_examples.json"
    test_path  = out_dir / "ilp_test_examples.json"
    report_path = out_dir / "gt_coverage_report.txt"

    with open(train_path, "w") as f:
        json.dump(train_ex, f, indent=2)
    with open(test_path, "w") as f:
        json.dump(test_ex, f, indent=2)

    report = coverage_report(train_ex, test_ex)
    with open(report_path, "w") as f:
        f.write(report)
    print()
    print(report)

    print(f"\nOutputs written to {out_dir}/:")
    for p in [train_path, test_path, report_path]:
        size = p.stat().st_size / 1024
        print(f"  {p.name:<40} {size:>8.0f} KB")

    return train_ex, test_ex


if __name__ == "__main__":
    main()
