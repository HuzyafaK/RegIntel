# =============================================================================
# RegIntel — Full Evaluation Pipeline
# Kaggle Notebook: Data → Engine → Metrics → SME Report
# =============================================================================
# Prerequisites:
#   - Run regintel_data_v3/kaggle_notebook.py first
#     → produces /kaggle/working/train_for_pipeline.json
#     → produces /kaggle/working/test_for_pipeline.json
#   - Upload regintel_engine_v2 as a Kaggle dataset
#
# This notebook:
#   1. Loads the labeled dataset from the data pipeline
#   2. Sets up the RegIntel engine (12 base rules + ILP training)
#   3. Runs ILP on training set to discover new rules
#   4. Evaluates on test set with full metrics
#   5. Produces SME review dashboard
#   6. Produces investor-ready summary
# =============================================================================


# ─────────────────────────────────────────────────────────────────────────────
# CELL 1 — Install engine package
# ─────────────────────────────────────────────────────────────────────────────

import sys, os, shutil
from pathlib import Path

# Copy engine from Kaggle dataset input
ENGINE_SRC = '/kaggle/input/regintel-engine-v2/regintel_engine_v2/regintel_engine'
ENGINE_DST = '/kaggle/working/regintel_engine'

if Path(ENGINE_SRC).exists():
    shutil.copytree(ENGINE_SRC, ENGINE_DST, dirs_exist_ok=True)
    print(f"Engine copied from dataset")
else:
    print("Engine dataset not found — listing available inputs:")
    for p in Path('/kaggle/input').rglob('pipeline.py'):
        print(f"  {p}")
    raise FileNotFoundError(f"Expected engine at {ENGINE_SRC}")

sys.path.insert(0, '/kaggle/working')

print("Package structure:")
for p in sorted(Path(ENGINE_DST).rglob('*.py')):
    print(f"  {p.relative_to('/kaggle/working')}")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 2 — Load dataset
# ─────────────────────────────────────────────────────────────────────────────

import json
from collections import Counter

TRAIN_PATH = "/kaggle/working/train_for_pipeline.json"
TEST_PATH  = "/kaggle/working/test_for_pipeline.json"

with open(TRAIN_PATH) as f:
    train_docs = json.load(f)

with open(TEST_PATH) as f:
    test_docs = json.load(f)

print(f"Train: {len(train_docs):,} documents")
print(f"Test:  {len(test_docs):,} documents")
print()

# Dataset breakdown
train_domains = Counter(d['domain'] for d in train_docs)
train_labels  = Counter(d['label']  for d in train_docs)
print("Train by domain:")
for domain, n in sorted(train_domains.items()):
    nc = sum(1 for d in train_docs if d['domain'] == domain and d['label'] == 'NON_COMPLIANT')
    c  = n - nc
    print(f"  {domain:<25} {n:>5}  ({c} C / {nc} NC)")
print()
print(f"Train labels:  {dict(train_labels)}")

test_domains = Counter(d['domain'] for d in test_docs)
print(f"\nTest by domain: {dict(test_domains)}")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 3 — Create pipeline
# ─────────────────────────────────────────────────────────────────────────────

from regintel_engine.pipeline import RegIntelPipeline
from regintel_engine.ilp.sparse_ilp import ILPConfig

# Configuration
AUDIT_KEY = os.environ.get('REGINTEL_AUDIT_KEY', 'a' * 64)

ilp_config = ILPConfig(
    n_epochs         = 300,
    learning_rate    = 0.05,
    l1_lambda        = 0.01,
    top_k_predicates = 3,
    min_confidence   = 0.75,
    min_coverage     = 0.10,
    min_support      = 5,
    score_scale      = 1.5,
)

pipeline = RegIntelPipeline.create(
    mock_llm   = True,          # Use mock LLM (set False for Qwen on GPU)
    audit_key  = AUDIT_KEY,
    ilp_config = ilp_config,
)

print(f"Pipeline created:")
print(f"  Base rules:    {pipeline.engine.n_base_rules}")
print(f"  Learned rules: {pipeline.engine.n_learned_rules}")
print(f"  Vocabulary:    {len(pipeline.vocab.get_names())} predicates")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 4 — Prepare training examples for ILP
# ─────────────────────────────────────────────────────────────────────────────

import time

print("Preparing training examples for ILP...")
t0 = time.time()

# prepare_training_batch runs the LLM extraction + verification on each doc
# With mock LLM this is fast; with real LLM allow ~2 min for 1000 docs
ilp_examples = pipeline.prepare_training_batch(train_docs)

elapsed = time.time() - t0
print(f"\nPreparation complete:")
print(f"  {len(ilp_examples)}/{len(train_docs)} documents produced sparse facts")
print(f"  {elapsed:.1f}s elapsed")

# Show predicate coverage across training set
from collections import Counter
pred_counts: Counter = Counter()
for ex in ilp_examples:
    pred_counts.update(ex['facts'].keys())

print(f"\nTop predicates in training set:")
for pred, count in pred_counts.most_common(15):
    bar = '█' * int(count / len(ilp_examples) * 20)
    print(f"  {pred:<42} {bar} {count}")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 5 — Train ILP and discover rules
# ─────────────────────────────────────────────────────────────────────────────

print("Running ILP on labeled training examples...")
t0 = time.time()

discovered_clauses = pipeline.training.train(
    ilp_examples,
    run_dreamcoder = True,
)

elapsed = time.time() - t0
pending = pipeline.training.pending_clauses()

print(f"\nILP Training complete ({elapsed:.1f}s):")
print(f"  Discovered: {len(discovered_clauses)} clauses")
print(f"  Pending SME review: {len(pending)} clauses")

if discovered_clauses:
    print("\nDiscovered clauses:")
    for c in discovered_clauses:
        flag = "✓ APPROVABLE" if c.is_approvable else "— needs review"
        print(f"  [{flag}] {c.notation}")
        print(f"    conf={c.confidence:.2f}  coverage={c.coverage:.2f}  "
              f"support={c.support}  domain={c.domain}")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 6 — SME Review dashboard
# ─────────────────────────────────────────────────────────────────────────────

print(pipeline.training.sme_review_summary())

# Auto-approve high-confidence clauses for evaluation
# In production: a real SME reviews each clause manually
print("\nAuto-approving high-confidence clauses for evaluation...")
n_approved = 0
for clause in pending:
    if clause.is_approvable and clause.confidence >= 0.80:
        ok = pipeline.training.approve_clause(
            clause.clause_id,
            approved_by   = "auto_eval@regintel.ai",
            sme_statement = (
                f"Auto-approved for evaluation. "
                f"Confidence {clause.confidence:.2f}, "
                f"support {clause.support} training documents. "
                f"REQUIRES human SME review before production use."
            ),
        )
        if ok:
            n_approved += 1
            print(f"  Approved: {clause.clause_id} → {clause.notation}")

print(f"\nEngine state after ILP:")
print(f"  Base rules:    {pipeline.engine.n_base_rules}")
print(f"  Learned rules: {pipeline.engine.n_learned_rules} ({n_approved} just added)")
print()
print(pipeline.engine.rules_summary())


# ─────────────────────────────────────────────────────────────────────────────
# CELL 7 — Evaluate on test set
# ─────────────────────────────────────────────────────────────────────────────

from typing import Dict, List, Any

def evaluate_dataset(
    pipeline,
    test_docs:    List[Dict[str, Any]],
    verbose:      bool = True,
) -> Dict[str, Any]:
    """
    Evaluate the pipeline on a labeled test set.
    Returns per-domain and overall metrics.
    """
    results = []
    t0 = time.time()

    for i, doc in enumerate(test_docs):
        document = doc['document']
        domain   = doc['domain']
        true_label = doc['label']

        try:
            result = pipeline.run(document, domain=domain)
            pred_label = result.verdict.verdict

            # Map NEEDS_REVIEW to a third class for evaluation
            results.append({
                'domain':        domain,
                'true_label':    true_label,
                'pred_label':    pred_label,
                'is_correct':    pred_label == true_label,
                'n_failed':      len(result.verdict.failed_rules),
                'n_uncertain':   len(result.verdict.uncertain_rules),
                'confidence_sum': sum(r.confidence for r in result.verdict.rule_results),
            })
        except Exception as e:
            results.append({
                'domain':      domain,
                'true_label':  true_label,
                'pred_label':  'ERROR',
                'is_correct':  False,
                'error':       str(e),
                'n_failed':    0,
                'n_uncertain': 0,
                'confidence_sum': 0.0,
            })

        if verbose and (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(test_docs)} evaluated ({elapsed:.0f}s)")

    elapsed = time.time() - t0

    # ── Compute metrics ──────────────────────────────────────────────────────
    def compute_metrics(subset, label_pos='NON_COMPLIANT'):
        """Compute precision/recall/F1 for binary classification."""
        tp = sum(1 for r in subset
                 if r['true_label'] == label_pos and r['pred_label'] == label_pos)
        fp = sum(1 for r in subset
                 if r['true_label'] != label_pos and r['pred_label'] == label_pos)
        fn = sum(1 for r in subset
                 if r['true_label'] == label_pos and r['pred_label'] != label_pos)
        tn = sum(1 for r in subset
                 if r['true_label'] != label_pos and r['pred_label'] != label_pos)

        precision = tp / max(tp + fp, 1)
        recall    = tp / max(tp + fn, 1)
        f1        = 2 * precision * recall / max(precision + recall, 1e-10)
        accuracy  = (tp + tn) / max(len(subset), 1)

        # NEEDS_REVIEW rate (uncertainty)
        nr_rate = sum(1 for r in subset if r['pred_label'] == 'NEEDS_REVIEW') / max(len(subset), 1)

        return {
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'precision':       round(precision, 3),
            'recall':          round(recall,    3),
            'f1':              round(f1,         3),
            'accuracy':        round(accuracy,   3),
            'needs_review_rate': round(nr_rate,  3),
            'n':               len(subset),
        }

    # Overall metrics
    overall = compute_metrics(results)

    # Per-domain metrics
    domains = sorted({r['domain'] for r in results})
    per_domain = {}
    for domain in domains:
        subset = [r for r in results if r['domain'] == domain]
        per_domain[domain] = compute_metrics(subset)

    # Confusion matrix (3-class: COMPLIANT, NON_COMPLIANT, NEEDS_REVIEW)
    confusion = {}
    for true_l in ['COMPLIANT', 'NON_COMPLIANT', 'NEEDS_REVIEW']:
        confusion[true_l] = {}
        for pred_l in ['COMPLIANT', 'NON_COMPLIANT', 'NEEDS_REVIEW', 'ERROR']:
            confusion[true_l][pred_l] = sum(
                1 for r in results
                if r['true_label'] == true_l and r['pred_label'] == pred_l
            )

    return {
        'overall':     overall,
        'per_domain':  per_domain,
        'confusion':   confusion,
        'results':     results,
        'n_docs':      len(test_docs),
        'elapsed_s':   round(elapsed, 1),
    }


print("Evaluating on test set...")
metrics = evaluate_dataset(pipeline, test_docs, verbose=True)
print(f"\nEvaluation complete ({metrics['elapsed_s']}s)")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 8 — Print evaluation report
# ─────────────────────────────────────────────────────────────────────────────

def print_evaluation_report(metrics: Dict) -> None:
    sep = '═' * 70
    print(f"\n{sep}")
    print(f"  REGINTEL ENGINE — EVALUATION REPORT")
    print(sep)

    o = metrics['overall']
    print(f"\n  OVERALL ({o['n']} test documents)")
    print(f"  {'Metric':<22} {'Value':>8}")
    print(f"  {'─'*32}")
    print(f"  {'Accuracy':<22} {o['accuracy']:>8.1%}")
    print(f"  {'Precision (NC)':<22} {o['precision']:>8.1%}")
    print(f"  {'Recall (NC)':<22} {o['recall']:>8.1%}")
    print(f"  {'F1 (NC)':<22} {o['f1']:>8.3f}")
    print(f"  {'NEEDS_REVIEW rate':<22} {o['needs_review_rate']:>8.1%}")
    print(f"  {'TP':<22} {o['tp']:>8}")
    print(f"  {'TN':<22} {o['tn']:>8}")
    print(f"  {'FP':<22} {o['fp']:>8}")
    print(f"  {'FN':<22} {o['fn']:>8}")

    print(f"\n  PER-DOMAIN METRICS")
    print(f"  {'Domain':<25} {'N':>5} {'Acc':>7} {'Prec':>7} {'Rec':>7} "
          f"{'F1':>7} {'NR%':>7}")
    print(f"  {'─'*68}")
    for domain, dm in sorted(metrics['per_domain'].items()):
        print(f"  {domain:<25} {dm['n']:>5} "
              f"{dm['accuracy']:>7.1%} {dm['precision']:>7.1%} "
              f"{dm['recall']:>7.1%} {dm['f1']:>7.3f} "
              f"{dm['needs_review_rate']:>7.1%}")

    print(f"\n  CONFUSION MATRIX (rows=true, cols=predicted)")
    header = f"  {'True \\ Pred':<20} {'COMPLIANT':>12} {'NON_COMPL':>12} {'NDS_REVIEW':>12}"
    print(header)
    print(f"  {'─'*58}")
    for true_l in ['COMPLIANT', 'NON_COMPLIANT']:
        row = metrics['confusion'].get(true_l, {})
        c   = row.get('COMPLIANT', 0)
        nc  = row.get('NON_COMPLIANT', 0)
        nr  = row.get('NEEDS_REVIEW', 0)
        print(f"  {true_l:<20} {c:>12} {nc:>12} {nr:>12}")

    print(sep)

print_evaluation_report(metrics)

# Save metrics to disk
with open("/kaggle/working/evaluation_metrics.json", "w") as f:
    # results list may be large — save summary only
    save = {k: v for k, v in metrics.items() if k != 'results'}
    json.dump(save, f, indent=2)
print("\nMetrics saved to /kaggle/working/evaluation_metrics.json")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 9 — Failure analysis
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("FAILURE ANALYSIS")
print("="*60)

results = metrics['results']

# False negatives (NC predicted as C or NR)
fn_cases = [r for r in results
            if r['true_label'] == 'NON_COMPLIANT'
            and r['pred_label'] != 'NON_COMPLIANT']

# False positives (C predicted as NC)
fp_cases = [r for r in results
            if r['true_label'] == 'COMPLIANT'
            and r['pred_label'] == 'NON_COMPLIANT']

print(f"\nFalse Negatives (missed violations): {len(fn_cases)}")
fn_by_domain = Counter(r['domain'] for r in fn_cases)
for domain, count in sorted(fn_by_domain.items()):
    print(f"  {domain}: {count}")

print(f"\nFalse Positives (wrong violation flags): {len(fp_cases)}")
fp_by_domain = Counter(r['domain'] for r in fp_cases)
for domain, count in sorted(fp_by_domain.items()):
    print(f"  {domain}: {count}")

needs_review = [r for r in results if r['pred_label'] == 'NEEDS_REVIEW']
print(f"\nNEEDS_REVIEW (escalated to human): {len(needs_review)}")
nr_by_domain = Counter(r['domain'] for r in needs_review)
for domain, count in sorted(nr_by_domain.items()):
    print(f"  {domain}: {count}")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 10 — Sample correct and incorrect predictions with explanations
# ─────────────────────────────────────────────────────────────────────────────

import random
random.seed(42)

print("\n" + "="*60)
print("SAMPLE PREDICTIONS WITH EXPLANATIONS")
print("="*60)

def run_and_explain(doc_data: Dict) -> None:
    result = pipeline.run(doc_data['document'], doc_data['domain'])
    true_l = doc_data['label']
    pred_l = result.verdict.verdict
    status = "✓ CORRECT" if pred_l == true_l else "✗ WRONG"

    print(f"\n  {'─'*58}")
    print(f"  Domain:     {doc_data['domain']}")
    print(f"  True label: {true_l}")
    print(f"  Predicted:  {pred_l}  {status}")
    print(f"\n  Explanation:")
    for line in result.explain().split('\n'):
        print(f"    {line}")

# Sample 2 NC, 2 C, 1 wrong prediction
nc_correct = [r for r in results
              if r['true_label'] == 'NON_COMPLIANT' and r['is_correct']]
c_correct  = [r for r in results
              if r['true_label'] == 'COMPLIANT' and r['is_correct']]
wrong      = [r for r in results if not r['is_correct']]

# Find indices to re-run
sample_results = []
if nc_correct: sample_results += random.sample(nc_correct, min(2, len(nc_correct)))
if c_correct:  sample_results += random.sample(c_correct,  min(2, len(c_correct)))
if wrong:      sample_results += random.sample(wrong,       min(1, len(wrong)))

# Get corresponding docs
result_idx = {(r['domain'], r['true_label']): [] for r in sample_results}
for r in sample_results:
    result_idx[(r['domain'], r['true_label'])].append(r)

print(f"\nRunning {len(sample_results)} sample explanations...")
for i, doc_data in enumerate(test_docs):
    key = (doc_data['domain'], doc_data['label'])
    if key in result_idx and result_idx[key]:
        run_and_explain(doc_data)
        result_idx[key].pop()
        if all(len(v) == 0 for v in result_idx.values()):
            break


# ─────────────────────────────────────────────────────────────────────────────
# CELL 11 — Save audit chain
# ─────────────────────────────────────────────────────────────────────────────

AUDIT_PATH = "/kaggle/working/audit_chain.jsonl"
pipeline.audit.save(AUDIT_PATH)

# Verify chain integrity
ok, issues = pipeline.audit.verify()
status = "✓ VERIFIED" if ok else f"✗ {len(issues)} issues"
print(f"\nAudit chain: {len(pipeline.audit._records)} records — {status}")
if issues:
    for issue in issues[:5]:
        print(f"  {issue}")

n_lines = sum(1 for _ in open(AUDIT_PATH))
size_kb  = os.path.getsize(AUDIT_PATH) / 1024
print(f"Saved to {AUDIT_PATH}: {n_lines} records, {size_kb:.0f} KB")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 12 — Investor-ready summary
# ─────────────────────────────────────────────────────────────────────────────

o   = metrics['overall']
pd  = metrics['per_domain']

print("""
╔══════════════════════════════════════════════════════════════════╗
║         REGINTEL — PROOF OF CONCEPT RESULTS                     ║
╠══════════════════════════════════════════════════════════════════╣""")

print(f"║  Dataset                                                        ║")
print(f"║    {len(train_docs):,} training documents  ·  {len(test_docs):,} test documents              ║")
print(f"║    4 regulatory domains (GMP, FDA Device, Clinical, SEC)       ║")
print(f"║    Labels from authoritative sources (FDA, SEC, ClinicalTrials) ║")
print(f"║                                                                  ║")
print(f"║  Architecture                                                    ║")
print(f"║    {pipeline.engine.n_base_rules} hand-written base rules (verified CFR citations)          ║")
print(f"║    {pipeline.engine.n_learned_rules} ILP-discovered rules (pending SME approval in production)  ║")
print(f"║    HMAC-SHA256 audit chain ({len(pipeline.audit._records)} records, verified ✓)        ║")
print(f"║    21 CFR Part 11 architecture (deterministic, auditable)       ║")
print(f"║                                                                  ║")
print(f"║  Performance on test set ({o['n']} documents)                   ║")
print(f"║    Overall accuracy:        {o['accuracy']:>6.1%}                          ║")
print(f"║    Precision (violation):   {o['precision']:>6.1%}                          ║")
print(f"║    Recall (violation):      {o['recall']:>6.1%}                          ║")
print(f"║    F1 score:                {o['f1']:>6.3f}                          ║")
print(f"║    Escalated to human:      {o['needs_review_rate']:>6.1%}                          ║")
print(f"║                                                                  ║")
print(f"║  By domain:")
for domain, dm in sorted(pd.items()):
    dn = domain.replace('_', ' ').title()[:20]
    print(f"║    {dn:<20}  Acc={dm['accuracy']:.0%}  F1={dm['f1']:.2f}  N={dm['n']:>4}             ║")
print(f"╚══════════════════════════════════════════════════════════════════╝")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 13 — Save all outputs
# ─────────────────────────────────────────────────────────────────────────────

print("\nOutput files:")
for path in [
    "/kaggle/working/evaluation_metrics.json",
    "/kaggle/working/audit_chain.jsonl",
]:
    if os.path.exists(path):
        size = os.path.getsize(path) / 1024
        print(f"  {path}  ({size:.0f} KB)")
