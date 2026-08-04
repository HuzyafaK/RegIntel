# =============================================================================
# RegIntel — Preprocessing Pipeline (Groq API)
# =============================================================================
# This notebook runs LLM extraction on the entire training dataset using
# Groq's cloud API (Llama 3.3 70B) and saves the parsed facts to disk.
#
# Why preprocess first?
# - Avoids re-running LLM on every ILP training iteration (expensive)
# - Lets you inspect what the LLM extracted before training rules
# - Saves rate-limit budget (each doc is called exactly once)
#
# Outputs written to /kaggle/working/:
#   parsed_train.json   — 1265 docs with extracted facts + verdicts
#   parsed_test.json    — 150 docs with extracted facts + verdicts
#   parse_failures.json — docs where extraction returned no facts
#   parse_report.txt    — coverage and quality report
#
# Setup:
#   1. Add your GROQ_API_KEY as a Kaggle secret (Settings → Secrets)
#   2. Run cells 1-3 to set up the engine
#   3. Run cell 4 to start batch extraction (takes ~15-45 min depending on tier)
#   4. Run cell 5 to inspect results
#   5. Download parsed_train.json — use it as input to the ILP training notebook
#
# Groq rate limits:
#   Free tier:  30 req/min, 6000 tokens/min → ~85 min for 1265 docs
#   Dev tier:   14400 req/min              → ~3 min for 1265 docs
#   The notebook handles rate limits automatically with backoff.
# =============================================================================


# ─────────────────────────────────────────────────────────────────────────────
# CELL 1 — Install Groq + set up engine
# ─────────────────────────────────────────────────────────────────────────────

import subprocess, sys, os
subprocess.run([sys.executable, "-m", "pip", "install", "groq", "-q"], check=True)
print("groq installed ✓")

# Engine must already be set up (run the setup cells from kaggle_notebook_per_cell.py)
sys.path.insert(0, "/kaggle/working")
from regintel_engine.pipeline import RegIntelPipeline
from regintel_engine.ilp.sparse_ilp import ILPConfig

# Set your API key — either via Kaggle secret or directly here
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    # Try Kaggle secrets
    try:
        from kaggle_secrets import UserSecretsClient
        GROQ_API_KEY = UserSecretsClient().get_secret("GROQ_API_KEY")
        print("API key loaded from Kaggle secrets ✓")
    except Exception:
        raise ValueError(
            "GROQ_API_KEY not found.\n"
            "Option 1: Kaggle → Add-ons → Secrets → Add GROQ_API_KEY\n"
            "Option 2: GROQ_API_KEY = 'your_key_here' directly in this cell\n"
            "Get free key: https://console.groq.com"
        )

# Best Groq model for regulatory extraction:
# llama-3.3-70b-versatile  — 128k context, best quality
# llama-3.1-8b-instant     — fastest, good quality, higher rate limits
GROQ_MODEL = "llama-3.3-70b-versatile"

pipeline = RegIntelPipeline.create(
    backend          = "groq",
    groq_model       = GROQ_MODEL,
    groq_api_key     = GROQ_API_KEY,
    max_input_length = 12000,   # full documents — 70B handles this easily
    audit_key        = os.environ.get("REGINTEL_AUDIT_KEY", "a" * 64),
    ilp_config       = ILPConfig(
        n_epochs=400, min_support=5, min_confidence=0.75, score_scale=1.5),
)

print(f"\nPipeline ready:")
print(f"  Backend:    {pipeline.extractor._backend}")
print(f"  Model:      {pipeline.extractor._tokenizer}")   # model name stored here
print(f"  Max input:  {pipeline.extractor._max_input_length} chars")
print(f"  Base rules: {pipeline.engine.n_base_rules}")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 2 — Load dataset
# ─────────────────────────────────────────────────────────────────────────────

import json, os
from collections import Counter

# ── Configure paths ───────────────────────────────────────────────────────────
# train_for_pipeline.json should have ~1265 docs (from data pipeline v3)
# test_for_pipeline.json  should have ~150  docs
#
# If BOTH show 150 docs with identical breakdowns, you are loading the
# same file twice. Check:
#   1. Did the data pipeline run complete? (it produces 1265 train docs)
#   2. Are the paths below correct?
#   3. If you only have the 150-doc test set, set TRAIN_PATH = TEST_PATH
#      to run in test-only mode (evaluation only, no ILP training)

TRAIN_PATH = "/kaggle/working/train_for_pipeline.json"
TEST_PATH  = "/kaggle/working/test_for_pipeline.json"

# Auto-detect if correct files exist; try alternate locations
for label, path_var in [("TRAIN", "TRAIN_PATH"), ("TEST", "TEST_PATH")]:
    if not os.path.exists(eval(path_var)):
        # Search for JSON files with the right name
        candidates = []
        for root, dirs, files in os.walk("/kaggle"):
            for f in files:
                if "pipeline" in f and f.endswith(".json"):
                    candidates.append(os.path.join(root, f))
        print(f"WARNING: {eval(path_var)} not found.")
        if candidates:
            print(f"  Candidate files found:")
            for c in candidates:
                sz = os.path.getsize(c) / 1024
                print(f"    {c}  ({sz:.0f} KB)")
        print(f"  Update {path_var} above to the correct path.")

with open(TRAIN_PATH) as f:
    train_docs = json.load(f)
with open(TEST_PATH) as f:
    test_docs  = json.load(f)

# ── Validate no accidental duplicate ─────────────────────────────────────────
if len(train_docs) == len(test_docs):
    # Check if they're the same file
    train_ids = {d["doc_id"] for d in train_docs[:10]}
    test_ids  = {d["doc_id"] for d in test_docs[:10]}
    if train_ids == test_ids:
        print()
        print("⚠ WARNING: train and test appear to be the SAME FILE.")
        print("  train_for_pipeline.json should have ~1265 docs.")
        print("  test_for_pipeline.json should have ~150 docs.")
        print()
        print("  Likely cause: the data pipeline v3 notebook was not run,")
        print("  OR only the test split was saved as train_for_pipeline.json.")
        print()
        print("  Proceeding anyway — results will be evaluation-only (no ILP).")
    else:
        print(f"ℹ Train and test have same count ({len(train_docs)}) "
              f"but different doc IDs — looks OK.")

print(f"Train: {len(train_docs):,} documents")
print(f"Test:  {len(test_docs):,} documents")

for split_name, split_docs in [("TRAIN", train_docs), ("TEST", test_docs)]:
    domains = Counter(d["domain"] for d in split_docs)
    labels  = Counter(d["label"]  for d in split_docs)
    print(f"\n{split_name}:")
    for domain, n in sorted(domains.items()):
        nc = sum(1 for d in split_docs if d["domain"] == domain and d["label"] == "NON_COMPLIANT")
        print(f"  {domain:<22} {n:>5} ({n-nc} C / {nc} NC)")
    print(f"  Total C={labels['COMPLIANT']} NC={labels['NON_COMPLIANT']}")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 3 — Quick sanity check on 5 real documents
# ─────────────────────────────────────────────────────────────────────────────

import time

print("Running sanity check on 5 documents...\n")

# Pick 1-2 from each domain
sample_indices = []
seen_domains = set()
for i, doc in enumerate(train_docs):
    if doc["domain"] not in seen_domains:
        sample_indices.append(i)
        seen_domains.add(doc["domain"])
    if len(seen_domains) >= 4:
        break

all_good = True
for idx in sample_indices:
    doc = train_docs[idx]
    t0  = time.time()

    result  = pipeline.run(doc["document"], domain=doc["domain"])
    elapsed = time.time() - t0

    facts   = result.facts.facts
    verdict = result.verdict.verdict
    n_facts = len(facts)

    # Expected: NC docs should have some facts; not all NEEDS_REVIEW on real data
    flag = "✓" if n_facts > 0 else "⚠ no facts"
    print(f"{flag} [{doc['domain']}] doc_id={doc['doc_id'][:30]}")
    print(f"     label={doc['label']:<15}  verdict={verdict:<15}  "
          f"n_facts={n_facts}  ({elapsed:.1f}s)")
    print(f"     facts: {sorted(facts.keys())}")

    # Check ground truth predicates
    for gv in doc.get("ground_truth_violations", []):
        gt_pred = gv.get("predicate")
        if gt_pred:
            found = gt_pred in facts
            status = "✓ found" if found else "✗ MISSING"
            print(f"     GT predicate '{gt_pred}': {status}")
            if not found:
                all_good = False
    print()

if all_good:
    print("Sanity check: PASSED — Groq is extracting correctly ✓")
else:
    print("Sanity check: some GT predicates missing — review above")
    print("Tip: check the document text contains the expected evidence phrases")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 4 — Batch preprocessing (full dataset)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_batch(
    pipeline,
    docs:           list,
    split_name:     str,
    output_path:    str,
    failures_path:  str,
    checkpoint_every: int = 50,
) -> dict:
    """
    Run LLM extraction on a batch of documents and save results.

    Handles rate limits, errors, and checkpointing so you can resume
    if the notebook session times out.

    Each record saved contains:
      doc_id, domain, label, ground_truth_violations (from dataset)
      extracted_facts: {predicate: value, ...}
      fact_confidences: {predicate: float, ...}
      verdict: COMPLIANT | NON_COMPLIANT | NEEDS_REVIEW
      failed_rules: [rule_id, ...]
      elapsed_s: float
      error: str | null
    """
    import os

    # Resume from checkpoint if it exists
    results   = []
    done_ids  = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            results = json.load(f)
        done_ids = {r["doc_id"] for r in results}
        print(f"Resuming from checkpoint: {len(done_ids)} already done")

    failures      = []
    total         = len(docs)
    remaining     = [d for d in docs if d["doc_id"] not in done_ids]

    print(f"\n{split_name}: processing {len(remaining)}/{total} documents")
    print(f"Estimated time: {len(remaining) * 2 / 60:.0f}-{len(remaining) * 5 / 60:.0f} min")
    print()

    t_start = time.time()

    for i, doc in enumerate(remaining):
        doc_id  = doc["doc_id"]
        domain  = doc["domain"]
        label   = doc["label"]

        # Progress display
        if i % 10 == 0:
            elapsed  = time.time() - t_start
            rate     = (i + 1) / max(elapsed, 1)
            eta_min  = (len(remaining) - i) / max(rate, 0.01) / 60
            pct      = (len(done_ids) + i + 1) / total * 100
            print(f"  [{pct:5.1f}%] {i+1:4d}/{len(remaining)}  "
                  f"rate={rate:.1f}/s  ETA={eta_min:.0f}min  "
                  f"failures={len(failures)}")

        t0 = time.time()
        error = None

        try:
            result   = pipeline.run(doc["document"], domain=domain)
            facts    = dict(result.facts.facts)
            confs    = {
                k: result.facts.provenance[k].confidence
                for k in facts
                if k in result.facts.provenance
            }
            verdict      = result.verdict.verdict
            failed_rules = [r.rule_id for r in result.verdict.failed_rules]
            uncertain    = [r.rule_id for r in result.verdict.uncertain_rules]

        except Exception as e:
            facts         = {}
            confs         = {}
            verdict       = "ERROR"
            failed_rules  = []
            uncertain     = []
            error         = str(e)
            failures.append({"doc_id": doc_id, "domain": domain, "error": error})

        elapsed = time.time() - t0

        record = {
            "doc_id":                 doc_id,
            "domain":                 domain,
            "label":                  label,
            "ground_truth_violations":doc.get("ground_truth_violations", []),
            "label_confidence":       doc.get("label_confidence", "medium"),
            "extracted_facts":        facts,
            "fact_confidences":       confs,
            "verdict":                verdict,
            "failed_rules":           failed_rules,
            "uncertain_rules":        uncertain,
            "n_facts":                len(facts),
            "elapsed_s":              round(elapsed, 2),
            "error":                  error,
        }
        results.append(record)
        done_ids.add(doc_id)

        # Checkpoint
        if (i + 1) % checkpoint_every == 0:
            with open(output_path, "w") as f:
                json.dump(results, f)
            with open(failures_path, "w") as f:
                json.dump(failures, f)
            print(f"  → checkpoint saved ({len(results)} records)")

    # Final save
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    with open(failures_path, "w") as f:
        json.dump(failures, f, indent=2)

    total_time = time.time() - t_start
    return {
        "total":      total,
        "processed":  len(results),
        "failures":   len(failures),
        "elapsed_min": round(total_time / 60, 1),
        "results":    results,
        "failures_list": failures,
    }


# Run preprocessing
print("="*60)
print("BATCH PREPROCESSING — TRAIN SET")
print("="*60)

train_stats = preprocess_batch(
    pipeline     = pipeline,
    docs         = train_docs,
    split_name   = "TRAIN",
    output_path  = "/kaggle/working/parsed_train.json",
    failures_path= "/kaggle/working/parse_failures_train.json",
    checkpoint_every = 50,
)

print(f"\nTrain preprocessing complete:")
print(f"  Processed: {train_stats['processed']:,}/{train_stats['total']:,}")
print(f"  Failures:  {train_stats['failures']}")
print(f"  Time:      {train_stats['elapsed_min']:.1f} min")

print()
print("="*60)
print("BATCH PREPROCESSING — TEST SET")
print("="*60)

test_stats = preprocess_batch(
    pipeline     = pipeline,
    docs         = test_docs,
    split_name   = "TEST",
    output_path  = "/kaggle/working/parsed_test.json",
    failures_path= "/kaggle/working/parse_failures_test.json",
    checkpoint_every = 25,
)

print(f"\nTest preprocessing complete:")
print(f"  Processed: {test_stats['processed']:,}/{test_stats['total']:,}")
print(f"  Failures:  {test_stats['failures']}")
print(f"  Time:      {test_stats['elapsed_min']:.1f} min")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 5 — Extraction quality report
# ─────────────────────────────────────────────────────────────────────────────

def quality_report(results: list, split_name: str) -> None:
    total    = len(results)
    errors   = [r for r in results if r["verdict"] == "ERROR"]
    no_facts = [r for r in results if r["n_facts"] == 0 and r["verdict"] != "ERROR"]

    # Verdict distribution
    verdicts = Counter(r["verdict"] for r in results)

    # Fact coverage by domain
    by_domain: dict = {}
    for r in results:
        d = r["domain"]
        if d not in by_domain:
            by_domain[d] = {"total": 0, "with_facts": 0, "total_facts": 0,
                            "gt_found": 0, "gt_total": 0}
        s = by_domain[d]
        s["total"]       += 1
        s["with_facts"]  += int(r["n_facts"] > 0)
        s["total_facts"] += r["n_facts"]
        for gv in r["ground_truth_violations"]:
            gt = gv.get("predicate")
            if gt:
                s["gt_total"] += 1
                if gt in r["extracted_facts"]:
                    s["gt_found"] += 1

    # Predicate coverage
    pred_counts: Counter = Counter()
    for r in results:
        pred_counts.update(r["extracted_facts"].keys())

    sep = "─" * 68
    print(f"\n{'='*68}")
    print(f"  EXTRACTION QUALITY REPORT — {split_name} ({total} documents)")
    print(f"{'='*68}")
    print(f"\n  Verdict distribution:")
    for v, n in sorted(verdicts.items()):
        pct = n / total
        bar = "█" * int(pct * 30)
        print(f"    {v:<18} {n:>5} ({pct:5.1%})  {bar}")

    print(f"\n  Coverage:")
    print(f"    With facts:     {sum(r['n_facts'] > 0 for r in results):>5} / {total} "
          f"({sum(r['n_facts'] > 0 for r in results)/total:.1%})")
    print(f"    Errors:         {len(errors):>5}")
    print(f"    Zero facts:     {len(no_facts):>5}")

    print(f"\n  {sep}")
    print(f"  {'Domain':<22} {'N':>5} {'w/facts':>8} {'avg':>5} "
          f"{'GT found':>10} {'GT %':>7}")
    print(f"  {sep}")
    for domain, s in sorted(by_domain.items()):
        avg_facts = s["total_facts"] / max(s["total"], 1)
        gt_pct    = s["gt_found"] / max(s["gt_total"], 1)
        gt_str    = f"{s['gt_found']}/{s['gt_total']}"
        print(f"  {domain:<22} {s['total']:>5} {s['with_facts']:>8} "
              f"{avg_facts:>5.1f} {gt_str:>10} {gt_pct:>7.1%}")

    print(f"\n  Top extracted predicates:")
    for pred, count in pred_counts.most_common(15):
        pct = count / total
        bar = "█" * int(pct * 20)
        print(f"    {pred:<42} {bar} {count:>4} ({pct:.0%})")

    # Precision vs ground truth
    tp = fp = fn = 0
    for r in results:
        gt_preds = {gv["predicate"] for gv in r["ground_truth_violations"]
                    if gv.get("predicate")}
        extracted = set(r["extracted_facts"].keys())
        tp += len(gt_preds & extracted)
        fp += len(extracted - gt_preds)
        fn += len(gt_preds - extracted)

    if tp + fp + fn > 0:
        precision = tp / max(tp + fp, 1)
        recall    = tp / max(tp + fn, 1)
        f1        = 2 * precision * recall / max(precision + recall, 1e-10)
        print(f"\n  GT predicate extraction:")
        print(f"    Precision: {precision:.1%}  (extracted predicates that match GT)")
        print(f"    Recall:    {recall:.1%}  (GT predicates found by LLM)")
        print(f"    F1:        {f1:.3f}")

    print(f"{'='*68}\n")


with open("/kaggle/working/parsed_train.json") as f:
    parsed_train = json.load(f)
with open("/kaggle/working/parsed_test.json") as f:
    parsed_test  = json.load(f)

quality_report(parsed_train, "TRAIN")
quality_report(parsed_test,  "TEST")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 6 — Inspect individual extractions
# ─────────────────────────────────────────────────────────────────────────────

import random
random.seed(42)

print("="*60)
print("SAMPLE EXTRACTIONS — inspect LLM output quality")
print("="*60)

def show_extraction(record: dict) -> None:
    print(f"\n  {'─'*58}")
    print(f"  doc_id:  {record['doc_id']}")
    print(f"  domain:  {record['domain']}")
    print(f"  label:   {record['label']:<15}  verdict: {record['verdict']}")
    print(f"  n_facts: {record['n_facts']}  elapsed: {record['elapsed_s']}s")

    if record["extracted_facts"]:
        print(f"\n  Extracted facts:")
        for pred, val in sorted(record["extracted_facts"].items()):
            conf = record["fact_confidences"].get(pred, 0)
            print(f"    {pred:<40} = {str(val):<8}  conf={conf:.2f}")

    if record["ground_truth_violations"]:
        print(f"\n  Ground truth violations:")
        for gv in record["ground_truth_violations"]:
            gt_pred = gv.get("predicate", "?")
            found   = gt_pred in record["extracted_facts"]
            status  = "✓ extracted" if found else "✗ MISSING"
            print(f"    [{status}] {gt_pred}")
            print(f"      evidence: \"{gv.get('evidence_text', '')[:80]}\"")

    if record.get("failed_rules"):
        print(f"\n  Failed rules: {record['failed_rules']}")

# Show 2 NC, 2 C, 1 with GT match, 1 failure (if any)
nc_correct = [r for r in parsed_train
              if r["label"] == "NON_COMPLIANT" and r["verdict"] == "NON_COMPLIANT"]
c_correct  = [r for r in parsed_train
              if r["label"] == "COMPLIANT"     and r["verdict"] == "COMPLIANT"]
gt_match   = [r for r in parsed_train
              if r["ground_truth_violations"] and
              any(gv.get("predicate") in r["extracted_facts"]
                  for gv in r["ground_truth_violations"])]
failures   = [r for r in parsed_train if r["verdict"] == "ERROR"]

samples = []
if nc_correct: samples += random.sample(nc_correct, min(2, len(nc_correct)))
if c_correct:  samples += random.sample(c_correct,  min(2, len(c_correct)))
if gt_match:   samples += random.sample(gt_match,   min(1, len(gt_match)))
if failures:   samples += random.sample(failures,    min(1, len(failures)))

print(f"\nShowing {len(samples)} sample extractions:\n")
for rec in samples:
    show_extraction(rec)


# ─────────────────────────────────────────────────────────────────────────────
# CELL 7 — Save ILP-ready format
# ─────────────────────────────────────────────────────────────────────────────

def to_ilp_examples(parsed_records: list) -> list:
    """
    Convert parsed records to ILP training format.
    Each example: {doc_id, domain, label, facts: {pred: float}}
    Facts are already discrete (True/False from extraction).
    """
    examples = []
    for r in parsed_records:
        if r["verdict"] == "ERROR" or not r["extracted_facts"]:
            continue

        # Convert bool facts to float for ILP sparse dict
        facts_float = {}
        for pred, val in r["extracted_facts"].items():
            if isinstance(val, bool):
                facts_float[pred] = 1.0 if val else 0.0
            elif isinstance(val, (int, float)):
                facts_float[pred] = float(val)
            elif isinstance(val, str):
                facts_float[pred] = 1.0   # presence encoding

        examples.append({
            "doc_id":   r["doc_id"],
            "domain":   r["domain"],
            "label":    r["label"],
            "facts":    facts_float,
        })
    return examples


ilp_train = to_ilp_examples(parsed_train)
ilp_test  = to_ilp_examples(parsed_test)

with open("/kaggle/working/ilp_train_examples.json", "w") as f:
    json.dump(ilp_train, f, indent=2)
with open("/kaggle/working/ilp_test_examples.json", "w") as f:
    json.dump(ilp_test, f, indent=2)

print(f"ILP training examples: {len(ilp_train):,} / {len(parsed_train):,} "
      f"({len(ilp_train)/len(parsed_train):.1%} usable)")
print(f"ILP test examples:     {len(ilp_test):,} / {len(parsed_test):,} "
      f"({len(ilp_test)/len(parsed_test):.1%} usable)")

# Predicate coverage for ILP
pred_counts: Counter = Counter()
for ex in ilp_train:
    pred_counts.update(ex["facts"].keys())

print(f"\nTop predicates in ILP training set:")
print(f"  {'Predicate':<42} {'Count':>6} {'%':>6}")
print(f"  {'─'*56}")
for pred, count in pred_counts.most_common(15):
    pct = count / len(ilp_train)
    bar = "█" * int(pct * 15)
    print(f"  {pred:<42} {count:>6} {pct:>6.1%}  {bar}")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 8 — File summary
# ─────────────────────────────────────────────────────────────────────────────

print("\nOutput files:")
output_files = [
    "/kaggle/working/parsed_train.json",
    "/kaggle/working/parsed_test.json",
    "/kaggle/working/ilp_train_examples.json",
    "/kaggle/working/ilp_test_examples.json",
    "/kaggle/working/parse_failures_train.json",
    "/kaggle/working/parse_failures_test.json",
]
for path in output_files:
    if os.path.exists(path):
        size_kb = os.path.getsize(path) / 1024
        with open(path) as f:
            try:
                data = json.load(f)
                n = len(data)
            except Exception:
                n = "?"
        print(f"  ✓  {os.path.basename(path):<42} {size_kb:>8.0f} KB  ({n} records)")
    else:
        print(f"  —  {os.path.basename(path):<42} not created")

print()
print("Next step: use ilp_train_examples.json as input to the ILP training notebook")
print("           (kaggle_notebook_per_cell.py  Cell 13+)")
