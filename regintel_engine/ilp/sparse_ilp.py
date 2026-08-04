"""
regintel_engine.ilp.sparse_ilp
================================
Domain-constrained sparse ILP for rule discovery.

Key design decisions (from architecture planning)
-------------------------------------------------
1. ILP trains on real extracted facts — not JEPA embeddings
   This guarantees discovered rules check real extractable fields

2. Domain constraint: predicates from different domains cannot
   appear in the same rule body. Prevents spurious correlations.
   e.g. GMP calibration + clinical trial enrollment = nonsense rule

3. Sparse representation: absent facts excluded from scoring entirely
   (not padded with 0.5). ILP only learns from facts actually present
   in each document, which varies by document type.

4. Gradient descent with L1 regularization to discover sparse rule bodies.
   Most compliance rules involve 1-3 predicates, not 50.

5. Human-readable output: predicate names come from the controlled vocabulary
   so discovered rules are immediately comprehensible.

Architecture of sparse ILP
--------------------------
Input:  N documents × M predicates (sparse — absent = not scored)
        N × 1 labels (0=NON_COMPLIANT, 1=COMPLIANT)

For each head (compliant, non_compliant):
  - Learn a weight vector W of shape (M,)
  - score(doc) = sum(W[i] * value[i] for i in present_predicates)
  - Rule fires when score > threshold
  - L1 regularization → sparse W → few predicates per rule

Extract rules from converged weights:
  - Top-k predicates per head = rule body
  - Confidence = precision on training set
  - Coverage = recall on positive class
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..models import HornClause
from ..vocabulary import PredicateVocabulary

log = logging.getLogger("regintel.ilp")


# ─────────────────────────────────────────────────────────────────────────────
#  ILP CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ILPConfig:
    n_epochs:          int   = 200
    learning_rate:     float = 0.05
    l1_lambda:         float = 0.01   # sparsity regularization
    threshold:         float = 0.0    # decision boundary (unused — kept for API compat)
    top_k_predicates:  int   = 3      # max predicates per rule body
    min_confidence:    float = 0.75   # minimum rule precision
    min_coverage:      float = 0.10   # minimum recall
    min_support:       int   = 5      # minimum supporting documents
    score_scale:       float = 1.0    # sigmoid input scale factor.
    # score_scale > 1 → steeper sigmoid (more decisive).
    # score_scale < 1 → shallower sigmoid (softer gradients).
    # Default 1.0 = standard logistic regression.


# ─────────────────────────────────────────────────────────────────────────────
#  SPARSE ILP
# ─────────────────────────────────────────────────────────────────────────────

class SparseILP:
    """
    Domain-constrained sparse ILP for compliance rule discovery.

    Trains on (sparse_fact_dict, label) pairs.
    Discovers Horn clauses within a single regulatory domain.
    Produces human-readable rules using controlled vocabulary names.

    Pure Python/stdlib — no torch dependency.
    Uses simple gradient descent with sigmoid activation.
    """

    def __init__(
        self,
        vocab:  PredicateVocabulary,
        config: Optional[ILPConfig] = None,
    ) -> None:
        self.vocab  = vocab
        self.config = config or ILPConfig()

    def induce(
        self,
        examples: List[Dict[str, Any]],
        domain:   str,
    ) -> List[HornClause]:
        """
        Discover compliance rules from labeled examples.

        Parameters
        ----------
        examples  List of {"facts": sparse_dict, "label": "COMPLIANT"|"NON_COMPLIANT"}
        domain    Regulatory domain to constrain predicate vocabulary

        Returns
        -------
        List of HornClause objects (pending SME approval)
        """
        if not examples:
            return []

        # Get domain-specific predicates
        domain_preds = self.vocab.get_names(domain)
        if not domain_preds:
            log.warning(f"[ILP] No predicates for domain: {domain}")
            return []

        # Build data matrix
        X, y, pred_names = self._build_matrix(examples, domain_preds)

        if len(X) < self.config.min_support * 2:
            log.warning(f"[ILP] Too few examples ({len(X)}) for domain {domain}")
            return []

        n_pos = sum(1 for yi in y if yi == 1)
        n_neg = len(y) - n_pos
        log.info(f"[ILP] domain={domain} examples={len(X)} "
                 f"preds={len(pred_names)} pos={n_pos} neg={n_neg}")

        clauses = []

        # Discover rules for both heads
        for head, target_label in [("non_compliant", 0), ("compliant", 1)]:
            head_clauses = self._discover_for_head(
                X, y, pred_names, head, target_label, domain)
            clauses.extend(head_clauses)

        log.info(f"[ILP] domain={domain} discovered {len(clauses)} clauses")
        return clauses

    def _build_matrix(
        self,
        examples:     List[Dict[str, Any]],
        domain_preds: List[str],
    ) -> Tuple[List[List[Tuple[int, float]]], List[int], List[str]]:
        """
        Build sparse matrix representation.
        Each row is a list of (predicate_index, value) for present predicates.
        """
        # Only include predicates that appear in at least 2 examples
        pred_counts: Dict[str, int] = {}
        for ex in examples:
            for pred in ex["facts"]:
                if pred in domain_preds:
                    pred_counts[pred] = pred_counts.get(pred, 0) + 1

        active_preds = [p for p, c in pred_counts.items()
                        if c >= max(2, self.config.min_support)]

        if not active_preds:
            # Fall back to all domain predicates
            active_preds = domain_preds

        pred_idx = {p: i for i, p in enumerate(active_preds)}

        X = []
        y = []
        for ex in examples:
            row = []
            for pred, val in ex["facts"].items():
                if pred in pred_idx and isinstance(val, (int, float)):
                    row.append((pred_idx[pred], float(val)))
            if row:
                X.append(row)
                y.append(1 if ex["label"] == "COMPLIANT" else 0)

        return X, y, active_preds

    def _discover_for_head(
        self,
        X:            List[List[Tuple[int, float]]],
        y:            List[int],
        pred_names:   List[str],
        head:         str,
        target_label: int,
        domain:       str,
    ) -> List[HornClause]:
        """Run gradient descent to find predicates predicting target_label."""
        n_preds = len(pred_names)
        n_docs  = len(X)

        if n_preds == 0:
            return []

        # Initialize weights
        weights = [random.gauss(0, 0.01) for _ in range(n_preds)]

        cfg = self.config

        for epoch in range(cfg.n_epochs):
            grad = [0.0] * n_preds
            loss = 0.0

            for i, (row, label) in enumerate(zip(X, y)):
                # Forward pass: score = sum(w[j] * x[j]) for present predicates
                score = sum(weights[j] * v for j, v in row)
                score = score * cfg.score_scale

                # Sigmoid (numerically stable)
                if score >= 0:
                    sig = 1.0 / (1.0 + math.exp(-min(score, 500)))
                else:
                    ex  = math.exp(min(score, 500))
                    sig = ex / (1.0 + ex)

                target = float(label == target_label)
                error  = sig - target

                # Clamp for stability
                error = max(-10.0, min(10.0, error))

                loss += -(target * math.log(max(sig, 1e-10)) +
                          (1 - target) * math.log(max(1 - sig, 1e-10)))

                # Backward pass — gradient w.r.t. weights
                for j, v in row:
                    grad[j] += error * v * cfg.score_scale

            # L1 regularization gradient
            for j in range(n_preds):
                if weights[j] > 0:
                    grad[j] += cfg.l1_lambda
                elif weights[j] < 0:
                    grad[j] -= cfg.l1_lambda

            # Update weights
            lr = cfg.learning_rate / max(1, epoch // 50 + 1)
            for j in range(n_preds):
                weights[j] -= lr * grad[j] / n_docs

        # Extract top-k predicates by absolute weight
        indexed = sorted(
            enumerate(weights),
            key=lambda x: -abs(x[1]),
        )

        # Both heads: predicates that predict the target have POSITIVE weights.
        #
        # Why? For target_label=0 (non_compliant):
        #   NC doc → target=1.0, error = sig-1.0 < 0 → negative grad
        #   → weight INCREASES toward positive territory
        # For target_label=1 (compliant):
        #   C doc → target=1.0, same logic → weight INCREASES
        # In both cases the predicates correlated with the target
        # end up with POSITIVE weights after training.
        top_k = [(j, w) for j, w in indexed
                 if w > 0.05][:cfg.top_k_predicates]

        if not top_k:
            return []

        body_preds = [pred_names[j] for j, _ in top_k]

        # Check domain constraint
        if not self.vocab.check_cross_domain(body_preds):
            log.debug(f"[ILP] Cross-domain rule rejected: {body_preds}")
            return []

        # Evaluate rule on training data
        confidence, coverage, support = self._evaluate_rule(
            body_preds, head, target_label, X, y, pred_names)

        if (confidence < cfg.min_confidence
                or coverage < cfg.min_coverage
                or support < cfg.min_support):
            return []

        import uuid
        clause_id = (f"HC-{domain[:3].upper()}-"
                     f"{uuid.uuid4().hex[:8].upper()}")

        return [HornClause(
            clause_id  = clause_id,
            head       = head,
            body       = body_preds,
            confidence = round(confidence, 3),
            coverage   = round(coverage, 3),
            support    = support,
            domain     = domain,
        )]

    def _evaluate_rule(
        self,
        body_preds:   List[str],
        head:         str,
        target_label: int,
        X:            List[List[Tuple[int, float]]],
        y:            List[int],
        pred_names:   List[str],
    ) -> Tuple[float, float, int]:
        """
        Compute precision, recall, and support for a rule body.
        Returns (confidence, coverage, support).
        """
        pred_set = set(pred_names.index(p) for p in body_preds
                       if p in pred_names)
        if not pred_set:
            return 0.0, 0.0, 0

        tp = fp = fn = 0

        for row, label in zip(X, y):
            present_idx = {j for j, _ in row}
            # AND logic: rule fires only when ALL body predicates are present.
            # intersection() gives OR — issubset() gives AND (correct).
            fires = pred_set.issubset(present_idx)

            if fires and label == target_label:
                tp += 1
            elif fires and label != target_label:
                fp += 1
            elif not fires and label == target_label:
                fn += 1

        if tp + fp == 0:
            return 0.0, 0.0, 0

        precision = tp / (tp + fp)
        recall    = tp / max(tp + fn, 1)
        support   = tp

        return precision, recall, support


# ─────────────────────────────────────────────────────────────────────────────
#  DREAMCODER
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Primitive:
    """A reusable abstraction promoted from recurring predicate combinations."""
    name:        str
    predicates:  List[str]
    domain:      str
    frequency:   int
    description: str = ""


class DreamCoder:
    """
    Abstracts recurring predicate combinations into reusable primitives.

    Counts how often predicate combinations co-occur across discovered rules.
    Promotes high-frequency combinations to primitives.
    Future ILP cycles can use primitives as single units.

    Example:
      api_coa_missing + deviation_without_capa appear together in 31 rules
      → promoted to primitive: gmp_documentation_failure
      Future rules: "non_compliant :- gmp_documentation_failure"
      More expressive, fewer terms, same meaning.
    """

    MIN_FREQUENCY = 3   # minimum co-occurrences to promote to primitive

    def __init__(self) -> None:
        self.primitives: Dict[str, Primitive] = {}
        self.library:    List[Primitive]      = []

    def wake(self, clauses: List[HornClause]) -> List[Primitive]:
        """
        Wake phase: find recurring predicate combinations across rules.
        Returns newly discovered primitives.
        """
        # Count co-occurrences of predicate pairs across rule bodies
        pair_counts: Dict[Tuple, int] = {}
        for clause in clauses:
            body = sorted(clause.body)
            for i in range(len(body)):
                for j in range(i + 1, len(body)):
                    pair = (body[i], body[j], clause.domain)
                    pair_counts[pair] = pair_counts.get(pair, 0) + 1

        # Also count single-predicate frequencies
        single_counts: Dict[Tuple, int] = {}
        for clause in clauses:
            for pred in clause.body:
                key = (pred, clause.domain)
                single_counts[key] = single_counts.get(key, 0) + 1

        new_primitives = []

        # Promote pairs that co-occur frequently
        for (p1, p2, domain), count in pair_counts.items():
            if count >= self.MIN_FREQUENCY:
                prim_name = f"{domain[:3]}_{'_'.join(sorted([p1, p2]))[:40]}"
                if prim_name not in self.primitives:
                    prim = Primitive(
                        name       = prim_name,
                        predicates = [p1, p2],
                        domain     = domain,
                        frequency  = count,
                        description = f"Co-occurring {domain} pattern: {p1} + {p2}",
                    )
                    self.primitives[prim_name] = prim
                    self.library.append(prim)     # FIX: also populate library
                    new_primitives.append(prim)

        # Promote single predicates that appear in many rules
        for (pred, domain), count in single_counts.items():
            if count >= self.MIN_FREQUENCY * 2:
                prim_name = f"{domain[:3]}_{pred}"
                if prim_name not in self.primitives:
                    prim = Primitive(
                        name        = prim_name,
                        predicates  = [pred],
                        domain      = domain,
                        frequency   = count,
                        description = f"High-frequency {domain} predicate: {pred}",
                    )
                    self.primitives[prim_name] = prim
                    self.library.append(prim)     # FIX: also populate library
                    new_primitives.append(prim)

        log.info(f"[DreamCoder] wake: {len(new_primitives)} new primitives "
                 f"(total {len(self.primitives)})")
        return new_primitives

    def sleep(self, clauses: List[HornClause]) -> List[HornClause]:
        """
        Sleep phase: rewrite rule bodies using primitive names.
        More expressive rules with fewer terms.
        """
        if not self.primitives:
            return clauses

        compressed = []
        for clause in clauses:
            new_body = self._compress_body(clause.body, clause.domain)
            if new_body != clause.body:
                log.debug(f"[DreamCoder] compressed {clause.body} → {new_body}")
            new_clause           = HornClause(
                clause_id  = clause.clause_id,
                head       = clause.head,
                body       = new_body,
                confidence = clause.confidence,
                coverage   = clause.coverage,
                support    = clause.support,
                domain     = clause.domain,
            )
            compressed.append(new_clause)

        return compressed

    def _compress_body(self, body: List[str], domain: str) -> List[str]:
        """Replace predicate pairs with primitive names where available."""
        body_set = set(body)
        result   = list(body)

        for prim in self.primitives.values():
            if prim.domain != domain:
                continue
            if all(p in body_set for p in prim.predicates):
                # Replace predicates with primitive
                result = [p for p in result if p not in prim.predicates]
                result.append(prim.name)
                body_set -= set(prim.predicates)
                body_set.add(prim.name)
                break   # one substitution at a time

        return result
