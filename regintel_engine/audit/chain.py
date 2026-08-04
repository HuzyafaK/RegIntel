"""
regintel_engine.audit.chain
=============================
HMAC-SHA256 tamper-evident audit chain.

21 CFR Part 11 compliance properties:
  - Every record sealed with HMAC-SHA256
  - Each record includes hash of previous record (chain link)
  - Any modification to any record breaks all subsequent signatures
  - Full provenance: document → facts → rules → verdict
  - Extracted_text included for every fact (enables independent verification)
  - Regulatory citations included for every rule result
  - SME approval record included for all ILP rules

The audit record is formatted in regulatory language.
Every regulator can read it and verify independently.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..models import AuditRecord, RuleResult, Verdict

log = logging.getLogger("regintel.audit")


class HMACauditChain:
    """
    Tamper-evident HMAC-SHA256 audit chain.

    Each record:
      1. Built from verdict + facts + rule_results
      2. Previous record hash included
      3. HMAC-SHA256 signature computed
      4. Appended to in-memory chain and optionally saved to disk

    Usage
    -----
        chain = HMACauditChain(key_env="REGINTEL_AUDIT_KEY")
        record = chain.seal(verdict)
        chain.save("audit_log.jsonl")

        # Verify integrity
        ok, issues = chain.verify()
    """

    SYSTEM_VERSION = "regintel-1.0.0"

    def __init__(
        self,
        key_env:  str         = "REGINTEL_AUDIT_KEY",
        key_hex:  Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        key_env   Environment variable containing the HMAC key (hex)
        key_hex   Direct key override (for testing only)
        """
        if key_hex:
            self._key = bytes.fromhex(key_hex)
        else:
            raw = os.environ.get(key_env, "")
            if not raw:
                # Generate deterministic key from env name for development
                raw = hashlib.sha256(key_env.encode()).hexdigest()
                log.warning(f"[Audit] {key_env} not set — using derived key. "
                            f"Set the environment variable for production.")
            try:
                self._key = bytes.fromhex(raw)
            except ValueError:
                self._key = hashlib.sha256(raw.encode()).digest()

        self._records: List[AuditRecord] = []
        self._prev_hash: str = "0" * 64   # genesis block

    def seal(self, verdict: Verdict) -> AuditRecord:
        """
        Seal a verdict into a tamper-evident audit record.
        Returns the signed AuditRecord.
        """
        # Serialise facts (with provenance)
        extracted_facts = self._serialise_facts(verdict.facts)

        # Serialise rule results (with citations)
        rule_results = self._serialise_rules(verdict.rule_results)

        # Compute hashes
        doc_hash   = verdict.doc_hash
        facts_hash = hashlib.sha256(
            json.dumps(verdict.facts.facts, sort_keys=True, default=str).encode()
        ).hexdigest()

        record_id  = (f"REC-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
                      f"-{doc_hash[:8]}")

        # Build record payload (without signature)
        payload = {
            "record_id":       record_id,
            "doc_hash":        doc_hash,
            "facts_hash":      facts_hash,
            "verdict":         verdict.verdict,
            "domain":          verdict.domain,
            "timestamp":       verdict.timestamp,
            "system_version":  self.SYSTEM_VERSION,
            "extracted_facts": extracted_facts,
            "rule_results":    rule_results,
            "prev_hash":       self._prev_hash,
        }

        # Compute record hash
        payload_json = json.dumps(payload, sort_keys=True, default=str)
        record_hash  = hashlib.sha256(payload_json.encode()).hexdigest()

        # Compute HMAC signature
        signature = hmac.new(
            self._key,
            payload_json.encode(),
            hashlib.sha256,
        ).hexdigest()

        record = AuditRecord(
            record_id       = record_id,
            doc_hash        = doc_hash,
            facts_hash      = facts_hash,
            verdict         = verdict.verdict,
            domain          = verdict.domain,
            timestamp       = verdict.timestamp,
            system_version  = self.SYSTEM_VERSION,
            extracted_facts = extracted_facts,
            rule_results    = rule_results,
            prev_hash       = self._prev_hash,
            record_hash     = record_hash,
            signature       = signature,
        )

        self._prev_hash = record_hash
        self._records.append(record)

        log.debug(f"[Audit] sealed {record_id}: {verdict.verdict}")
        return record

    def verify(self) -> tuple:
        """
        Verify integrity of the entire audit chain.
        Returns (all_ok: bool, issues: List[str]).
        """
        issues   = []
        prev_hash = "0" * 64

        for record in self._records:
            # Rebuild payload without signature
            payload = {
                "record_id":       record.record_id,
                "doc_hash":        record.doc_hash,
                "facts_hash":      record.facts_hash,
                "verdict":         record.verdict,
                "domain":          record.domain,
                "timestamp":       record.timestamp,
                "system_version":  record.system_version,
                "extracted_facts": record.extracted_facts,
                "rule_results":    record.rule_results,
                "prev_hash":       record.prev_hash,
            }
            payload_json = json.dumps(payload, sort_keys=True, default=str)

            # Check chain link
            if record.prev_hash != prev_hash:
                issues.append(f"{record.record_id}: chain link broken")

            # Check record hash
            expected_hash = hashlib.sha256(payload_json.encode()).hexdigest()
            if record.record_hash != expected_hash:
                issues.append(f"{record.record_id}: record hash mismatch — tampered")

            # Check HMAC signature
            expected_sig = hmac.new(
                self._key,
                payload_json.encode(),
                hashlib.sha256,
            ).hexdigest()
            if record.signature != expected_sig:
                issues.append(f"{record.record_id}: HMAC signature invalid — tampered")

            prev_hash = record.record_hash

        all_ok = len(issues) == 0
        return all_ok, issues

    def save(self, path: str) -> None:
        """Save audit chain to JSONL file."""
        with open(path, "w") as f:
            for record in self._records:
                f.write(json.dumps(record.to_dict(), default=str) + "\n")
        log.info(f"[Audit] saved {len(self._records)} records → {path}")

    @classmethod
    def load(cls, path: str, key_env: str = "REGINTEL_AUDIT_KEY") -> "HMACauditChain":
        """Load audit chain from JSONL file."""
        chain = cls(key_env=key_env)
        chain._records = []

        with open(path) as f:
            for line in f:
                data = json.loads(line.strip())
                record = AuditRecord(**data)
                chain._records.append(record)
                chain._prev_hash = record.record_hash

        log.info(f"[Audit] loaded {len(chain._records)} records from {path}")
        return chain

    def format_record(self, record: AuditRecord) -> str:
        """Format a record in human-readable regulatory language."""
        lines = [
            "═" * 70,
            "REGULATORY COMPLIANCE ASSESSMENT RECORD",
            f"System: RegIntel {record.system_version}",
            "═" * 70,
            f"Record ID:   {record.record_id}",
            f"Document:    {record.doc_hash[:16]}...",
            f"Domain:      {record.domain}",
            f"Date:        {record.timestamp[:19]} UTC",
            f"Verdict:     {record.verdict}",
            "",
        ]

        # Failed rules
        failed = [r for r in record.rule_results if r.get("passed") is False]
        if failed:
            lines.append("FINDINGS:")
            for result in failed:
                lines.extend([
                    f"  {'─'*60}",
                    f"  Rule:        {result['rule_id']} ({result['severity'].upper()})",
                    f"  Finding:     {result['message']}",
                    f"  Regulation:  {result['regulation']}",
                    f"  Source:      {result['rule_source']}",
                ])
                if result.get("evidence"):
                    for pred, ev in result["evidence"].items():
                        if ev.get("value") is not None:
                            lines.append(
                                f"  Evidence:    {pred} = {ev['value']}"
                                f"  ← \"{ev.get('extracted_text', '')[:80]}\""
                            )
            lines.append("")

        # Audit integrity
        lines.extend([
            "AUDIT INTEGRITY:",
            f"  Document hash:  {record.doc_hash[:32]}...",
            f"  Record hash:    {record.record_hash[:32]}...",
            f"  HMAC signature: {record.signature[:32]}...",
            f"  Chain link:     {record.prev_hash[:16]}...",
            "═" * 70,
        ])

        return "\n".join(lines)

    # ── Serialisation helpers ─────────────────────────────────────────────────

    @staticmethod
    def _serialise_facts(facts) -> List[Dict]:
        """Serialise FactDict with full provenance."""
        result = []
        for name, value in facts.facts.items():
            prov = facts.provenance.get(name)
            result.append({
                "name":           name,
                "value":          value,
                "extracted_text": prov.extracted_text if prov else "",
                "confidence":     prov.confidence    if prov else 0.0,
                "reasoning":      prov.reasoning     if prov else "",
                "source_type":    prov.source_type   if prov else "structured",
            })
        for name in facts.absent_facts:
            result.append({
                "name":           name,
                "value":          None,
                "extracted_text": "",
                "confidence":     0.0,
                "reasoning":      "absent or low confidence",
                "source_type":    "absent",
            })
        return result

    @staticmethod
    def _serialise_rules(rule_results: List[RuleResult]) -> List[Dict]:
        """Serialise RuleResults with citations."""
        result = []
        for rr in rule_results:
            result.append({
                "rule_id":         rr.rule_id,
                "rule_name":       rr.rule_name,
                "domain":          rr.domain,
                "severity":        rr.severity,
                "passed":          rr.passed,
                "message":         rr.message,
                "regulation":      rr.regulation,
                "regulation_text": rr.regulation_text,
                "regulation_url":  rr.regulation_url,
                "evidence":        rr.evidence,
                "rule_source":     rr.rule_source,
                "confidence":      rr.confidence,
                "approved_by":     rr.approved_by,
                "approved_on":     rr.approved_on,
            })
        return result
