"""RegIntel Neuro-Symbolic Compliance Engine v1.0"""
__version__ = "1.0.0"

from .pipeline import RegIntelPipeline, PipelineResult
from .models import (
    ExtractedFact, FactDict, RuleResult, Verdict,
    ComplianceRule, HornClause, AuditRecord,
)
from .vocabulary import PredicateVocabulary, EncodingTables

__all__ = [
    "RegIntelPipeline", "PipelineResult",
    "ExtractedFact", "FactDict", "RuleResult", "Verdict",
    "ComplianceRule", "HornClause", "AuditRecord",
    "PredicateVocabulary", "EncodingTables",
]
