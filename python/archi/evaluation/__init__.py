"""Smoke evaluation harness for archi instances and bundles.

Executes a bundle's smoke question set (``eval/smoke.jsonl``) against an
instance's running MCP endpoint and writes an evaluation report (``report.md``).
"""
from __future__ import annotations

from archi.evaluation.runner import (
    EvalResult,
    SmokeItem,
    evaluate_instance,
    load_smoke_set,
)

__all__ = [
    "EvalResult",
    "SmokeItem",
    "evaluate_instance",
    "load_smoke_set",
]
