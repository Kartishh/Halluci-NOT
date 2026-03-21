"""
data/logger.py

Logic-Grounded Pelican (LGP)
---------------------------------
Semantic Gradient & Trace Logger

This module provides deterministic logging and audit trace generation for:

- Reflexion cycles
- Drift reports
- PoT execution logs
- Agreement model scores
- Final SymbolicTable snapshot

Design Guarantees:
- Deterministic structured JSON logs
- Append-only semantic trace history
- Separation between runtime logging and audit artifacts
- Production-safe serialization

Author: LGP Framework
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from symbolic.table import get_symbolic_table
from symbolic.ssce_algorithm import DriftReport


# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------

logger = logging.getLogger("LGP.SemanticLogger")
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Semantic Gradient Logger
# ---------------------------------------------------------------------------

class SemanticLogger:
    """
    Append-only structured logger for LGP reasoning traces.

    This logger does NOT replace Python logging.
    It creates structured trace artifacts required by LGP's
    traceability and evaluation standard.
    """

    def __init__(self) -> None:
        self._trace: List[Dict[str, Any]] = []
        self._pot_logs: List[Dict[str, Any]] = []
        self._agreement_scores: List[float] = []
        self._drift_reports: List[Dict[str, Any]] = []
        self._latency_log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Semantic Gradient Logging
    # ------------------------------------------------------------------

    def log_step(
        self,
        step_id: str,
        state_snapshot: Dict[str, Any],
        message: Optional[str] = None,
    ) -> None:
        """
        Record a reasoning step snapshot.
        """
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "step_id": step_id,
            "state": state_snapshot,
            "message": message,
        }
        self._trace.append(entry)

    # ------------------------------------------------------------------
    # Drift Reporting
    # ------------------------------------------------------------------

    def log_drift(self, reports: List[DriftReport]) -> None:
        """
        Persist structured drift reports.
        """
        for r in reports:
            self._drift_reports.append(
                {
                    "variable": r.variable,
                    "old_value": r.old_value,
                    "new_value": r.new_value,
                    "justified": r.justified,
                    "reason": r.reason,
                }
            )
    def log_drift_report(self, report: Dict[str, Any]) -> None:
        """
        Log structured drift report for audit.
        """
        self._drift_reports.append(report)

    # ------------------------------------------------------------------
    # PoT Execution Logging
    # ------------------------------------------------------------------

    def log_pot_execution(
        self,
        script: str,
        output: Any,
        success: bool,
        error: Optional[str] = None,
    ) -> None:
        """
        Log Program-of-Thought execution result.
        """
        self._pot_logs.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "script": script,
                "output": output,
                "success": success,
                "error": error,
            }
        )

    # ------------------------------------------------------------------
    # Agreement Model Logging
    # ------------------------------------------------------------------

    def log_agreement_score(self, score: float) -> None:
        """
        Record NLI agreement score.
        """
        self._agreement_scores.append(score)

    # ------------------------------------------------------------------
    # Latency Logging
    # ------------------------------------------------------------------

    def log_latency(self, stage: str, latency_ms: float) -> None:
        """
        Record per-stage latency in milliseconds.

        Args:
            stage: Pipeline stage name (e.g., 'decomposition', 'sandbox_execution')
            latency_ms: Time in milliseconds
        """
        self._latency_log.append({
            "stage": stage,
            "latency_ms": round(latency_ms, 2),
            "timestamp": datetime.utcnow().isoformat(),
        })

    # ------------------------------------------------------------------
    # Final Audit Artifact
    # ------------------------------------------------------------------

    def build_final_report(self) -> Dict[str, Any]:
        """
        Generate deterministic final audit JSON.

        Required by LGP Code Quality Standard.
        Includes aggregate statistics for evaluation.
        """
        symbolic_snapshot = get_symbolic_table().snapshot()

        # Aggregate latency stats
        latency_stats = {}
        if self._latency_log:
            from collections import defaultdict
            by_stage: Dict[str, List[float]] = defaultdict(list)
            for entry in self._latency_log:
                by_stage[entry["stage"]].append(entry["latency_ms"])
            for stage, times in by_stage.items():
                latency_stats[stage] = {
                    "count": len(times),
                    "total_ms": round(sum(times), 2),
                    "avg_ms": round(sum(times) / len(times), 2),
                    "max_ms": round(max(times), 2),
                }

        total_latency = sum(e["latency_ms"] for e in self._latency_log)

        return {
            "symbolic_table": symbolic_snapshot,
            "semantic_trace": self._trace,
            "drift_reports": self._drift_reports,
            "pot_execution_log": self._pot_logs,
            "agreement_scores": self._agreement_scores,
            "latency_log": self._latency_log,
            "aggregate": {
                "drift_frequency": len(self._drift_reports),
                "reasoning_depth": len(self._pot_logs),
                "total_latency_ms": round(total_latency, 2),
                "latency_by_stage": latency_stats,
            },
        }

    def export_json(self, filepath: str) -> None:
        """
        Persist final audit report to disk.
        """
        try:
            report = self.build_final_report()
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            logger.info(f"Audit report exported to {filepath}")
        except Exception as e:
            logger.error(f"Failed to export audit report: {str(e)}")
            raise

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """
        Clear all trace artifacts.
        """
        self._trace.clear()
        self._pot_logs.clear()
        self._agreement_scores.clear()
        self._drift_reports.clear()
        self._latency_log.clear()


# ---------------------------------------------------------------------------
# Convenience Accessor
# ---------------------------------------------------------------------------


def get_semantic_logger() -> SemanticLogger:
    """
    Public accessor for SemanticLogger.
    """
    return SemanticLogger()