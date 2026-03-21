"""
symbolic/ssce_algorithm.py

Logic-Grounded Pelican (LGP)
---------------------------------
Symbolic State Consistency Enforcement (SSCE)

FINAL STRICT VERSION
--------------------
Responsibilities:
- Detect symbolic drift
- Return structured DriftReport objects
- Raise deterministic enforcement errors
- DO NOT mutate symbolic table
- DO NOT swallow errors

Policy layer is responsible for:
- Logging drift reports
- Terminating pipeline
- Updating symbolic table after validation

Author: LGP Framework
"""

from __future__ import annotations

import logging
from typing import Dict, Any, List
from dataclasses import dataclass

from symbolic.table import get_symbolic_table


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("LGP.SSCE")
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Drift Report
# ---------------------------------------------------------------------------

@dataclass
class DriftReport:
    variable: str
    old_value: Any
    new_value: Any
    reason: str


# ---------------------------------------------------------------------------
# Custom Enforcement Exception
# ---------------------------------------------------------------------------

class SSCEEnforcementError(RuntimeError):
    def __init__(self, reports: List[DriftReport]):
        self.reports = reports
        message = "\n".join(r.reason for r in reports)
        super().__init__(f"SSCE Enforcement Failed:\n{message}")


# ---------------------------------------------------------------------------
# SSCE Engine
# ---------------------------------------------------------------------------

class SSCEEngine:
    """
    Strict symbolic invariant checker.

    Invariant (v1 strict mode):
    - New variable → allowed
    - Same value → allowed
    - Value change → drift
    """

    def __init__(self) -> None:
        self._table = get_symbolic_table()

    # ------------------------------------------------------------------
    # Drift Detection
    # ------------------------------------------------------------------

    def check_step(
        self,
        local_vars: Dict[str, Any],
    ) -> List[DriftReport]:
        """
        Detect drift without mutating state.
        """
        reports: List[DriftReport] = []

        for var_name, new_value in local_vars.items():
            record = self._table.get(var_name)

            # Case 1: New variable → safe
            if record is None:
                continue

            old_value = record.value

            # Case 2: Same value → safe
            if old_value == new_value:
                continue

            # Case 3: Value changed → DRIFT
            reason = (
                f"Symbolic Drift detected for '{var_name}': "
                f"{old_value} → {new_value} (redefinition without transformation)."
            )

            logger.warning(reason)

            reports.append(
                DriftReport(
                    variable=var_name,
                    old_value=old_value,
                    new_value=new_value,
                    reason=reason,
                )
            )

        return reports

    # ------------------------------------------------------------------
    # Strict Enforcement
    # ------------------------------------------------------------------

    def enforce(self, local_vars: Dict[str, Any]) -> None:
        """
        Raise SSCEEnforcementError if drift detected.
        Does NOT mutate symbolic table.
        """
        reports = self.check_step(local_vars)

        if reports:
            raise SSCEEnforcementError(reports)


# ---------------------------------------------------------------------------
# Convenience Accessor
# ---------------------------------------------------------------------------


def get_ssce_engine() -> SSCEEngine:
    return SSCEEngine()
