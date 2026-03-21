"""
core/state_manager.py

Logic-Grounded Pelican (LGP)
---------------------------------
6-Tuple State Manager (STRICT MODE)

This version removes internal drift swallowing.
SSCE enforcement errors are propagated upward.
Policy layer decides how to handle failures.

Author: LGP Framework
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from symbolic.table import get_symbolic_table


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("LGP.StateManager")
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# 6-Tuple State
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LGPState:
    q: str
    e: Dict[str, Any]
    o: Optional[Any]
    u: float
    f: bool
    h_q: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# State Manager (NO SSCE INSIDE)
# ---------------------------------------------------------------------------

class StateManager:
    """
    Deterministic state transition manager.

    IMPORTANT:
    - Does NOT perform SSCE enforcement.
    - Assumes enforcement already happened in policy layer.
    - Does NOT swallow errors.
    """

    def __init__(self) -> None:
        self._symbolic_table = get_symbolic_table()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self, query: str) -> LGPState:
        logger.info("Initializing LGP state.")
        return LGPState(
            q=query,
            e={},
            o=None,
            u=0.0,
            f=True,
            h_q=[query],
        )

    # ------------------------------------------------------------------
    # Deterministic Update (Pure)
    # ------------------------------------------------------------------

    def update_6tuple(
        self,
        current_state: LGPState,
        new_query: Optional[str] = None,
        new_evidence: Optional[Dict[str, Any]] = None,
        observation: Optional[Any] = None,
        utility_delta: float = 0.0,
        factual_status: Optional[bool] = None,
    ) -> LGPState:
        """
        Pure state update.
        Assumes evidence already validated.
        """

        new_evidence = new_evidence or {}

        # Commit to symbolic table (safe because validated upstream)
        for var, value in new_evidence.items():
            self._symbolic_table.set(var, value, current_state.q)

        updated_query = new_query if new_query else current_state.q
        updated_history = list(current_state.h_q)

        if new_query and new_query not in updated_history:
            updated_history.append(new_query)

        updated_utility = current_state.u + utility_delta
        updated_factual = factual_status if factual_status is not None else current_state.f

        return LGPState(
            q=updated_query,
            e=new_evidence,
            o=observation,
            u=updated_utility,
            f=updated_factual,
            h_q=updated_history,
        )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        logger.info("Resetting StateManager and SymbolicTable.")
        self._symbolic_table.clear()
