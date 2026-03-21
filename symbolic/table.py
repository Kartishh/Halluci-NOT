"""
symbolic/table.py

Logic-Grounded Pelican (LGP)
---------------------------------
Global Symbolic Table Singleton

This module provides a deterministic, thread-safe symbolic variable store
used by the SSCE algorithm to detect Symbolic Drift across reasoning steps.

Design Goals:
- Deterministic behavior
- Thread safety
- Explicit variable lifecycle tracking
- Alias resolution support
- Full traceability for audit logs

Author: LGP Framework
"""

from __future__ import annotations

import threading
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime


# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------

logger = logging.getLogger("LGP.SymbolicTable")
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SymbolRecord:
    """
    Immutable record representing a variable in the Symbolic Table.

    Attributes:
        name: Canonical variable name
        value: Deterministic evaluated value
        timestamp: Last update time (UTC)
        source_step: Reasoning step identifier
    """
    name: str
    value: Any
    timestamp: datetime
    source_step: str


# ---------------------------------------------------------------------------
# Singleton Implementation
# ---------------------------------------------------------------------------

class SymbolicTable:
    """
    Thread-safe Singleton for managing symbolic variable state.

    Guarantees:
        - Single global instance
        - Atomic read/write operations
        - Deterministic update semantics
    """

    _instance: Optional["SymbolicTable"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "SymbolicTable":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SymbolicTable, cls).__new__(cls)
                cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """Initialize internal state containers."""
        self._table: Dict[str, SymbolRecord] = {}
        self._alias_map: Dict[str, str] = {}
        self._rw_lock = threading.RLock()
        logger.info("SymbolicTable initialized.")

    # ------------------------------------------------------------------
    # Core CRUD Operations
    # ------------------------------------------------------------------

    def set(
        self,
        name: str,
        value: Any,
        source_step: str,
    ) -> None:
        """
        Insert or update a variable in the symbolic table.

        Args:
            name: Variable name
            value: Deterministic value
            source_step: Step identifier
        """
        if not name:
            raise ValueError("Variable name cannot be empty.")

        canonical = self._resolve_alias(name)

        with self._rw_lock:
            record = SymbolRecord(
                name=canonical,
                value=value,
                timestamp=datetime.utcnow(),
                source_step=source_step,
            )
            self._table[canonical] = record
            logger.debug(f"Set variable '{canonical}' = {value}")

    def get(self, name: str) -> Optional[SymbolRecord]:
        """
        Retrieve a variable record.

        Args:
            name: Variable name

        Returns:
            SymbolRecord or None
        """
        if not name:
            return None

        canonical = self._resolve_alias(name)

        with self._rw_lock:
            return self._table.get(canonical)

    def exists(self, name: str) -> bool:
        """Check if variable exists in table."""
        return self.get(name) is not None

    def delete(self, name: str) -> None:
        """Remove variable from table."""
        canonical = self._resolve_alias(name)
        with self._rw_lock:
            if canonical in self._table:
                del self._table[canonical]
                logger.debug(f"Deleted variable '{canonical}'")

    def clear(self) -> None:
        """Clear entire symbolic table."""
        with self._rw_lock:
            self._table.clear()
            self._alias_map.clear()
            logger.info("SymbolicTable cleared.")

    # ------------------------------------------------------------------
    # Alias Handling
    # ------------------------------------------------------------------

    def register_alias(self, alias: str, canonical_name: str) -> None:
        """
        Register an alias for an existing canonical variable.

        Example:
            register_alias("total_price", "cost")
        """
        if not alias or not canonical_name:
            raise ValueError("Alias and canonical name must be non-empty.")

        with self._rw_lock:
            self._alias_map[alias] = canonical_name
            logger.debug(f"Alias registered: {alias} -> {canonical_name}")

    def _resolve_alias(self, name: str) -> str:
        """Resolve alias to canonical variable name."""
        return self._alias_map.get(name, name)

    # ------------------------------------------------------------------
    # Inspection & Traceability
    # ------------------------------------------------------------------

    def snapshot(self) -> Dict[str, Dict[str, Any]]:
        """
        Return a serializable snapshot of the current symbolic table.

        Used for final traceability JSON block.
        """
        with self._rw_lock:
            return {
                name: {
                    "value": record.value,
                    "timestamp": record.timestamp.isoformat(),
                    "source_step": record.source_step,
                }
                for name, record in self._table.items()
            }

    def diff(
        self,
        name: str,
        new_value: Any,
    ) -> Tuple[bool, Optional[Any]]:
        """
        Compare new value with stored value.

        Returns:
            (is_different, old_value)
        """
        record = self.get(name)
        if record is None:
            return False, None

        return record.value != new_value, record.value


# ---------------------------------------------------------------------------
# Convenience Accessor
# ---------------------------------------------------------------------------

def get_symbolic_table() -> SymbolicTable:
    """
    Public accessor to enforce Singleton usage pattern.
    """
    return SymbolicTable()
