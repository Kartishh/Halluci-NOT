"""
verifier/numeric_nli.py

HalluciNOT (LGP)
---------------------------------
Numeric Consistency Gate (Process Guard)

Purpose:
    Ensure that the Symbolic Result (sandbox output) matches
    the Textual Claim in BOTH value AND entity.

    This replaces the blanket NLI skip for numeric outputs.
    Without this gate, the system is just a calculator — it verifies
    that code *ran*, not that the answer is *correct*.

How it works:
    1. Extract all numeric claims from the query text
       (regex for numbers + Gemini for entity-value associations)
    2. Compare claimed values against SymbolicTable values
    3. Flag mismatches using math.isclose (handles rounding)
    4. Return detailed NumericContradiction reports

Example scenario:
    Query: "The total is 150 apples. price = 5. total = price * 3"

    Step 1: Extract claims → {"total": 150}
    Step 2: Sandbox computes → {"total": 15}
    Step 3: Gate checks → 150 != 15 → CONTRADICTION
    Step 4: Report: "Variable 'total' claimed as 150 but computed as 15"

Author: LGP Framework
"""

from __future__ import annotations

import math
import re
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from symbolic.table import get_symbolic_table

logger = logging.getLogger("LGP.NumericNLI")
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Result Schema
# ---------------------------------------------------------------------------

@dataclass
class NumericClaim:
    """A numeric value claimed in the text."""
    entity: str          # Variable or entity name
    claimed_value: float # The number stated in text
    source_text: str     # The text fragment where it was found


@dataclass
class NumericContradiction:
    """A mismatch between claimed and computed values."""
    variable: str
    claimed_value: float
    computed_value: float
    relative_error: float
    source_text: str
    reason: str


@dataclass
class NumericConsistencyResult:
    """Full result of numeric consistency checking."""
    is_consistent: bool
    claims_found: List[NumericClaim]
    contradictions: List[NumericContradiction]
    checked_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_consistent": self.is_consistent,
            "claims_found": len(self.claims_found),
            "contradictions": [
                {
                    "variable": c.variable,
                    "claimed": c.claimed_value,
                    "computed": c.computed_value,
                    "relative_error": round(c.relative_error, 4),
                    "reason": c.reason,
                }
                for c in self.contradictions
            ],
            "checked_count": self.checked_count,
        }


# ---------------------------------------------------------------------------
# Numeric Claim Extraction
# ---------------------------------------------------------------------------

# Patterns for entity-value associations in natural language
_CLAIM_PATTERNS = [
    # "the total is 150" / "total equals 150"
    r"(?:the\s+)?(\w+(?:_\w+)*)\s+(?:is|are|equals?|was|were|=)\s+([-+]?\d[\d,]*\.?\d*)",
    # "150 apples" / "150 total"
    r"([-+]?\d[\d,]*\.?\d*)\s+(\w+(?:_\w+)*)",
    # "there are 150" / "she has 150"
    r"(?:there\s+(?:are|is|were|was)|(?:he|she|it|they)\s+(?:has|have|had|earns?|makes?|gets?|pays?))\s+([-+]?\d[\d,]*\.?\d*)",
    # "$150 / 150 dollars
    r"\$\s*([-+]?\d[\d,]*\.?\d*)",
]

# Common entity keywords that help match numbers to variables
_ENTITY_KEYWORDS = {
    "total", "sum", "cost", "price", "amount", "count", "number",
    "average", "mean", "profit", "loss", "revenue", "balance",
    "quantity", "rate", "percent", "percentage", "difference",
    "result", "answer", "value", "score", "distance", "time",
    "weight", "height", "width", "length", "area", "volume",
    "speed", "tax", "discount", "remaining", "left", "earned",
}


def extract_numeric_claims(text: str) -> List[NumericClaim]:
    """
    Extract numeric claims from natural language text.

    Identifies entity-value associations like:
        "The total cost is 150"  → NumericClaim(entity="total_cost", value=150)
        "She earns $5000"        → NumericClaim(entity="earnings", value=5000)
        "150 apples"             → NumericClaim(entity="apples", value=150)
    """
    claims: List[NumericClaim] = []
    seen_values: set = set()

    # Pattern 1: "the X is/equals NUMBER"
    for match in re.finditer(
        r"(?:the\s+)?(\w+(?:\s+\w+)?)\s+(?:is|are|equals?|was|were)\s+([-+]?\d[\d,]*\.?\d*)",
        text,
        re.IGNORECASE,
    ):
        entity = _normalize_entity(match.group(1))
        try:
            value = float(match.group(2).replace(",", ""))
        except ValueError:
            continue

        key = (entity, value)
        if key not in seen_values:
            seen_values.add(key)
            claims.append(NumericClaim(
                entity=entity,
                claimed_value=value,
                source_text=match.group(0),
            ))

    # Pattern 2: "NUMBER entity_word"
    for match in re.finditer(
        r"([-+]?\d[\d,]*\.?\d*)\s+(\w+)",
        text,
        re.IGNORECASE,
    ):
        entity_word = match.group(2).lower()
        if entity_word in _ENTITY_KEYWORDS or entity_word.endswith("s"):
            try:
                value = float(match.group(1).replace(",", ""))
            except ValueError:
                continue

            entity = _normalize_entity(entity_word)
            key = (entity, value)
            if key not in seen_values:
                seen_values.add(key)
                claims.append(NumericClaim(
                    entity=entity,
                    claimed_value=value,
                    source_text=match.group(0),
                ))

    # Pattern 3: "$NUMBER"
    for match in re.finditer(r"\$\s*([-+]?\d[\d,]*\.?\d*)", text):
        try:
            value = float(match.group(1).replace(",", ""))
        except ValueError:
            continue

        key = ("dollar_amount", value)
        if key not in seen_values:
            seen_values.add(key)
            claims.append(NumericClaim(
                entity="dollar_amount",
                claimed_value=value,
                source_text=match.group(0),
            ))

    return claims


def _normalize_entity(text: str) -> str:
    """Normalize entity name to snake_case variable format."""
    text = text.strip().lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_]", "", text)
    return text


# ---------------------------------------------------------------------------
# Numeric Consistency Gate
# ---------------------------------------------------------------------------

class NumericConsistencyGate:
    """
    Process Guard that compares textual claims against computed values.

    Replaces the blanket NLI skip for numeric outputs.
    Uses math.isclose for float comparison with configurable tolerance.
    """

    def __init__(self, rel_tolerance: float = 0.01, abs_tolerance: float = 0.5):
        """
        Args:
            rel_tolerance: Relative tolerance for math.isclose (default 1%)
            abs_tolerance: Absolute tolerance for very small numbers
        """
        self.rel_tolerance = rel_tolerance
        self.abs_tolerance = abs_tolerance
        self._table = get_symbolic_table()

    # ------------------------------------------------------------------
    # Main Check
    # ------------------------------------------------------------------

    def check(
        self,
        query_text: str,
        sandbox_output: Dict[str, Any],
    ) -> NumericConsistencyResult:
        """
        Compare textual claims in query against sandbox output.

        Args:
            query_text: Original query with potential numeric claims
            sandbox_output: Dict of variable → value from sandbox

        Returns:
            NumericConsistencyResult with consistency verdict
        """
        claims = extract_numeric_claims(query_text)
        contradictions: List[NumericContradiction] = []
        checked = 0

        for claim in claims:
            # Try to match claim entity to a sandbox variable
            matched_var, computed_val = self._find_matching_variable(
                claim, sandbox_output
            )

            if matched_var is None:
                continue

            checked += 1

            if not isinstance(computed_val, (int, float)):
                continue

            # Use math.isclose for robust comparison
            is_close = math.isclose(
                claim.claimed_value,
                float(computed_val),
                rel_tol=self.rel_tolerance,
                abs_tol=self.abs_tolerance,
            )

            if not is_close:
                rel_error = (
                    abs(claim.claimed_value - computed_val)
                    / abs(computed_val)
                    if computed_val != 0 else float("inf")
                )

                contradictions.append(NumericContradiction(
                    variable=matched_var,
                    claimed_value=claim.claimed_value,
                    computed_value=float(computed_val),
                    relative_error=rel_error,
                    source_text=claim.source_text,
                    reason=(
                        f"Numeric inconsistency: '{matched_var}' claimed as "
                        f"{claim.claimed_value} in text but computed as "
                        f"{computed_val} by sandbox "
                        f"(relative error: {rel_error:.2%})"
                    ),
                ))

        is_consistent = len(contradictions) == 0

        if not is_consistent:
            for c in contradictions:
                logger.warning(c.reason)

        return NumericConsistencyResult(
            is_consistent=is_consistent,
            claims_found=claims,
            contradictions=contradictions,
            checked_count=checked,
        )

    # ------------------------------------------------------------------
    # Variable Matching
    # ------------------------------------------------------------------

    def _find_matching_variable(
        self,
        claim: NumericClaim,
        sandbox_output: Dict[str, Any],
    ) -> Tuple[Optional[str], Optional[Any]]:
        """
        Match a textual claim entity to a sandbox output variable.

        6-level matching strategy (ordered by specificity):
            1. Exact match in sandbox output
            2. SymbolicTable alias resolution (entity → canonical → sandbox)
            3. Reverse alias scan (sandbox var → check if entity is a known alias)
            4. Pluralization handling (remove/add trailing 's')
            5. Token overlap (e.g., "total_cost" ↔ "cost")
            6. Reverse alias map scan (any registered alias → canonical → sandbox)

        On successful match, auto-registers the entity as an alias
        for the matched variable, building a progressive alias graph.
        """
        entity = claim.entity

        # ----- Level 1: Exact match -----
        if entity in sandbox_output:
            return entity, sandbox_output[entity]

        # ----- Level 2: Alias resolution via SymbolicTable -----
        # The table resolves "total_price" → "cost" if alias was registered
        record = self._table.get(entity)
        if record is not None:
            canonical = record.name
            # Check if the canonical name is in sandbox output
            if canonical in sandbox_output:
                self._table.register_alias(entity, canonical)
                return canonical, sandbox_output[canonical]
            # Otherwise return the table's stored value
            return canonical, record.value

        # ----- Level 3: Reverse alias — check if sandbox vars resolve to entity -----
        for var_name, value in sandbox_output.items():
            var_record = self._table.get(var_name)
            if var_record is not None and var_record.name == entity:
                return var_name, value

        # ----- Level 4: Pluralization variants -----
        singular = entity.rstrip("s") if entity.endswith("s") else entity
        plural = entity + "s" if not entity.endswith("s") else entity

        for variant in (singular, plural):
            if variant in sandbox_output:
                self._table.register_alias(entity, variant)
                return variant, sandbox_output[variant]

            variant_record = self._table.get(variant)
            if variant_record is not None:
                self._table.register_alias(entity, variant_record.name)
                return variant_record.name, variant_record.value

        # ----- Level 5: Token overlap scoring -----
        entity_tokens = set(entity.split("_"))
        best_match: Optional[str] = None
        best_score: float = 0.0

        for var_name, value in sandbox_output.items():
            var_tokens = set(var_name.split("_"))
            if not entity_tokens or not var_tokens:
                continue

            overlap = entity_tokens & var_tokens
            score = len(overlap) / max(len(entity_tokens), len(var_tokens))

            if score > best_score and score >= 0.5:
                best_score = score
                best_match = var_name

        if best_match is not None:
            self._table.register_alias(entity, best_match)
            return best_match, sandbox_output[best_match]

        # ----- Level 6: Substring containment (last resort) -----
        for var_name, value in sandbox_output.items():
            if entity in var_name or var_name in entity:
                self._table.register_alias(entity, var_name)
                return var_name, value

        return None, None


# ---------------------------------------------------------------------------
# Convenience Accessor
# ---------------------------------------------------------------------------


def get_numeric_consistency_gate() -> NumericConsistencyGate:
    """Public accessor for NumericConsistencyGate."""
    return NumericConsistencyGate()
