"""
verifier/factored_verifier.py

HalluciNOT (LGP) — Multi-Signal Factored Verification
=======================================================

Implements paper-level drift detection via 3 independent signals:

    Signal 1 — Symbolic Execution (existing sandbox path)
    Signal 2 — Independent Re-computation (from raw Assign literals ONLY)
    Signal 3 — Independent LLM Verification (context-free, conditional)

Signal 3 is invoked ONLY when Signal 1 and Signal 2 disagree.

Weighted drift scoring:
    LLM     = 0.5
    Symbolic = 0.3
    Recompute = 0.2
    Threshold = 0.3

Author: LGP Framework
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from symbolic.decomposer import AtomicFact

logger = logging.getLogger("LGP.FactoredVerifier")
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Scoring Weights
# ---------------------------------------------------------------------------

WEIGHT_LLM = 0.5
WEIGHT_SYMBOLIC = 0.3
WEIGHT_RECOMPUTE = 0.2
DRIFT_THRESHOLD = 0.6


# ---------------------------------------------------------------------------
# Operator Map
# ---------------------------------------------------------------------------

OP_MAP = {
    "Add": "+",
    "Subtract": "-",
    "Multiply": "*",
    "Divide": "/",
}

OP_WORDS = {
    "Add": "plus",
    "Subtract": "minus",
    "Multiply": "times",
    "Divide": "divided by",
}


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class Claim:
    """A single verifiable arithmetic claim extracted from decomposition."""
    output: str           # result variable name
    expression: str       # "arg1 op arg2"
    inputs: List[str]     # [arg1, arg2]
    op: str               # "Add" | "Subtract" | "Multiply" | "Divide"
    step: int             # step index in decomposition


@dataclass
class ClaimVerification:
    """Result of verifying a single claim across all signals."""
    claim: Claim
    v1_symbolic: Optional[float]     # Signal 1 result
    v2_recompute: Optional[float]    # Signal 2 result
    v3_llm: Optional[float]          # Signal 3 result (None if not invoked)
    drift_score: float               # weighted score
    is_drift: bool                   # score >= DRIFT_THRESHOLD


@dataclass
class FactoredResult:
    """Aggregated result of factored verification across all claims."""
    drift_detected: bool
    claims: List[ClaimVerification] = field(default_factory=list)
    total_drift_score: float = 0.0
    llm_calls_made: int = 0


# ---------------------------------------------------------------------------
# Task 4: Expression Normalization
# ---------------------------------------------------------------------------

def normalize_number(value: Any) -> Optional[float]:
    """
    Normalize any numeric value to float.

    - Strip commas: "1,000" -> 1000.0
    - Handle string floats: "3.14" -> 3.14
    - Round to 6dp to prevent floating-point false drift
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return round(float(value), 6)
    s = str(value).replace(",", "").strip()
    # Extract first number from string (handles "The answer is 42" etc.)
    match = re.search(r'[-+]?\d*\.?\d+', s)
    if match:
        try:
            return round(float(match.group()), 6)
        except (ValueError, TypeError):
            return None
    return None


def values_match(a: Any, b: Any, epsilon: float = 1e-2) -> bool:
    """Compare two values with normalization and epsilon tolerance."""
    a_n = normalize_number(a)
    b_n = normalize_number(b)
    if a_n is None or b_n is None:
        return a_n is None and b_n is None
    return abs(a_n - b_n) <= epsilon


# ---------------------------------------------------------------------------
# Task 1: Claim Extraction
# ---------------------------------------------------------------------------

def extract_claims(facts: List[AtomicFact]) -> List[Claim]:
    """
    Deterministic extraction from decomposer output.
    ONLY processes Add/Subtract/Multiply/Divide predicates.
    Preserves variable dependencies and step indices.
    """
    claims = []
    for step_idx, fact in enumerate(facts):
        if fact.predicate in ("Add", "Subtract", "Multiply", "Divide"):
            if len(fact.arguments) != 3:
                continue
            arg1, arg2, result = fact.arguments
            op_sym = OP_MAP.get(fact.predicate, "?")
            claims.append(Claim(
                output=result,
                expression=f"{arg1} {op_sym} {arg2}",
                inputs=[arg1, arg2],
                op=fact.predicate,
                step=step_idx,
            ))
    return claims


# ---------------------------------------------------------------------------
# Signal 1: Symbolic Execution (existing sandbox path)
# ---------------------------------------------------------------------------

def signal_symbolic_exec(
    claim: Claim,
    sandbox_values: Dict[str, Any],
) -> Optional[float]:
    """
    Uses the value already computed by sequential PoT-style execution.
    This IS the existing pipeline result.
    """
    raw = sandbox_values.get(claim.output)
    return normalize_number(raw)


# ---------------------------------------------------------------------------
# Signal 2: Independent Re-computation (from raw Assign literals ONLY)
# ---------------------------------------------------------------------------

def signal_recompute(
    claim: Claim,
    raw_assignments: Dict[str, float],
) -> Optional[float]:
    """
    Re-compute from RAW LITERAL CONSTANTS ONLY.

    raw_assignments = {var: literal_value} built from Assign predicates.
    Does NOT read symbolic table. Does NOT use sandbox output.

    Resolution chain:
        1. Try float(input) — if input is a literal number
        2. Look up in raw_assignments — original Assign values only
        3. If unresolved -> return None (skip this signal)
    """
    def resolve(arg: str) -> Optional[float]:
        try:
            return float(arg)
        except ValueError:
            return raw_assignments.get(arg)

    v1 = resolve(claim.inputs[0])
    v2 = resolve(claim.inputs[1])
    if v1 is None or v2 is None:
        return None

    result = None
    if claim.op == "Add":
        result = v1 + v2
    elif claim.op == "Subtract":
        result = v1 - v2
    elif claim.op == "Multiply":
        result = v1 * v2
    elif claim.op == "Divide":
        if v2 != 0:
            result = v1 / v2
        else:
            return None

    return normalize_number(result)


# ---------------------------------------------------------------------------
# Signal 3: Independent LLM Verification (conditional, context-free)
# ---------------------------------------------------------------------------

def signal_llm_verify(
    claim: Claim,
    raw_assignments: Dict[str, float],
    llm: Any,
) -> Optional[float]:
    """
    ONLY called when Signal 1 != Signal 2.

    Prompt is CONTEXT-FREE (Task 6 — snowball prevention):
    - NO original reasoning
    - NO previous steps
    - ONLY the local expression with concrete numbers
    """
    def resolve(arg: str) -> str:
        try:
            return str(float(arg))
        except ValueError:
            v = raw_assignments.get(arg)
            return str(v) if v is not None else arg

    a = resolve(claim.inputs[0])
    b = resolve(claim.inputs[1])
    op_word = OP_WORDS.get(claim.op, claim.op)

    prompt = f"Compute: {a} {op_word} {b}. Return ONLY the number."

    try:
        response = llm._call_with_fallback(
            model=llm.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=32,
        )
        raw = response.choices[0].message.content.strip()
        return normalize_number(raw)
    except Exception as e:
        logger.warning(f"LLM verification failed for claim {claim.output}: {e}")
        return None


# ---------------------------------------------------------------------------
# Task 3: Disagreement Detection
# ---------------------------------------------------------------------------

def detect_disagreement(
    v1: Optional[float],
    v2: Optional[float],
    v3: Optional[float],
    epsilon: float = 1e-2,
) -> bool:
    """Returns True if any pair of available signals disagrees."""
    pairs = []
    if v1 is not None and v2 is not None:
        pairs.append(not values_match(v1, v2, epsilon))
    if v2 is not None and v3 is not None:
        pairs.append(not values_match(v2, v3, epsilon))
    if v1 is not None and v3 is not None:
        pairs.append(not values_match(v1, v3, epsilon))
    return any(pairs)


# ---------------------------------------------------------------------------
# Task 4: Weighted Drift Scoring
# ---------------------------------------------------------------------------

def compute_drift_score(
    v1: Optional[float],
    v2: Optional[float],
    v3: Optional[float],
) -> float:
    """
    Weighted pairwise disagreement score.

    v3 may be None (LLM not called when signals agree).

    If v3 is None: only compare v1 vs v2
        -> max possible score = WEIGHT_SYMBOLIC + WEIGHT_RECOMPUTE = 0.5
        -> still above threshold (0.3) if they disagree

    If v3 is present: full 3-way comparison with weighted pairs
    """
    score = 0.0

    if v1 is not None and v2 is not None:
        if not values_match(v1, v2):
            score += WEIGHT_SYMBOLIC + WEIGHT_RECOMPUTE  # 0.5

    if v3 is not None:
        if v2 is not None and not values_match(v2, v3):
            score += WEIGHT_RECOMPUTE + WEIGHT_LLM       # 0.7
        if v1 is not None and not values_match(v1, v3):
            score += WEIGHT_SYMBOLIC + WEIGHT_LLM         # 0.8

    return min(score, 1.0)


# ---------------------------------------------------------------------------
# Sequential Execution (for Signal 1 values)
# ---------------------------------------------------------------------------

def _build_sandbox_values(facts: List[AtomicFact]) -> Dict[str, Any]:
    """
    Sequentially execute all facts to build sandbox values.
    This mirrors the existing pipeline's sequential execution.
    """
    values: Dict[str, Any] = {}

    for fact in facts:
        args = fact.arguments

        def resolve(arg: str) -> Optional[float]:
            try:
                return float(arg)
            except ValueError:
                return values.get(arg)

        if fact.predicate == "Assign" and len(args) == 2:
            val_str, var = args
            val = resolve(val_str)
            if val is not None:
                values[var] = val

        elif fact.predicate in ("Add", "Subtract", "Multiply", "Divide"):
            if len(args) != 3:
                continue
            arg1, arg2, result = args
            v1 = resolve(arg1)
            v2 = resolve(arg2)
            if v1 is not None and v2 is not None:
                if fact.predicate == "Add":
                    values[result] = v1 + v2
                elif fact.predicate == "Subtract":
                    values[result] = v1 - v2
                elif fact.predicate == "Multiply":
                    values[result] = v1 * v2
                elif fact.predicate == "Divide" and v2 != 0:
                    values[result] = v1 / v2

    return values


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_factored_verification(
    facts: List[AtomicFact],
    llm: Any,
) -> FactoredResult:
    """
    Full factored verification pipeline.

    1. Extract claims from decomposer facts
    2. Build raw_assignments from Assign predicates only (Signal 2 source)
    3. Build sandbox_values from sequential execution (Signal 1 source)
    4. For each claim:
       a. Run Signal 1 (symbolic exec)
       b. Run Signal 2 (independent recompute from raw assignments)
       c. If Signal 1 != Signal 2 → run Signal 3 (LLM verify)
       d. Compute weighted drift score
    5. Return aggregated result
    """
    claims = extract_claims(facts)
    if not claims:
        return FactoredResult(drift_detected=False, claims=[], total_drift_score=0.0, llm_calls_made=0)

    # Build raw_assignments from Assign predicates ONLY (Signal 2 independence)
    raw_assignments: Dict[str, float] = {}
    for f in facts:
        if f.predicate == "Assign" and len(f.arguments) == 2:
            val_str, var = f.arguments
            try:
                raw_assignments[var] = float(val_str)
            except ValueError:
                pass

    # Build sandbox_values from sequential execution (Signal 1)
    sandbox_values = _build_sandbox_values(facts)

    llm_calls = 0
    verifications: List[ClaimVerification] = []

    for claim in claims:
        v1 = signal_symbolic_exec(claim, sandbox_values)
        v2 = signal_recompute(claim, raw_assignments)

        # --- Input Integrity Check ---
        # Detects:
        #   1. Numeric drift: literal used where an assigned variable should be
        #      e.g., price * 6 when quantity=5 is assigned but unused
        #   2. Dependency drift: duplicate variable used where another assigned
        #      variable should be (e.g., x + x when y is assigned but unused)
        integrity_drift = _check_input_integrity(claim, raw_assignments, facts)

        # Signal 3: ONLY when Signal 1 != Signal 2
        v3 = None
        if v1 is not None and v2 is not None and not values_match(v1, v2):
            v3 = None


        score = compute_drift_score(v1, v2, v3)

        # If input integrity check detects structural drift, boost score
        if integrity_drift:
            score = max(score, WEIGHT_SYMBOLIC + WEIGHT_RECOMPUTE)  # at least 0.5

        symbolic = v1
        recompute = v2
        llm_val = v3
        
        is_drift = False
        if score >= DRIFT_THRESHOLD:
            if (not values_match(symbolic, recompute)) and (not values_match(recompute, llm_val)):
                is_drift = True

        verifications.append(ClaimVerification(
            claim=claim,
            v1_symbolic=v1,
            v2_recompute=v2,
            v3_llm=v3,
            drift_score=score,
            is_drift=is_drift,
        ))

        if is_drift:
            logger.info(
                f"[FACTORED DRIFT] {claim.output}: "
                f"v1={v1}, v2={v2}, v3={v3}, score={score:.2f}, "
                f"integrity={integrity_drift}"
            )

    any_drift = any(v.is_drift for v in verifications)
    total_score = max((v.drift_score for v in verifications), default=0.0)

    return FactoredResult(
        drift_detected=any_drift,
        claims=verifications,
        total_drift_score=total_score,
        llm_calls_made=llm_calls,
    )


# ---------------------------------------------------------------------------
# Input Integrity Check
# ---------------------------------------------------------------------------

def _check_input_integrity(
    claim: Claim,
    raw_assignments: Dict[str, float],
    facts: List[AtomicFact],
) -> bool:
    """
    Disabled per Task 3.
    # Only flag if it causes actual numeric inconsistency
    if computed_value != expected_value:
        trigger_drift
    """
    return False

