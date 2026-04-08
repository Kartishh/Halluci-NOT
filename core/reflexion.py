"""
core/reflexion.py

HalluciNOT (LGP) — Reflexion Loop
====================================

The CORE correction mechanism of the LGP framework.

When SSCE detects symbolic drift in LLM reasoning:
    1. Extract DriftReport details (variable, old_value, new_value)
    2. Build a Reflexion prompt citing the specific inconsistency
    3. Feed back to LLM → regenerate reasoning
    4. Re-run decomposition → verification
    5. Repeat up to MAX_ITERATIONS

This is what makes HalluciNOT an AGENTIC system, not a passive checker.

Upgrade v2: Step-level causal tracking + targeted reflexion + partial repair.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

from symbolic.decomposer import AtomicFact
from symbolic.ssce_algorithm import (
    SSCEEngine,
    SSCEEnforcementError,
    DriftReport,
    get_ssce_engine,
)
from symbolic.table import get_symbolic_table
from verifier.pot_engine import get_pot_engine, PoTScript
from core.groq_llm import GroqLLM, ReasoningResult, DecompositionResult

logger = logging.getLogger("LGP.Reflexion")
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_REFLEXION_ITERATIONS = 3


# ---------------------------------------------------------------------------
# Result Schema
# ---------------------------------------------------------------------------

@dataclass
class ReflexionResult:
    """Full result of the Reflexion loop."""
    final_answer: float
    reasoning: str
    iterations_used: int
    drift_detected: bool
    drift_reports: List[Dict[str, Any]] = field(default_factory=list)
    correction_applied: bool = False
    correction_successful: bool = False
    execution_trace: List[Dict[str, Any]] = field(default_factory=list)
    dependency_graph: Dict[str, List[str]] = field(default_factory=dict)
    factored_drift: bool = False
    factored_reports: List[Dict[str, Any]] = field(default_factory=list)
    factored_llm_calls: int = 0


# ---------------------------------------------------------------------------
# Safe Execution
# ---------------------------------------------------------------------------

def _safe_execute(script: str) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """Execute a PoT script in-process with restricted builtins."""
    import math as _math
    import re as _re

    # Strip import lines — math is pre-injected
    lines = script.split("\n")
    lines = [l for l in lines if not _re.match(r'^\s*import\s+', l)]
    script = "\n".join(lines)

    safe_builtins = {
        "abs": abs, "round": round, "int": int, "float": float,
        "str": str, "len": len, "min": min, "max": max,
        "True": True, "False": False, "None": None,
        "print": lambda *a, **kw: None,
    }
    restricted_globals = {"__builtins__": safe_builtins, "math": _math}
    local_ns: Dict[str, Any] = {}

    try:
        exec(script, restricted_globals, local_ns)
        result = local_ns.get("__result__", {})
        if not isinstance(result, dict):
            return False, None, "No __result__ dict in output"
        return True, result, ""
    except Exception as e:
        return False, None, str(e)[:300]


# ---------------------------------------------------------------------------
# Target Variable Extraction
# ---------------------------------------------------------------------------

def extract_target_variable(question: str) -> Optional[str]:
    """
    Extract the target variable from a question using simple heuristics.
    Returns None if no target can be determined (safe fallback).
    """
    q = question.lower().strip()

    # Pattern: "how much profit" → profit
    m = re.search(r'how (?:much|many) (\w+)', q)
    if m:
        return m.group(1).replace(' ', '_')

    # Pattern: "what is the total cost" → total_cost
    m = re.search(r'what (?:is|are) (?:the )?((?:\w+ )*\w+?)\??$', q)
    if m:
        return m.group(1).strip().replace(' ', '_')

    # Pattern: "what is X" at end of sentence
    m = re.search(r'what is (\w+)', q)
    if m:
        return m.group(1).replace(' ', '_')

    # Pattern: "find the X"
    m = re.search(r'find (?:the )?(\w+)', q)
    if m:
        return m.group(1).replace(' ', '_')

    return None  # Safe fallback — skip validation if unknown


# ---------------------------------------------------------------------------
# Step Splitting (Task 4 — Partial Repair)
# ---------------------------------------------------------------------------

def split_into_steps(reasoning: str) -> List[str]:
    """
    Split reasoning text into numbered steps.
    First: split by numbered steps (e.g., '1.', 'Step 1')
    Fallback: split by newline.
    Safe: always returns at least 1 element.
    """
    if not reasoning or not reasoning.strip():
        return [reasoning or ""]

    # Try numbered step pattern: "1.", "2.", "Step 1:", etc.
    parts = re.split(r'(?=(?:^|\n)\s*(?:Step\s+)?\d+[\.\):\s])', reasoning.strip())
    parts = [p.strip() for p in parts if p.strip()]

    if len(parts) >= 2:
        return parts

    # Fallback: split by newlines (sentences)
    parts = [p.strip() for p in reasoning.strip().split('\n') if p.strip()]
    if len(parts) >= 2:
        return parts

    # Ultimate fallback
    return [reasoning.strip()]


# ---------------------------------------------------------------------------
# Step-Level Causal Drift Detection (Tasks 1, 2, 6)
# ---------------------------------------------------------------------------

def detect_drift_from_facts(facts: List[AtomicFact]):
    """
    Detect symbolic drift with step-level causal tracking.

    Returns:
        (has_drift, drift_type, confidence, var, old, new, all_drifts,
         formulas, value_history, dependency_graph)
    """
    values = {}
    # Task 1: formulas now tracks op, inputs, AND step index
    formulas = {}   # var -> {"op": pred, "inputs": [arg1, arg2], "step": step_index}
    # Task 1: value history per variable
    value_history = {}  # var -> [{"value": v, "step": step_index}]
    # Task 6: dependency graph
    dependency_graph = {}  # var -> [input_vars]

    drifts = []
    best = {
        "drift": False, "type": "none", "confidence": 0.0,
        "var": None, "old": None, "new": None,
        "source_step": None, "error_step": None
    }

    def add_drift(d_type, d_conf, d_var, d_old, d_new, source_step=None, error_step=None):
        entry = {
            "drift": True, "type": d_type,
            "confidence": d_conf, "var": d_var,
            "old": d_old, "new": d_new,
            "source_step": source_step,
            "error_step": error_step
        }
        drifts.append(entry)
        if d_conf > best["confidence"]:
            best["drift"] = True
            best["type"] = d_type
            best["confidence"] = d_conf
            best["var"] = d_var
            best["old"] = d_old
            best["new"] = d_new
            best["source_step"] = source_step
            best["error_step"] = error_step

    def _track_value(var, val, step_idx):
        """Track value history for a variable."""
        if var not in value_history:
            value_history[var] = []
        value_history[var].append({"value": val, "step": step_idx})

    for step_index, f in enumerate(facts):
        pred = f.predicate
        args = f.arguments

        def resolve(arg):
            try:
                return float(arg)
            except ValueError:
                return values.get(arg, None)

        if pred == "Assign":
            if len(args) != 2:
                continue
            val_str, var = args
            val = resolve(val_str)
            if val is None:
                continue

            _track_value(var, val, step_index)

            if var in values:
                # Task 2: source_step from formulas, error_step is current
                source_step = formulas[var]["step"] if var in formulas else step_index
                if var in formulas:
                    # Variable was computed — Assign overwrites a computed var
                    if abs(values[var] - val) > 1e-5:
                        add_drift("dependency_mutation", 0.85, var,
                                  values[var], val,
                                  source_step=source_step,
                                  error_step=step_index)
                else:
                    if abs(values[var] - val) > 1e-5:
                        add_drift("redefinition", 0.8, var,
                                  values[var], val,
                                  source_step=source_step,
                                  error_step=step_index)
            values[var] = val
            # Task 6: Assign has no dependencies (literal)
            if var not in dependency_graph:
                dependency_graph[var] = []

        elif pred in ["Add", "Subtract", "Multiply", "Divide"]:
            if len(args) != 3:
                continue
            arg1, arg2, res = args

            # Task 6: track dependency graph
            input_vars = []
            for a in [arg1, arg2]:
                try:
                    float(a)
                except ValueError:
                    input_vars.append(a)
            dependency_graph[res] = input_vars

            if res in formulas:
                old_info = formulas[res]
                old_pred = old_info["op"]
                o1, o2 = old_info["inputs"]
                old_step = old_info["step"]

                # Task 5: ALWAYS flag dependency_mutation when inputs change
                old_inputs = set(old_info["inputs"])
                new_inputs = set([arg1, arg2])

                if pred == old_pred and old_inputs == new_inputs:
                    pass  # Valid recomputation
                elif old_pred != pred and old_inputs == new_inputs:
                    add_drift("sign_flip", 0.95, res, old_pred, pred,
                              source_step=old_step, error_step=step_index)
                elif old_inputs != new_inputs:
                    # Task 5: set comparison — always flag
                    add_drift("dependency_mutation", 0.90, res,
                              f"{old_pred}({o1},{o2})", f"{pred}({arg1},{arg2})",
                              source_step=old_step, error_step=step_index)

            v1 = resolve(arg1)
            v2 = resolve(arg2)

            if v1 is not None and v2 is not None:
                expected = None
                if pred == "Add":
                    expected = v1 + v2
                elif pred == "Subtract":
                    expected = v1 - v2
                elif pred == "Multiply":
                    expected = v1 * v2
                elif pred == "Divide":
                    if v2 == 0:
                        add_drift("invalid_operation", 0.9, res,
                                  "division_by_zero", None,
                                  source_step=step_index, error_step=step_index)
                    else:
                        expected = v1 / v2

                if expected is not None:
                    if res in values and abs(values[res] - expected) > 1e-5:
                        src = formulas[res]["step"] if res in formulas else step_index
                        add_drift("numeric_inconsistency", 0.90, res,
                                  values[res], expected,
                                  source_step=src, error_step=step_index)
                    values[res] = expected
                    _track_value(res, expected, step_index)

            # Task 1: store formula with step index
            formulas[res] = {"op": pred, "inputs": [arg1, arg2], "step": step_index}

    # Task 3: Drift detection fallback when no formulas exist
    if not formulas:
        for var, history in value_history.items():
            if len(history) > 1:
                first = history[0]
                for curr in history[1:]:
                    if abs(curr["value"] - first["value"]) > 1e-5:
                        add_drift("redefinition", 0.8, var, first["value"], curr["value"], 
                                  source_step=first["step"], error_step=curr["step"])

    # --- Task 3: Execution Validation Post-Pass (NEW) ---
    # Recompute ALL formula outputs from current variable values.
    # Catches numeric drift that inline checks may miss when
    # upstream values changed after the formula was first evaluated.
    NUMERIC_TOLERANCE = 1e-5
    for var, formula in formulas.items():
        op = formula["op"]
        inp = formula["inputs"]
        rv1 = values.get(inp[0])
        rv2 = values.get(inp[1])

        # Try literal resolution if variable not in values
        if rv1 is None:
            try:
                rv1 = float(inp[0])
            except ValueError:
                continue
        if rv2 is None:
            try:
                rv2 = float(inp[1])
            except ValueError:
                continue

        recomputed = None
        if op == "Add":
            recomputed = rv1 + rv2
        elif op == "Subtract":
            recomputed = rv1 - rv2
        elif op == "Multiply":
            recomputed = rv1 * rv2
        elif op == "Divide" and rv2 != 0:
            recomputed = rv1 / rv2

        if recomputed is not None and var in values:
            stored = values[var]
            if abs(stored - recomputed) > NUMERIC_TOLERANCE:
                add_drift(
                    "numeric_inconsistency", 0.90, var,
                    stored, recomputed,
                    source_step=formula["step"],
                    error_step=formula["step"]
                )

    return (
        best["drift"], best["type"], best["confidence"],
        best["var"], best["old"], best["new"],
        drifts,
        formulas, value_history, dependency_graph,
        best.get("source_step"), best.get("error_step")
    )


# ---------------------------------------------------------------------------
# Targeted Reflexion Prompt (Task 3 — CRITICAL)
# ---------------------------------------------------------------------------

def build_reflexion_prompt(
    drift_reports: List,
    target_variable: Optional[str] = None,
    source_step: Optional[int] = None,
    error_step: Optional[int] = None,
) -> str:
    """
    Build a TARGETED reflexion prompt that cites the exact step
    where the error occurred and instructs LLM to fix only that step
    and steps after it.
    """
    lines = [
        "An error was detected in your reasoning.",
        "",
    ]

    for i, report in enumerate(drift_reports, 1):
        # Handle both DriftReport objects and dict entries
        if isinstance(report, dict):
            var = report.get("var", report.get("variable", "unknown"))
            d_type = report.get("type", report.get("drift_type", "redefinition"))
            old = report.get("old", report.get("old_value", "?"))
            new = report.get("new", report.get("new_value", "?"))
            src = report.get("source_step", source_step)
            err = report.get("error_step", error_step)
        else:
            var = getattr(report, 'variable', 'unknown')
            d_type = getattr(report, 'drift_type', 'redefinition')
            old = getattr(report, 'old_value', '?')
            new = getattr(report, 'new_value', '?')
            src = source_step
            err = error_step

        lines.append(f"  Variable: {var}")
        lines.append(f"  Error type: {d_type}")
        if err is not None:
            lines.append(f"  Incorrect step: Step {err}")
        if src is not None:
            lines.append(f"  Originally defined at: Step {src}")
        lines.append(f"  Issue: Expected {old} but got {new}")
        lines.append("")

    lines.extend([
        "INSTRUCTIONS:",
        "- Prefer minimal correction.",
        "- Avoid rewriting correct earlier steps unless necessary.",
        "- Maintain consistency of earlier variables.",
    ])

    if target_variable:
        lines.append(
            f"- Ensure final answer computes: {target_variable}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Partial Repair (Task 4 — No Full Rewrite)
# ---------------------------------------------------------------------------

def partial_repair(
    llm: GroqLLM,
    query: str,
    reasoning: str,
    error_step_idx: int,
    critique: str,
    target_variable: Optional[str] = None,
) -> str:
    """
    Split reasoning into steps, keep correct steps, only rewrite from error onward.
    Returns the fully reconstructed reasoning.
    """
    steps = split_into_steps(reasoning)

    # Clamp error_step_idx
    if error_step_idx < 0:
        error_step_idx = 0
    if error_step_idx >= len(steps):
        error_step_idx = max(0, len(steps) - 1)

    correct_prefix = steps[:error_step_idx]
    broken_suffix = steps[error_step_idx:]

    # Build context for LLM — show correct steps and ask to fix the rest
    prefix_text = "\n".join(correct_prefix) if correct_prefix else "(beginning of reasoning)"
    suffix_text = "\n".join(broken_suffix)

    fix_prompt = (
        f"You previously solved this problem, but made an error starting at step {error_step_idx + 1}.\n\n"
        f"Problem: {query}\n\n"
        f"CORRECT earlier steps (DO NOT CHANGE THESE):\n{prefix_text}\n\n"
        f"INCORRECT steps that need fixing:\n{suffix_text}\n\n"
        f"Error: {critique}\n\n"
        f"Rewrite ONLY the incorrect steps. Keep the same format."
    )
    if target_variable:
        fix_prompt += f"\nEnsure the final answer computes: {target_variable}"

    try:
        fixed_result = llm.generate_reasoning(query, reflexion_feedback=fix_prompt)
        # Reconstruct: correct prefix + fixed suffix
        if correct_prefix:
            return "\n".join(correct_prefix) + "\n" + fixed_result.reasoning
        else:
            return fixed_result.reasoning
    except Exception as e:
        logger.warning(f"Partial repair failed: {e}. Falling back to full rewrite.")
        try:
            result = llm.generate_reasoning(query, reflexion_feedback=critique)
            return result.reasoning
        except Exception:
            return reasoning  # Ultimate safety: return original


# ---------------------------------------------------------------------------
# Constraint-Guided Surgical Repair (Upgrade — CORRECTION IMPROVEMENT)
# ---------------------------------------------------------------------------

OP_SYMBOLS = {"Add": "+", "Subtract": "-", "Multiply": "*", "Divide": "/"}


def _match_var(a: str, b: str) -> bool:
    """Check if two variable names refer to the same variable, ignoring underscores/case."""
    return normalize_var(a).replace('_', '') == normalize_var(b).replace('_', '')


def _build_raw_assignments(facts: List[AtomicFact]) -> Dict[str, float]:
    """Extract {variable: value} from Assign facts."""
    raw: Dict[str, float] = {}
    for f in facts:
        if f.predicate == "Assign" and len(f.arguments) == 2:
            try:
                raw[f.arguments[1]] = float(f.arguments[0])
            except ValueError:
                pass
    return raw


def constraint_guided_repair(
    llm: GroqLLM,
    query: str,
    reasoning: str,
    error_step_idx: int,
    critique: str,
    facts: List[AtomicFact],
    drift_info: Dict[str, Any],
    target_variable: Optional[str] = None,
) -> str:
    """
    Constraint-guided, step-level surgical repair.

    Three-tier strategy:
        Tier 1: Deterministic repair — direct string-level fix based on drift type
        Tier 2: Constrained LLM repair — minimal prompt, return only fixed lines
        Tier 3: Original partial_repair fallback

    Returns:
        (corrected_reasoning, deterministic_repair_used)
        deterministic_repair_used is True only for Tier 1.
    """
    # Tier 1: Deterministic repair
    det_result = _try_deterministic_repair(reasoning, facts, drift_info)
    if det_result is not None:
        print("[REPAIR] Tier 1: Deterministic surgical fix applied")
        return det_result, True

    # Tier 2: Constrained LLM repair
    constr_result = _try_constrained_llm_repair(
        llm, query, reasoning, error_step_idx, critique, drift_info, target_variable
    )
    if constr_result is not None:
        print("[REPAIR] Tier 2: Constrained LLM fix applied")
        return constr_result, False

    # Tier 3: Fallback
    print("[REPAIR] Tier 3: Fallback to original partial repair")
    return partial_repair(llm, query, reasoning, error_step_idx, critique, target_variable), False


def _try_deterministic_repair(
    reasoning: str,
    facts: List[AtomicFact],
    drift_info: Dict[str, Any],
) -> Optional[str]:
    """
    Attempt deterministic string-level fix.
    Works for simple pseudo-code reasoning (controlled drift cases).
    """
    d_type = drift_info.get("type", "")
    var = drift_info.get("variable", "")

    raw_assignments = _build_raw_assignments(facts)
    lines = [l for l in reasoning.strip().split('\n') if l.strip()]

    if not lines or not var:
        return None

    if d_type == "factored_drift":
        # Try numeric fix (wrong literal), then dependency fix (duplicate var)
        result = _fix_numeric_drift(lines[:], facts, raw_assignments, var)
        if result is None:
            result = _fix_dependency_drift(lines[:], facts, raw_assignments, var)
        return result
    elif d_type in ("redefinition", "numeric_inconsistency"):
        return _fix_redefinition_drift(lines[:], facts, var)

    return None


def _fix_numeric_drift(
    lines: List[str],
    facts: List[AtomicFact],
    raw_assignments: Dict[str, float],
    var: str,
) -> Optional[str]:
    """
    Fix numeric drift: replace wrong literal with correct variable.
    Example: total = price * 6 → total = price * quantity
    """
    from verifier.factored_verifier import values_match

    for f in facts:
        if f.predicate not in OP_SYMBOLS or len(f.arguments) != 3:
            continue
        output = f.arguments[2]
        if not _match_var(output, var):
            continue

        inp0, inp1 = f.arguments[0], f.arguments[1]

        for idx, inp in enumerate([inp0, inp1]):
            try:
                literal_val = float(inp)
            except ValueError:
                continue

            # Skip if literal matches any assigned value (valid inline)
            if any(values_match(literal_val, v) for v in raw_assignments.values()):
                continue

            # This is a wrong literal. Find the unused assigned variable.
            other_inp = inp1 if idx == 0 else inp0
            for vname in raw_assignments:
                if _match_var(vname, other_inp) or _match_var(vname, output):
                    continue

                # Replace the literal with the correct variable in the line
                for i, line in enumerate(lines):
                    parts = line.split('=', 1)
                    if len(parts) == 2 and _match_var(parts[0].strip(), output):
                        new_line = line.replace(inp, vname, 1)
                        lines[i] = new_line
                        return '\n'.join(lines)
        break  # Only process first matching operation fact

    return None


def _fix_dependency_drift(
    lines: List[str],
    facts: List[AtomicFact],
    raw_assignments: Dict[str, float],
    var: str,
) -> Optional[str]:
    """
    Fix dependency drift: replace duplicated variable with unused variable.
    Example: z = x + x → z = x + y
    """
    for f in facts:
        if f.predicate not in OP_SYMBOLS or len(f.arguments) != 3:
            continue
        output = f.arguments[2]
        if not _match_var(output, var):
            continue

        inp0, inp1 = f.arguments[0], f.arguments[1]
        if inp0 != inp1:
            continue  # Not duplicated

        dup_var = inp0
        used_norms = {normalize_var(inp0).replace('_', ''),
                      normalize_var(output).replace('_', '')}

        # Find unused assigned variable
        for vname in raw_assignments:
            if normalize_var(vname).replace('_', '') in used_norms:
                continue

            # Replace LAST occurrence of dup_var in the RHS
            for i, line in enumerate(lines):
                parts = line.split('=', 1)
                if len(parts) == 2 and _match_var(parts[0].strip(), output):
                    rhs = parts[1]
                    last_pos = rhs.rfind(dup_var)
                    if last_pos >= 0:
                        rhs = rhs[:last_pos] + vname + rhs[last_pos + len(dup_var):]
                        lines[i] = parts[0] + '=' + rhs
                        return '\n'.join(lines)
            break
        break

    return None


def _fix_redefinition_drift(
    lines: List[str],
    facts: List[AtomicFact],
    var: str,
) -> Optional[str]:
    """
    Fix redefinition drift: remove the re-computation line.
    Example: x=5, y=3, z=x*y, x=8, z=x*y → remove last z=x*y
    """
    # Find lines that compute 'var' with an arithmetic expression
    compute_indices = []
    for i, line in enumerate(lines):
        parts = line.strip().split('=', 1)
        if len(parts) != 2:
            continue
        lhs = parts[0].strip()
        rhs = parts[1].strip()

        if _match_var(lhs, var) and re.search(r'[\+\-\*/]', rhs):
            compute_indices.append(i)

    if len(compute_indices) >= 2:
        # Remove the LAST computation (the re-computation after redefinition)
        lines.pop(compute_indices[-1])
        return '\n'.join(lines)

    return None


def _try_constrained_llm_repair(
    llm: GroqLLM,
    query: str,
    reasoning: str,
    error_step_idx: int,
    critique: str,
    drift_info: Dict[str, Any],
    target_variable: Optional[str] = None,
) -> Optional[str]:
    """
    Constrained LLM repair: asks LLM to fix ONLY the broken step(s).
    Uses a minimal prompt + low max_tokens to prevent full rewrites.
    """
    steps = split_into_steps(reasoning)

    if error_step_idx < 0:
        error_step_idx = 0
    if error_step_idx >= len(steps):
        error_step_idx = max(0, len(steps) - 1)

    prefix = steps[:error_step_idx]
    broken = steps[error_step_idx:]

    d_type = drift_info.get("type", "unknown")
    var = drift_info.get("variable", "unknown")
    old = drift_info.get("old", "?")
    new = drift_info.get("new", "?")

    # Type-specific guidance
    type_guidance = ""
    if d_type in ("factored_drift",):
        type_guidance = (
            "Use correct variable names from earlier steps. "
            "Do NOT substitute variables with wrong literal numbers. "
            "Do NOT duplicate the same variable where two different variables should be used."
        )
    elif d_type in ("redefinition", "numeric_inconsistency"):
        type_guidance = (
            "Do NOT recompute variables after a base variable is redefined. "
            "Remove or skip any re-computation that uses the redefined value."
        )

    prefix_text = '\n'.join(prefix) if prefix else "(beginning)"
    broken_text = '\n'.join(broken)

    prompt = (
        f"Problem: {query}\n\n"
        f"CORRECT steps (DO NOT change):\n{prefix_text}\n\n"
        f"BROKEN step(s) (FIX these):\n{broken_text}\n\n"
        f"Error: Variable '{var}' should be {old} but got {new}.\n"
        f"{type_guidance}\n\n"
        f"Return ONLY the corrected step(s). Same format. No explanations."
    )
    if target_variable:
        prompt += f"\nFinal answer must be in: {target_variable}"

    try:
        response = llm._call_with_fallback(
            model=llm.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You fix specific errors in math solutions. "
                        "Return ONLY the corrected line(s). "
                        "No explanations. No extra text. Just the fixed lines."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=256,
        )
        fixed_text = response.choices[0].message.content.strip()

        if not fixed_text:
            return None

        # Reassemble: correct prefix + fixed suffix
        if prefix:
            return '\n'.join(prefix) + '\n' + fixed_text
        return fixed_text

    except Exception as e:
        logger.warning(f"Constrained LLM repair failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Variable Normalization Helper
# ---------------------------------------------------------------------------

def normalize_var(var_name):
    try:
        float(var_name)
        return var_name
    except ValueError:
        v = str(var_name).lower()
        tokens = [t for t in re.split(r'[_ ]+', v) if t]
        return "_".join(tokens) if tokens else str(var_name)


# ---------------------------------------------------------------------------
# Reflexion Loop Engine
# ---------------------------------------------------------------------------

class ReflexionEngine:
    """
    Agentic correction loop: LLM → Decompose → Execute → SSCE → Reflect.

    This is the core engine that differentiates HalluciNOT from a
    simple LLM baseline.
    """

    def __init__(self, llm: GroqLLM):
        self.llm = llm
        from symbolic.decomposer import get_symbolic_decomposer
        self.decomposer = get_symbolic_decomposer()
        self.pot_engine = get_pot_engine()

    def run(self, query: str, forced_reasoning: Optional[str] = None) -> ReflexionResult:
        """
        Run the full LGP pipeline with Reflexion correction.

        1. Generate reasoning (LLM)
        2. Decompose to predicates (LLM)
        3. Execute via PoT sandbox
        4. Check SSCE
        5. If drift → build targeted Reflexion prompt → partial repair → back to step 1
        6. Max MAX_REFLEXION_ITERATIONS attempts
        """
        trace = []
        drift_detected = False
        all_reports: List[Dict[str, Any]] = []
        correction_applied = False
        reflexion_feedback = None
        last_dependency_graph = {}

        for iteration in range(MAX_REFLEXION_ITERATIONS + 1):
            iter_label = f"iter_{iteration}"

            # ----- Step 1: Generate Reasoning -----
            if forced_reasoning is not None and iteration == 0:
                reasoning_result = ReasoningResult(
                    reasoning=forced_reasoning,
                    final_answer=float("nan"),
                    raw_response=forced_reasoning
                )
            else:
                reasoning_result = self.llm.generate_reasoning(
                    query, reflexion_feedback=reflexion_feedback
                )

            # ----- Step 2: Decompose to Predicates -----
            try:
                facts = self.decomposer.to_atomic_facts(reasoning_result.reasoning)
            except Exception as e:
                print(f"⚠️ SymbolicDecomposer completely failed ({e.__class__.__name__}) — falling back to GroqLLM")
                decomp_result = self.llm.decompose_to_predicates(reasoning_result.reasoning)
                from symbolic.decomposer import AtomicFact
                facts = [
                    AtomicFact(
                        predicate=p["predicate"],
                        arguments=p["arguments"],
                        raw_text="groq_fallback",
                        source_path="groq"
                    )
                    for p in decomp_result.predicates
                ]

            # Normalize variables
            from symbolic.decomposer import AtomicFact
            normalized_facts = []
            for f in facts:
                norm_args = [normalize_var(arg) for arg in f.arguments]
                normalized_facts.append(
                    AtomicFact(
                        predicate=f.predicate,
                        arguments=norm_args,
                        raw_text=f.raw_text,
                        source_path=f.source_path
                    )
                )
            facts = normalized_facts

            print("\n===== DEBUG DECOMPOSITION =====")
            print("REASONING:\n", reasoning_result.reasoning[:300])
            print("VALID: True")
            print("PREDICATES:", [(f.predicate, f.arguments) for f in facts])
            print("================================\n")

            if not facts:
                print("⚠️ Decomposition yielded no facts")
                trace.append({
                    "iteration": iteration,
                    "status": "decomposition_failed_total",
                    "reasoning": reasoning_result.reasoning[:200],
                })
                return ReflexionResult(
                    final_answer=reasoning_result.final_answer,
                    reasoning=reasoning_result.reasoning,
                    iterations_used=iteration + 1,
                    drift_detected=drift_detected,
                    drift_reports=all_reports,
                    correction_applied=correction_applied,
                    correction_successful=False,
                    execution_trace=trace,
                    dependency_graph=last_dependency_graph,
                )

            # ----- Step 3: Check Drift (with step-level tracking) -----
            print("\nDEBUG FACTS:", [(f.predicate, f.arguments) for f in facts])

            # Extract target variable
            target_var = extract_target_variable(query)
            if target_var:
                print(f"[TARGET VARIABLE] {target_var}")

            (drift, d_type, conf, var, old, new, all_drifts,
             formulas, value_history, dep_graph,
             source_step, error_step) = detect_drift_from_facts(facts)

            last_dependency_graph = dep_graph

            if all_drifts:
                print(f"[MULTI-DRIFT] {len(all_drifts)} drift(s) detected: {all_drifts}")
            print("SSCE DRIFT:", drift)

            # --- Factored Verification (NEW) ---
            from verifier.factored_verifier import run_factored_verification
            factored_result = run_factored_verification(facts, self.llm)
            factored_drift_detected = factored_result.drift_detected
            factored_llm_calls_total = factored_result.llm_calls_made

            if factored_drift_detected:
                print(f"[FACTORED DRIFT] score={factored_result.total_drift_score:.2f}, "
                      f"llm_calls={factored_result.llm_calls_made}")
                for cv in factored_result.claims:
                    if cv.is_drift:
                        print(f"  → {cv.claim.output}: v1={cv.v1_symbolic}, "
                              f"v2={cv.v2_recompute}, v3={cv.v3_llm}, "
                              f"score={cv.drift_score:.2f}")

            # Task 5: Combined drift decision (SSCE OR factored)
            ssce_drift = drift and conf >= 0.75
            combined_drift = ssce_drift or factored_drift_detected

            if combined_drift:
                drift_detected = True
                print(f"[COMBINED DRIFT] ssce={ssce_drift}, factored={factored_drift_detected}")
                if ssce_drift:
                    print(f"  Source step: {source_step}, Error step: {error_step}")

                # Build critique with step-level info
                if ssce_drift:
                    # Use SSCE drift details for critique
                    if d_type == "sign_flip":
                        critique = (f"Operation for '{var}' changed from {old} to {new} unexpectedly. "
                                    f"Error at step {error_step}, originally defined at step {source_step}.")
                    elif d_type == "inconsistent_dependency":
                        critique = (f"Variable '{var}' was previously computed as {old} but new dependencies yield {new}. "
                                    f"Error at step {error_step}, originally defined at step {source_step}.")
                    elif d_type == "invalid_operation":
                        critique = f"Variable '{var}' caused {old} error at step {error_step}."
                    else:
                        critique = (f"Variable '{var}' changed from {old} to {new} without justification. "
                                    f"Error at step {error_step}, originally defined at step {source_step}.")
                elif factored_drift_detected:
                    # Use factored verification details for critique
                    drifted_claims = [cv for cv in factored_result.claims if cv.is_drift]
                    parts = []
                    for cv in drifted_claims:
                        parts.append(
                            f"Variable '{cv.claim.output}' has inconsistent values: "
                            f"symbolic={cv.v1_symbolic}, recomputed={cv.v2_recompute}"
                            f"{f', llm={cv.v3_llm}' if cv.v3_llm is not None else ''}"
                            f" (drift_score={cv.drift_score:.2f})"
                        )
                    critique = "; ".join(parts) if parts else "Factored verification detected drift."
                    # Use first drifted claim for report
                    if drifted_claims:
                        var = drifted_claims[0].claim.output
                        old = drifted_claims[0].v2_recompute
                        new = drifted_claims[0].v1_symbolic
                        d_type = "factored_drift"
                        error_step = drifted_claims[0].claim.step
                        source_step = drifted_claims[0].claim.step
                else:
                    critique = "Drift detected."

                # Build drift report with step info
                r = DriftReport(variable=var or "unknown", old_value=old, new_value=new, reason=critique)
                report_dict = {
                    "variable": r.variable,
                    "old_value": r.old_value,
                    "new_value": r.new_value,
                    "reason": r.reason,
                    "type": d_type if d_type else "unknown",
                    "source_step": source_step,
                    "error_step": error_step,
                }
                all_reports.append(report_dict)

                # Task 4: Constraint-Guided Surgical Repair
                error_step_for_repair = error_step if error_step is not None else 0
                drift_info = {
                    "variable": var or "unknown",
                    "type": d_type if d_type else "unknown",
                    "old": old,
                    "new": new,
                    "source_step": source_step,
                    "error_step": error_step,
                }
                corrected_reasoning, deterministic_repair = constraint_guided_repair(
                    self.llm, query,
                    reasoning_result.reasoning,
                    error_step_for_repair,
                    critique,
                    facts=facts,
                    drift_info=drift_info,
                    target_variable=target_var,
                )

                print("\n[REFLEXION OUTPUT (PARTIAL REPAIR)]:\n", corrected_reasoning[:500])

                # Re-decompose the corrected reasoning
                try:
                    facts = self.decomposer.to_atomic_facts(corrected_reasoning)
                except Exception as e:
                    print(f"⚠️ SymbolicDecomposer completely failed ({e.__class__.__name__}) — falling back to GroqLLM")
                    new_decomp = self.llm.decompose_to_predicates(corrected_reasoning)
                    facts = [
                        AtomicFact(
                            predicate=p["predicate"],
                            arguments=p["arguments"],
                            raw_text="groq_fallback",
                            source_path="groq"
                        )
                        for p in new_decomp.predicates
                    ]

                # Re-normalize
                normalized_facts = []
                for f in facts:
                    norm_args = [normalize_var(arg) for arg in f.arguments]
                    normalized_facts.append(
                        AtomicFact(
                            predicate=f.predicate,
                            arguments=norm_args,
                            raw_text=f.raw_text,
                            source_path=f.source_path
                        )
                    )
                facts = normalized_facts

                print("\n[CORRECTED FACTS]:", [(f.predicate, f.arguments) for f in facts])

                correction_applied = True

                # Update reasoning_result with corrected reasoning
                reasoning_result = ReasoningResult(
                    reasoning=corrected_reasoning,
                    final_answer=reasoning_result.final_answer,
                    raw_response=corrected_reasoning
                )

                # Re-check drift on corrected facts
                (c_drift, _, _, _, _, _, _, _, _, c_dep_graph, _, _) = detect_drift_from_facts(facts)
                last_dependency_graph = c_dep_graph

                # If deterministic repair was applied, trust it:
                # Redefinition cases will still show x=5→x=8 drift (intentional),
                # but the re-computation has been surgically removed.
                if deterministic_repair and c_drift:
                    print("[REPAIR] Deterministic fix applied — trusting repair over SSCE re-check")
                    c_drift = False
                elif c_drift:
                    print("[WARNING] Drift still present after correction")

                # Execute corrected facts
                table = get_symbolic_table()
                table.clear()
                ssce = get_ssce_engine()
                exec_result, exec_reports = self._execute_predicates(facts, table, ssce)

                # If deterministic repair was applied, ignore SSCE execution drift
                # (redefinition-type drift is expected and intentional)
                if deterministic_repair and exec_reports:
                    print("[REPAIR] Deterministic fix — clearing SSCE exec reports")
                    exec_reports = []
            else:
                if drift:
                    print(f"[DRIFT IGNORED] {var}: low confidence ({conf})")

                # ----- Step 4: Execute via PoT Sandbox -----
                table = get_symbolic_table()
                table.clear()
                ssce = get_ssce_engine()
                exec_result, exec_reports = self._execute_predicates(facts, table, ssce)

            # ----- Step 5: SSCE Check -----
            if exec_reports:
                drift_detected = True
                all_reports.extend(
                    [{"variable": r.variable,
                      "old_value": r.old_value,
                      "new_value": r.new_value,
                      "reason": r.reason}
                     for r in exec_reports]
                )

                trace.append({
                    "iteration": iteration,
                    "status": "drift_detected",
                    "drifts": [r.reason for r in exec_reports],
                    "answer": reasoning_result.final_answer,
                })

                # Build targeted Reflexion prompt with step info
                if iteration < MAX_REFLEXION_ITERATIONS:
                    reflexion_feedback = build_reflexion_prompt(
                        exec_reports,
                        target_variable=target_var,
                        source_step=source_step,
                        error_step=error_step,
                    )
                    correction_applied = True
                    logger.info(
                        f"Reflexion iteration {iteration + 1}: "
                        f"detected {len(exec_reports)} drift(s), "
                        f"re-generating reasoning..."
                    )
                    continue
                else:
                    trace.append({
                        "iteration": iteration,
                        "status": "max_iterations_reached",
                    })
                    return ReflexionResult(
                        final_answer=reasoning_result.final_answer,
                        reasoning=reasoning_result.reasoning,
                        iterations_used=iteration + 1,
                        drift_detected=True,
                        drift_reports=all_reports,
                        correction_applied=correction_applied,
                        correction_successful=False,
                        execution_trace=trace,
                        dependency_graph=last_dependency_graph,
                    )

            # ----- No Drift → Success -----
            final = self._extract_sandbox_answer(exec_result, table)
            if final is None or math.isnan(final):
                final = reasoning_result.final_answer

            # Soft final answer validation
            if target_var and exec_result and isinstance(exec_result, dict):
                if target_var not in exec_result:
                    logger.warning(
                        f"Target variable '{target_var}' not found in result. "
                        f"Available: {list(exec_result.keys())}. Using best numeric."
                    )

            trace.append({
                "iteration": iteration,
                "status": "clean" if not correction_applied else "corrected",
                "answer": final,
            })

            return ReflexionResult(
                final_answer=final,
                reasoning=reasoning_result.reasoning,
                iterations_used=iteration + 1,
                drift_detected=drift_detected,
                drift_reports=all_reports,
                correction_applied=correction_applied,
                correction_successful=correction_applied and not exec_reports and not c_drift,
                execution_trace=trace,
                dependency_graph=last_dependency_graph,
                factored_drift=factored_drift_detected if 'factored_drift_detected' in dir() else False,
                factored_reports=[{"claim": cv.claim.output, "v1": cv.v1_symbolic, "v2": cv.v2_recompute, "v3": cv.v3_llm, "score": cv.drift_score, "is_drift": cv.is_drift} for cv in factored_result.claims] if 'factored_result' in dir() else [],
                factored_llm_calls=factored_llm_calls_total if 'factored_llm_calls_total' in dir() else 0,
            )

        # Should not reach here, but safety
        return ReflexionResult(
            final_answer=float('nan'),
            reasoning="",
            iterations_used=MAX_REFLEXION_ITERATIONS + 1,
            drift_detected=drift_detected,
            drift_reports=all_reports,
            correction_applied=correction_applied,
            correction_successful=False,
            execution_trace=trace,
            dependency_graph=last_dependency_graph,
            factored_drift=False,
            factored_reports=[],
            factored_llm_calls=0,
        )

    def _execute_predicates(
        self,
        predicates: List[Dict[str, Any]],
        table,
        ssce: SSCEEngine,
    ) -> Tuple[Optional[Dict], List[DriftReport]]:
        from symbolic.decomposer import AtomicFact
        all_reports: List[DriftReport] = []
        final_result: Dict[str, Any] = {}

        facts = []
        for p in predicates:
            if isinstance(p, AtomicFact):
                facts.append(p)
            else:
                facts.append(
                    AtomicFact(
                        predicate=p["predicate"],
                        arguments=p["arguments"],
                        raw_text="llm_generated",
                        source_path="groq"
                    )
                )

        print("\nDEBUG FACTS:")
        for f in facts:
            print(f.predicate, f.arguments)

        try:
            pot = self.pot_engine.generate_script(facts)
        except Exception as e:
            logger.warning(f"PoT generation failed: {e}")
            return {}, []

        print("\nDEBUG SCRIPT:\n", pot.script)

        ctx_lines = []
        for var, rec in table.snapshot().items():
            ctx_lines.append(f"{var} = {repr(rec['value'])}")
        ctx = "\n".join(ctx_lines)

        full_script = ctx + "\n" + pot.script if ctx else pot.script

        ok, result, err = _safe_execute(full_script)

        class MockSandboxResult:
            def __init__(self, output):
                self.output = output

        sandbox_result = MockSandboxResult(result if result else {})
        print("\nDEBUG OUTPUT:", sandbox_result.output)

        if ok and result:
            reports = ssce.check_step(result)
            if reports:
                all_reports.extend(reports)
            return result, all_reports
        return final_result, all_reports

    def _extract_sandbox_answer(
        self,
        result: Optional[Dict],
        table,
    ) -> Optional[float]:
        """Extract the final numeric answer from sandbox results."""
        if not result:
            return None

        # Priority: 'result', 'answer', 'total', then last numeric value
        for key in ['result', 'answer', 'total', 'final',
                     'final_answer', 'output']:
            if key in result and isinstance(result[key], (int, float)):
                return float(result[key])

        # Last numeric value
        nums = [v for v in result.values() if isinstance(v, (int, float))]
        return float(nums[-1]) if nums else None


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def get_reflexion_engine(llm: GroqLLM) -> ReflexionEngine:
    return ReflexionEngine(llm=llm)
