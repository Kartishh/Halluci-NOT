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

from symbolic.decomposer import extract_equations
from symbolic.ssce_algorithm import (
    SSCEEngine,
    SSCEEnforcementError,
    DriftReport,
    get_ssce_engine,
)
from symbolic.table import get_symbolic_table
from verifier.pot_engine import get_pot_engine, PoTScript
from core.gemini_llm import GeminiLLM, ReasoningResult, DecompositionResult

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
    import ast

    # Strip import lines — math is pre-injected
    lines = script.split("\n")
    lines = [l for l in lines if not _re.match(r'^\s*import\s+', l)]
    script = "\n".join(lines)

    safe_builtins = {
        "abs": abs, "round": round, "int": int, "float": float,
        "str": str, "len": len, "min": min, "max": max,
        "True": True, "False": False, "None": None,
        "isinstance": isinstance, "print": lambda *a, **kw: None,
    }
    restricted_globals = {"__builtins__": safe_builtins, "math": _math}
    
    # Task 6: Safe execution guard (abort on undefined variable, return empty result)
    try:
        tree = ast.parse(script)
        defined = set(safe_builtins.keys()) | {'math', '__result__'}
        # Assuming simple linear PoT script without complex control flow
        for node in tree.body:
            # Check all loaded names in this statement
            for subnode in ast.walk(node):
                if isinstance(subnode, ast.Name) and isinstance(subnode.ctx, ast.Load):
                    if subnode.id not in defined:
                        return True, {}, f"Undefined variable: {subnode.id}"
            
            # Add assigned variables to defined set for subsequent statements
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        defined.add(target.id)
    except Exception as e:
        return False, {}, f"Execution failed: {e}"

    local_ns: Dict[str, Any] = {}

    try:
        exec(script, restricted_globals, local_ns)
        result = local_ns.get("__result__", {})
        if not isinstance(result, dict):
            return False, None, "No __result__ dict in output"
        if not result:
            return False, {}, "Empty execution result"
        return True, result, ""
    except Exception as e:
        return False, {}, str(e)[:300]


# ---------------------------------------------------------------------------
# Target Variable Extraction
# ---------------------------------------------------------------------------

def extract_target_variable(question: str, computed_vars=None) -> Optional[str]:
    """
    Extract the target variable from a question.
    PRIMARY: use last computed variable from execution results.
    FALLBACK: return None (disable broken heuristic).
    """
    # PRIMARY: use last computed variable
    if computed_vars:
        return list(computed_vars.keys())[-1]

    # fallback: None (disable broken heuristic)
    return None


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
                    if abs(values[var] - val) > 1e-2:
                        add_drift("dependency_mutation", 0.85, var,
                                  values[var], val,
                                  source_step=source_step,
                                  error_step=step_index)
                else:
                    if abs(values[var] - val) > 1e-2:
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
                    # Task 4: Relax dependency mutation
                    v1_val = resolve(arg1)
                    v2_val = resolve(arg2)
                    values_differ = True
                    if res in values and v1_val is not None and v2_val is not None:
                        expected_new = None
                        if pred == "Add": expected_new = v1_val + v2_val
                        elif pred == "Subtract": expected_new = v1_val - v2_val
                        elif pred == "Multiply": expected_new = v1_val * v2_val
                        elif pred == "Divide" and v2_val != 0: expected_new = v1_val / v2_val
                        
                        if expected_new is not None:
                            values_differ = abs(values[res] - expected_new) > 1e-2

                    if values_differ:
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
                    if res in values and abs(values[res] - expected) > 1e-2:
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
                    if abs(curr["value"] - first["value"]) > 1e-2:
                        add_drift("redefinition", 0.8, var, first["value"], curr["value"], 
                                  source_step=first["step"], error_step=curr["step"])

    # --- Task 3: Execution Validation Post-Pass (NEW) ---
    # Recompute ALL formula outputs from current variable values.
    # Catches numeric drift that inline checks may miss when
    # upstream values changed after the formula was first evaluated.
    NUMERIC_TOLERANCE = 1e-2
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
    result_mismatch: bool = False,
    computed_value: Optional[float] = None,
    llm_claimed: Optional[float] = None,
) -> str:
    """
    Build a TARGETED reflexion prompt that cites the exact step
    where the error occurred and instructs LLM to fix only that step
    and steps after it.
    """
    lines = []
    
    # Handle result mismatch case - regenerate COMPLETE solution
    if result_mismatch and computed_value is not None and llm_claimed is not None:
        lines = [
            "You previously produced incorrect equations.",
            "",
            f"Expected: {computed_value}",
            f"Your answer: {llm_claimed}",
            "",
            "Re-generate the FULL solution from scratch using equations.",
            "",
            "Rules:",
            "1. Output ONLY equations",
            "2. Include ALL steps from start to final result",
            "3. Every variable must be defined before use",
            "4. Do NOT skip intermediate steps",
            "5. Final line must compute the result",
            "",
            "Example:",
            "a = 10",
            "b = 20",
            "c = a + b",
            "result = c",
            "",
            "Do NOT output partial fixes.",
        ]
        if target_variable:
            lines.append(f"Final answer must compute: {target_variable}")
    else:
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
    llm: GeminiLLM,
    query: str,
    reasoning: str,
    error_step_idx: int,
    critique: str,
    target_variable: Optional[str] = None,
) -> str:
    """
    Simple repair for equation-based reasoning.
    Ask LLM to regenerate complete solution given the critique.
    """
    fix_prompt = (
        f"Problem: {query}\n\n"
        f"Your previous solution had an error: {critique}\n\n"
        f"Generate a new solution using ONLY equations.\n"
        f"Each line: variable = expression\n"
        f"Define all variables before use.\n"
        f"Final line must compute the result.\n"
        f"Do NOT include any text or explanation.\n"
    )
    if target_variable:
        fix_prompt += f"The final answer should be in a variable called '{target_variable}' or 'result'.\n"

    try:
        result = llm.generate_reasoning(query, reflexion_feedback=fix_prompt)
        return result.reasoning
    except Exception as e:
        logger.warning(f"Repair failed: {e}. Returning original.")
        return reasoning


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
    llm: GeminiLLM,
    query: str,
    reasoning: str,
    error_step_idx: int,
    critique: str,
    drift_info: Dict[str, Any],
    target_variable: Optional[str] = None,
) -> str:
    """
    Simple repair: ask LLM to regenerate reasoning given the critique.
    """
    try:
        print("[REPAIR] Using LLM to regenerate reasoning")
        corrected = partial_repair(
            llm, query, reasoning, 0, critique, target_variable
        )
        return corrected
    except Exception as e:
        logger.warning(f"Repair failed: {e}. Returning original reasoning.")
        return reasoning


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
    llm: GeminiLLM,
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
        from core.nvidia_llm import _safe_extract_text
        fixed_text = _safe_extract_text(response)

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

    def __init__(self, llm: GeminiLLM):
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

            # ----- Step 1: Generate Independent Answer FIRST -----
            if forced_reasoning is not None and iteration == 0:
                reasoning_result = ReasoningResult(
                    reasoning=forced_reasoning,
                    final_answer=float("nan"),
                    raw_response=forced_reasoning
                )
                original_llm_answer = float("nan")
            else:
                # Get independent answer BEFORE generating reasoning
                original_llm_answer = self.llm.generate_answer_only(query)
                # Then generate reasoning
                reasoning_result = self.llm.generate_reasoning(
                    query, reflexion_feedback=reflexion_feedback
                )

            # ----- Step 2: Extract Equations -----
            force_drift = False
            equations = []
            try:
                equations = self.decomposer(reasoning_result.reasoning)
            except Exception as e:
                print(f"⚠️ Equation extraction failed ({e.__class__.__name__}) — forcing drift")
                force_drift = True

            if not equations:
                print("⚠️ No equations extracted — triggering drift repair")
                drift = True
                drift_reason = "decomposition_failure"
                if not all_reports:
                    all_reports = [{"variable": target_var or "result",
                      "old_value": None,
                      "new_value": None,
                      "reason": "decomposition_failure"}]
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

            print("\n===== DEBUG EQUATIONS =====")
            print("REASONING:\n", reasoning_result.reasoning[:300])
            print("EQUATIONS:", equations)
            print("==========================\n")

            # ----- Step 3: Execute Equations and Check Result Mismatch -----
            print("\nDEBUG EQUATIONS:", equations)

            # Extract target variable
            target_var = extract_target_variable(query)
            if target_var:
                print(f"[TARGET VARIABLE] {target_var}")

            # Execute equations directly
            temp_table = get_symbolic_table()
            temp_table.clear()
            temp_exec_res, _, exec_ok = self._execute_predicates(equations, temp_table, get_ssce_engine())
            computed_value = self._extract_computed_value(temp_exec_res)

            # DRIFT CHECK: Compare computed value vs INDEPENDENT original answer
            disagreement = False
            if computed_value is not None and original_llm_answer is not None:
                try:
                    if not math.isnan(computed_value) and not math.isnan(original_llm_answer):
                        disagreement = abs(float(computed_value) - float(original_llm_answer)) > 1e-2
                except (ValueError, TypeError):
                    pass

            print(f"[DRIFT DEBUG] computed={computed_value}, original={original_llm_answer}, disagreement={disagreement}, exec_ok={exec_ok}")

            # Force drift on execution failure or result mismatch
            combined_drift = False
            if not exec_ok or (computed_value is None):
                print("[FORCE DRIFT] Execution failed")
                combined_drift = True
            elif disagreement:
                print(f"[RESULT MISMATCH] computed={computed_value}, original={original_llm_answer}")
                combined_drift = True

            if combined_drift:
                drift_detected = True
                print(f"[COMBINED DRIFT] result_mismatch={disagreement}, exec_failed={not exec_ok}")

                # Simple drift report for result mismatch
                critique = f"Result mismatch: computed={computed_value}, LLM original={original_llm_answer}"
                report_dict = {
                    "variable": target_var or "result",
                    "old_value": original_llm_answer,
                    "new_value": computed_value,
                    "reason": critique,
                    "type": "result_mismatch",
                }
                all_reports.append(report_dict)

                # Build reflexion prompt
                reflexion_feedback = build_reflexion_prompt(
                    [report_dict],
                    target_variable=target_var,
                    result_mismatch=True,
                    computed_value=computed_value,
                    llm_claimed=original_llm_answer,
                )

                # Call repair to regenerate reasoning
                corrected_reasoning = constraint_guided_repair(
                    self.llm, query,
                    reasoning_result.reasoning,
                    0,
                    critique,
                    drift_info={"variable": target_var or "result", "type": "result_mismatch"},
                    target_variable=target_var,
                )

                # Handle empty repair output
                if not corrected_reasoning or len(corrected_reasoning.strip()) == 0:
                    print("[WARNING] Empty repair output — falling back to original reasoning")
                    corrected_reasoning = reasoning_result.reasoning

                print("\n[REFLEXION OUTPUT (PARTIAL REPAIR)]:\n", corrected_reasoning[:500])

                # Re-extract equations from corrected reasoning
                try:
                    equations = self.decomposer(corrected_reasoning)
                except Exception as e:
                    print(f"⚠️ Equation extraction failed ({e.__class__.__name__})")
                    equations = []

                print("\n[CORRECTED EQUATIONS]:", equations)

                correction_applied = True

# Update reasoning_result with corrected reasoning
                reasoning_result = ReasoningResult(
                    reasoning=corrected_reasoning,
                    final_answer=reasoning_result.final_answer,
                    raw_response=corrected_reasoning
                )

                # Execute corrected equations
                table = get_symbolic_table()
                table.clear()
                ssce = get_ssce_engine()
                exec_result, exec_reports, exec_ok = self._execute_predicates(equations, table, ssce)

                # Re-check result mismatch vs original
                new_computed = self._extract_computed_value(exec_result)
                if new_computed is not None and original_llm_answer is not None:
                    try:
                        if not math.isnan(new_computed) and not math.isnan(original_llm_answer):
                            if abs(float(new_computed) - float(original_llm_answer)) > 1e-2:
                                print(f"[WARNING] Drift still present after correction: computed={new_computed}, original={original_llm_answer}")
                    except (ValueError, TypeError):
                        pass
            else:
                # ----- Step 4: Execute via PoT Sandbox -----
                table = get_symbolic_table()
                table.clear()
                ssce = get_ssce_engine()
                exec_result, exec_reports, exec_ok = self._execute_predicates(equations, table, ssce)

            # ----- Step 5: Extract Final Answer -----
            final_computed_value = self._extract_computed_value(exec_result)

            # Use computed value if available, otherwise use original LLM answer
            final_answer = final_computed_value if final_computed_value is not None else original_llm_answer

            trace.append({
                "iteration": iteration,
                "status": "clean" if not correction_applied else "corrected",
                "answer": final_answer,
            })

            return ReflexionResult(
                final_answer=final_answer,
                reasoning=reasoning_result.reasoning,
                iterations_used=iteration + 1,
                drift_detected=drift_detected,
                drift_reports=all_reports,
                correction_applied=correction_applied,
                correction_successful=correction_applied and not exec_reports,
                execution_trace=trace,
                dependency_graph=last_dependency_graph,
            )

    def _execute_predicates(
        self,
        equations_or_facts,
        table,
        ssce: SSCEEngine,
    ) -> Tuple[Optional[Dict], List[DriftReport], bool]:
        all_reports: List[DriftReport] = []
        final_result: Dict[str, Any] = {}

        # Handle both equation strings and AtomicFacts
        if equations_or_facts and isinstance(equations_or_facts[0], str):
            equations = equations_or_facts
        else:
            facts = equations_or_facts
            from symbolic.decomposer import AtomicFact
            equations = []
            for f in facts:
                if isinstance(f, AtomicFact):
                    pred = f.predicate
                    args = f.arguments
                    if pred == "Assign":
                        equations.append(f"{args[1]} = {args[0]}")
                    elif pred == "Add":
                        equations.append(f"{args[2]} = {args[0]} + {args[1]}")
                    elif pred == "Subtract":
                        equations.append(f"{args[2]} = {args[0]} - {args[1]}")
                    elif pred == "Multiply":
                        equations.append(f"{args[2]} = {args[0]} * {args[1]}")
                    elif pred == "Divide":
                        equations.append(f"{args[2]} = {args[0]} / {args[1]}")

        print("\nDEBUG EQUATIONS:")
        for eq in equations:
            print(eq)

        try:
            pot = self.pot_engine.generate_script(equations)
        except Exception as e:
            logger.warning(f"PoT generation failed: {e}")
            return {}, [], False

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

        if ok and result is not None:
            reports = ssce.check_step(result)
            if reports:
                all_reports.extend(reports)
            return result, all_reports, ok
        return final_result, all_reports, ok

    def _extract_sandbox_answer(
        self,
        result: Optional[Dict],
        table,
        target_variable: Optional[str] = None,
    ) -> Optional[float]:
        """Extract the final numeric answer from sandbox results."""
        if not result:
            return None

        # Fallback if no target given
        if not target_variable:
            for key in ['result', 'answer', 'total', 'final', 'final_answer', 'output']:
                if key in result and isinstance(result[key], (int, float)):
                    return float(result[key])
            nums = [v for v in result.values() if isinstance(v, (int, float))]
            return float(nums[-1]) if nums else None

        # 1. Exact match
        if target_variable in result and isinstance(result[target_variable], (int, float)):
            return float(result[target_variable])

        # 2. Fuzzy match
        norm_target = target_variable.lower().replace("_", "")
        candidates = []
        for k, v in result.items():
            if isinstance(v, (int, float)):
                norm_k = k.lower().replace("_", "")
                if norm_target in norm_k or norm_k in norm_target:
                    candidates.append(k)

        if not candidates:
            # Fallback: use last computed value
            if result:
                nums = [v for v in result.values() if isinstance(v, (int, float))]
                return float(nums[-1]) if nums else None
            return None

        if len(candidates) == 1:
            return float(result[candidates[0]])

        # 3. If multiple candidates: choose variable used in final step
        # Since result is an ordered dict, the last matching key is from the final step
        for k in reversed(list(result.keys())):
            if k in candidates:
                return float(result[k])

        # Ultimate fallback: last numeric value
        if result:
            nums = [v for v in result.values() if isinstance(v, (int, float))]
            return float(nums[-1]) if nums else None
        return None

    def _extract_computed_value(self, result: Optional[Dict]) -> Optional[float]:
        """Extract computed value from execution result - simple version."""
        if not result:
            return None
        
        # If 'result' key exists, use it
        if 'result' in result and isinstance(result['result'], (int, float)):
            return float(result['result'])
        
        # Fallback: last numeric value
        nums = [v for v in result.values() if isinstance(v, (int, float))]
        return float(nums[-1]) if nums else None


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def get_reflexion_engine(llm: GeminiLLM) -> ReflexionEngine:
    return ReflexionEngine(llm=llm)
