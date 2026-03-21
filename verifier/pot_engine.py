"""
verifier/pot_engine.py

Logic-Grounded Pelican (LGP)
---------------------------------
Program-of-Thought (PoT) Generator

Purpose:
Translate validated AtomicFacts into deterministic, sandbox-executable
Python scripts.

Design Constraints:
- Deterministic code generation
- Restricted imports (math, sympy only)
- No dynamic execution features
- No file system access
- No network calls
- Explicit variable assignment tracking

This module DOES NOT execute code.
Execution is delegated to verifier/sandbox.py.

Author: LGP Framework
"""

from __future__ import annotations

import logging
from typing import List, Dict
from dataclasses import dataclass

from symbolic.decomposer import AtomicFact


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("LGP.PoTEngine")
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Execution Payload
# ---------------------------------------------------------------------------

@dataclass
class PoTScript:
    """
    Structured representation of generated PoT script.

    Attributes:
        script: Python code string
        output_variables: Variables expected in final namespace
    """
    script: str
    output_variables: List[str]

    def to_dict(self) -> Dict:
        return {
            "script": self.script,
            "output_variables": self.output_variables,
        }


# ---------------------------------------------------------------------------
# PoT Engine
# ---------------------------------------------------------------------------

class PoTEngine:
    """
    Converts AtomicFacts into executable Python scripts.

    Follows Minerva-style deterministic program synthesis.
    """

    ALLOWED_IMPORTS = ["math", "sympy"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_script(self, facts: List[AtomicFact]) -> PoTScript:
        """
        Convert atomic facts into a safe Python script.

        Raises:
            ValueError if facts list is empty or malformed
        """
        if not facts:
            raise ValueError("Cannot generate PoT script from empty fact list.")

        lines: List[str] = []
        output_vars: List[str] = []

        # --- Safe Imports ---
        lines.append("import math")
        lines.append("import sympy")
        lines.append("")

        # --- Generate deterministic code ---
        for fact in facts:
            code_line, target_var = self._translate_fact(fact)
            lines.append(code_line)

            if target_var:
                output_vars.append(target_var)

        # --- Final Output Packaging ---
        lines.append("")
        lines.append("__result__ = {}")
        for var in output_vars:
            lines.append(f"__result__['{var}'] = {var}")

        script = "\n".join(lines)

        logger.debug("Generated PoT script:\n" + script)

        return PoTScript(script=script, output_variables=output_vars)

    # ------------------------------------------------------------------
    # Fact Translation
    # ------------------------------------------------------------------

    def _translate_fact(self, fact: AtomicFact):
        """
        Translate AtomicFact into Python assignment.

        Supported predicates:
            Add(a, b, c)              → c = a + b
            Subtract(a, b, c)         → c = a - b
            Multiply(a, b, c)         → c = a * b
            Divide(a, b, c)           → c = a / b
            Assign(a, b)              → b = a
            GreaterThan(a, b)         → _gt_a_b = (a > b)
            LessThan(a, b)            → _lt_a_b = (a < b)
            Equals(a, b)             → _eq_a_b = (a == b)
            Conditional(cond, t, e)   → _cond_result = t if cond else e
        """

        pred = fact.predicate
        args = fact.arguments

        if pred == "Add":
            a, b, c = args
            return f"{c} = {a} + {b}", c

        if pred == "Subtract":
            a, b, c = args
            return f"{c} = {a} - {b}", c

        if pred == "Multiply":
            a, b, c = args
            return f"{c} = {a} * {b}", c

        if pred == "Divide":
            a, b, c = args
            return f"{c} = {a} / {b}", c

        if pred == "Assign":
            a, b = args
            return f"{b} = {a}", b

        if pred == "GreaterThan":
            a, b = args
            result_var = f"_gt_{a}_{b}"
            return f"{result_var} = ({a} > {b})", result_var

        if pred == "LessThan":
            a, b = args
            result_var = f"_lt_{a}_{b}"
            return f"{result_var} = ({a} < {b})", result_var

        if pred == "Equals":
            a, b = args
            result_var = f"_eq_{a}_{b}"
            return f"{result_var} = ({a} == {b})", result_var

        if pred == "Conditional":
            cond, then_var, else_var = args
            result_var = f"_cond_result"
            return (
                f"{result_var} = {then_var} if {cond} else {else_var}",
                result_var,
            )

        raise ValueError(f"Unsupported predicate for PoT translation: {pred}")


# ---------------------------------------------------------------------------
# Convenience Accessor
# ---------------------------------------------------------------------------


def get_pot_engine() -> PoTEngine:
    """
    Public accessor for PoTEngine.
    """
    return PoTEngine()
