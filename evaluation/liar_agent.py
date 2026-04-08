"""
evaluation/liar_agent.py

HalluciNOT (LGP) — Liar Agent (Drift Injector)
=================================================

Purpose:
    Intentionally inject symbolic drift into valid LLM reasoning
    to stress-test SSCE detection and Reflexion correction.

Drift Types:
    1. REDEFINITION — silently redefine a variable mid-reasoning
    2. SIGN_FLIP — flip + to - or vice versa
    3. VALUE_SWAP — swap two variable values
    4. STALE_REUSE — reuse a variable's old value after update
"""

from __future__ import annotations

import random
import re
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger("LGP.LiarAgent")


# ---------------------------------------------------------------------------
# Drift Types
# ---------------------------------------------------------------------------

class DriftType:
    REDEFINITION = "redefinition"
    SIGN_FLIP = "sign_flip"
    VALUE_SWAP = "value_swap"
    STALE_REUSE = "stale_reuse"


@dataclass
class DriftInjection:
    """Record of what drift was injected."""
    drift_type: str
    original_reasoning: str
    flawed_reasoning: str
    correct_answer: float
    drift_details: str


# ---------------------------------------------------------------------------
# Liar Agent
# ---------------------------------------------------------------------------

class LiarAgent:
    """
    Intentionally injects symbolic drift into valid CoT reasoning.

    Used to test SSCE detection and Reflexion correction.
    """

    def inject_drift(
        self,
        reasoning: str,
        correct_answer: float,
        drift_type: Optional[str] = None,
    ) -> DriftInjection:
        """
        Inject a specific drift type into reasoning.

        If drift_type is None, chooses randomly.
        """
        if drift_type is None:
            drift_type = random.choice([
                DriftType.REDEFINITION,
                DriftType.SIGN_FLIP,
                DriftType.VALUE_SWAP,
                DriftType.STALE_REUSE,
            ])

        injectors = {
            DriftType.REDEFINITION: self._inject_redefinition,
            DriftType.SIGN_FLIP: self._inject_sign_flip,
            DriftType.VALUE_SWAP: self._inject_value_swap,
            DriftType.STALE_REUSE: self._inject_stale_reuse,
        }

        injector = injectors.get(drift_type, self._inject_redefinition)
        flawed, details = injector(reasoning)

        return DriftInjection(
            drift_type=drift_type,
            original_reasoning=reasoning,
            flawed_reasoning=flawed,
            correct_answer=correct_answer,
            drift_details=details,
        )

    # ------------------------------------------------------------------
    # Drift Injectors
    # ------------------------------------------------------------------

    def _inject_redefinition(self, reasoning: str) -> Tuple[str, str]:
        """
        Silently redefine a variable mid-reasoning.

        Example:
            Step 1: x = 5
            Step 2: y = x * 3 = 15
            INJECTED: x = 8  (unjustified redefinition)
            Step 3: z = x + y = 8 + 15 = 23  (should be 5 + 15 = 20)
        """
        # Find variable assignments like "x = 5" or "variable = number"
        assignments = re.findall(
            r'(\b([a-zA-Z_]\w*)\s*=\s*(\d+\.?\d*))',
            reasoning
        )

        if len(assignments) < 2:
            return reasoning, "insufficient_assignments"

        # Pick a variable that's assigned early, redefine it later
        target = assignments[0]
        var_name = target[1]
        original_val = target[2]

        # Generate a different value
        new_val = str(int(float(original_val)) + random.randint(2, 8))

        # Find the position after the second assignment
        lines = reasoning.split('\n')
        inject_pos = min(len(lines) - 1, len(lines) // 2 + 1)

        lines.insert(
            inject_pos,
            f"Wait, let me reconsider. Actually {var_name} = {new_val}."
        )

        flawed = '\n'.join(lines)
        details = (
            f"Redefined '{var_name}' from {original_val} to {new_val} "
            f"at line {inject_pos} without justification"
        )

        return flawed, details

    def _inject_sign_flip(self, reasoning: str) -> Tuple[str, str]:
        """
        Flip a + to - or vice versa in a calculation step.

        Example:
            "total = price + tax" → "total = price - tax"
        """
        # Find addition/subtraction patterns
        add_pattern = re.search(r'(\w+\s*=\s*\w+)\s*\+\s*(\w+)', reasoning)
        sub_pattern = re.search(r'(\w+\s*=\s*\w+)\s*-\s*(\w+)', reasoning)

        if add_pattern:
            original = add_pattern.group(0)
            flipped = original.replace('+', '-', 1)
            flawed = reasoning.replace(original, flipped, 1)
            details = f"Flipped '+' to '-': '{original}' → '{flipped}'"
            return flawed, details

        if sub_pattern:
            original = sub_pattern.group(0)
            flipped = original.replace('-', '+', 1)
            flawed = reasoning.replace(original, flipped, 1)
            details = f"Flipped '-' to '+': '{original}' → '{flipped}'"
            return flawed, details

        return reasoning, "no_sign_to_flip"

    def _inject_value_swap(self, reasoning: str) -> Tuple[str, str]:
        """
        Swap the values of two variables.

        Example:
            "price = 10, quantity = 5" → "price = 5, quantity = 10"
        """
        assignments = re.findall(
            r'\b([a-zA-Z_]\w*)\s*=\s*(\d+\.?\d*)',
            reasoning
        )

        if len(assignments) < 2:
            return reasoning, "insufficient_variables"

        var1, val1 = assignments[0]
        var2, val2 = assignments[1]

        flawed = reasoning
        # Swap — replace first occurrence of each
        flawed = re.sub(
            rf'\b{var1}\s*=\s*{re.escape(val1)}\b',
            f'{var1} = {val2}',
            flawed,
            count=1
        )
        flawed = re.sub(
            rf'\b{var2}\s*=\s*{re.escape(val2)}\b',
            f'{var2} = {val1}',
            flawed,
            count=1
        )

        details = f"Swapped '{var1}={val1}' ↔ '{var2}={val2}'"
        return flawed, details

    def _inject_stale_reuse(self, reasoning: str) -> Tuple[str, str]:
        """
        Reuse a variable's old value after it was updated.

        Example:
            Step 1: x = 5
            Step 2: x = x + 3 = 8
            Step 3: y = x * 2 = 10  (uses old x=5 instead of x=8)
        """
        # Find var that's assigned twice
        assignments = re.findall(
            r'\b([a-zA-Z_]\w*)\s*=\s*(\d+\.?\d*)',
            reasoning
        )

        var_counts: Dict[str, List] = {}
        for var, val in assignments:
            var_counts.setdefault(var, []).append(val)

        # Find a var with multiple assignments
        target_var = None
        old_val = None
        new_val = None
        for var, vals in var_counts.items():
            if len(vals) >= 2:
                target_var = var
                old_val = vals[0]
                new_val = vals[-1]
                break

        if not target_var:
            return reasoning, "no_variable_updated_twice"

        # Find usage of the variable AFTER its update, replace with old value
        lines = reasoning.split('\n')
        found_update = False
        for i, line in enumerate(lines):
            if found_update and target_var in line and '=' in line:
                # This line uses the variable after update
                # Replace the variable reference with the old value
                lines[i] = line + f"  [using {target_var}={old_val} from earlier]"
                break
            if f'{target_var} = {new_val}' in line or f'{target_var}={new_val}' in line:
                found_update = True

        flawed = '\n'.join(lines)
        details = (
            f"Stale reuse: '{target_var}' updated to {new_val} "
            f"but later used as {old_val}"
        )
        return flawed, details


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def get_liar_agent() -> LiarAgent:
    return LiarAgent()
