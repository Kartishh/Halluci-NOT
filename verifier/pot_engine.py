"""
verifier/pot_engine.py

Logic-Grounded Pelican (LGP)
---------------------------------
Direct Script Generator

Takes equation lines and builds executable Python script.
No predicate parsing - just pass through equations directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict


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


class PoTEngine:
    """
    Converts equation lines into executable Python scripts.
    Direct pass-through - no reordering, no modification.
    """

    def generate_script(self, equations: List[str]) -> PoTScript:
        """
        Convert equation lines into a safe Python script.

        Args:
            equations: List of equation strings (e.g., ["x = 5", "y = x + 3"])

        Raises:
            ValueError if equations list is empty
        """
        if not equations:
            raise ValueError("Cannot generate script from empty equations.")

        script = "import math\n\n"
        script += "\n".join(equations)
        
        # Build __result__ by explicitly assigning each variable
        output_vars = []
        for eq in equations:
            if "=" in eq:
                var = eq.split("=")[0].strip()
                if var:
                    output_vars.append(var)
        
        script += "\n\n__result__ = {}\n"
        for var in output_vars:
            script += f"if isinstance({var}, (int, float)): __result__['{var}'] = {var}\n"

        return PoTScript(script=script, output_variables=output_vars)


def get_pot_engine() -> PoTEngine:
    """
    Public accessor for PoTEngine.
    """
    return PoTEngine()