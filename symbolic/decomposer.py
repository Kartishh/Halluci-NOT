"""
symbolic/decomposer.py

HalluciNOT (LGP)
---------------------------------
Direct Equation Extractor

Extracts clean equation lines from LLM reasoning for direct execution.
No predicate parsing - just extract and pass through.
"""

from __future__ import annotations


def extract_equations(text: str) -> list:
    """
    Extract clean equation lines from reasoning text.
    
    Args:
        text: LLM reasoning text containing equations
        
    Returns:
        List of clean equation strings (e.g., ["x = 5", "y = x + 3"])
    """
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if "=" in line and "==" not in line and not line.startswith("#") and not line.startswith("//"):
            lines.append(line)
    return lines


def get_symbolic_decomposer():
    """Public accessor - returns simple extract_equations function."""
    return extract_equations