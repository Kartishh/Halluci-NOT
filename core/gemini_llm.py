"""
core/gemini_llm.py

HalluciNOT (LGP) — LLM Wrapper (NVIDIA Backend)
==================================================
Backward-compatible re-export layer.

All downstream imports continue working unchanged:
    from core.gemini_llm import GeminiLLM, get_gemini_llm
    from core.gemini_llm import ReasoningResult, DecompositionResult

The actual implementation lives in core/nvidia_llm.py.
"""

from core.nvidia_llm import (
    NvidiaLLM,
    get_nvidia_llm,
    ReasoningResult,
    DecompositionResult,
    SUPPORTED_PREDICATES,
)

# ---------------------------------------------------------------------------
# Backward-compatible aliases
# ---------------------------------------------------------------------------

GeminiLLM = NvidiaLLM

def get_gemini_llm(model: str = "") -> NvidiaLLM:
    """Factory — returns NvidiaLLM instance (drop-in for old GeminiLLM)."""
    return get_nvidia_llm(model=model)