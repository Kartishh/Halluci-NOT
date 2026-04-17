"""
evaluation/baselines.py

HalluciNOT (LGP)
---------------------------------
Baseline Runners for Comparative Evaluation

Provides baseline LLM runners WITHOUT LGP enforcement to measure
the value-add of deterministic symbolic verification.

Baselines:
    1. Vanilla LLM — Direct query → response (no reasoning enforcement)
    2. Chain-of-Thought (CoT) — "Let's think step by step" prompting

Uses Google Gemini API (free tier) for cost-effective research evaluation.

Author: LGP Framework
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Callable, Dict, Optional

from evaluation.datasets import EvalSample
from evaluation.runner import EvalResult

logger = logging.getLogger("LGP.Eval.Baselines")
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Gemini API Client
# ---------------------------------------------------------------------------

from core.gemini_llm import get_gemini_llm

class GeminiClient:
    def __init__(self, model_name: Optional[str] = None):
        self.client = get_gemini_llm()

    def generate(self, prompt: str) -> str:
        try:
            response = self.client._call_gemini_with_fallback(
                prompt=prompt,
                system_prompt="Provide a direct answer.",
                temperature=0.0
            )
            return response
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise

_gemini_client: Optional[GeminiClient] = None

def _get_gemini_client() -> GeminiClient:
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = GeminiClient()
    return _gemini_client


# ---------------------------------------------------------------------------
# Vanilla LLM Baseline
# ---------------------------------------------------------------------------

def run_vanilla_baseline(sample: EvalSample) -> EvalResult:
    """
    Run a sample through vanilla LLM with no enforcement.

    Prompt: Just the query, no special instructions.
    """
    client = _get_gemini_client()

    prompt = (
        f"Answer the following question. "
        f"Provide only the final answer as a number or short phrase.\n\n"
        f"Question: {sample.query}\n\n"
        f"Answer:"
    )

    start_time = time.time()
    error_msg = None
    predicted = None

    try:
        response = client.generate(prompt)
        predicted = _parse_llm_response(response)
    except Exception as e:
        error_msg = str(e)

    elapsed_ms = (time.time() - start_time) * 1000

    return EvalResult(
        sample_id=sample.id,
        query=sample.query,
        expected_answer=sample.expected_answer,
        predicted_answer=predicted,
        dataset=sample.dataset,
        category=sample.category,
        execution_success=error_msg is None and predicted is not None,
        drift_detected=False,
        nli_triggered=False,
        latency_ms=round(elapsed_ms, 2),
        audit_trace={"baseline": "vanilla", "raw_response": predicted},
        error=error_msg,
    )


# ---------------------------------------------------------------------------
# Chain-of-Thought Baseline
# ---------------------------------------------------------------------------

def run_cot_baseline(sample: EvalSample) -> EvalResult:
    """
    Run a sample through LLM with Chain-of-Thought prompting.

    Prompt: "Let's think step by step" prefix.
    No deterministic verification — just reasoning.
    """
    client = _get_gemini_client()

    prompt = (
        f"Answer the following question by thinking step by step.\n\n"
        f"Question: {sample.query}\n\n"
        f"Let's think step by step:\n"
    )

    start_time = time.time()
    error_msg = None
    predicted = None
    raw_response = None

    try:
        raw_response = client.generate(prompt)
        predicted = _parse_llm_response(raw_response)
    except Exception as e:
        error_msg = str(e)

    elapsed_ms = (time.time() - start_time) * 1000

    return EvalResult(
        sample_id=sample.id,
        query=sample.query,
        expected_answer=sample.expected_answer,
        predicted_answer=predicted,
        dataset=sample.dataset,
        category=sample.category,
        execution_success=error_msg is None and predicted is not None,
        drift_detected=False,
        nli_triggered=False,
        latency_ms=round(elapsed_ms, 2),
        audit_trace={
            "baseline": "cot",
            "raw_response": raw_response,
            "extracted_answer": predicted,
        },
        error=error_msg,
    )


# ---------------------------------------------------------------------------
# Response Parsing
# ---------------------------------------------------------------------------

def _parse_llm_response(response: str) -> Any:
    """
    Extract the final answer from an LLM response.

    Tries:
        1. #### marker
        2. "The answer is X" pattern
        3. Last number in response
        4. Raw text as fallback
    """
    if not response:
        return None

    # Try #### format
    match = re.search(r"####\s*([-+]?\d*\.?\d+)", response)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass

    # Try "the answer is X" pattern
    match = re.search(
        r"(?:the answer is|the result is|= |equals)\s*([-+]?\d[\d,]*\.?\d*)",
        response,
        re.IGNORECASE,
    )
    if match:
        try:
            return float(match.group(1).replace(",", ""))
        except ValueError:
            pass

    # Try last number in response
    numbers = re.findall(r"[-+]?\d[\d,]*\.?\d*", response)
    if numbers:
        try:
            return float(numbers[-1].replace(",", ""))
        except ValueError:
            pass

    # Fallback: return raw text (trimmed)
    return response.strip()[:200]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_baseline_runner(name: str) -> Callable[[EvalSample], EvalResult]:
    """
    Get a baseline runner function by name.

    Args:
        name: 'vanilla' or 'cot'

    Returns:
        Runner function with signature (EvalSample) -> EvalResult
    """
    runners = {
        "vanilla": run_vanilla_baseline,
        "cot": run_cot_baseline,
    }

    runner = runners.get(name)
    if not runner:
        raise ValueError(f"Unknown baseline: {name}. Available: {list(runners.keys())}")

    return runner
