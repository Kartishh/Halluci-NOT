"""
core/groq_llm.py

HalluciNOT (LGP) — Groq LLM Wrapper
=====================================

Agentic reasoning layer using Groq API (llama3-70b-8192).
Uses OpenAI-compatible SDK with Groq's base URL.

Two capabilities:
    1. generate_reasoning()  — Chain-of-Thought step-by-step reasoning
    2. decompose_to_predicates() — Convert reasoning to JSON predicates
"""

from __future__ import annotations

import json
import os
import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from openai import OpenAI

logger = logging.getLogger("LGP.GroqLLM")
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "llama-3.1-8b-instant"

SUPPORTED_PREDICATES = [
    "Assign", "Add", "Subtract", "Multiply", "Divide",
    "GreaterThan", "LessThan", "Equals", "Conditional",
]


# ---------------------------------------------------------------------------
# Response Schema
# ---------------------------------------------------------------------------

@dataclass
class ReasoningResult:
    """Result from CoT reasoning generation."""
    reasoning: str        # Step-by-step reasoning text
    final_answer: float   # Extracted numeric answer
    raw_response: str     # Full LLM output


@dataclass
class DecompositionResult:
    """Result from LLM-based predicate decomposition."""
    predicates: List[Dict[str, Any]]  # List of predicate dicts
    raw_json: str                      # Raw JSON from LLM
    valid: bool                        # Schema validation passed


# ---------------------------------------------------------------------------
# Groq LLM Client
# ---------------------------------------------------------------------------

class GroqLLM:
    """
    Groq API wrapper for agentic reasoning.

    Uses llama3-70b-8192 via OpenAI-compatible endpoint.
    """

    def __init__(self, api_key: str = "", model: str = GROQ_MODEL):
        self.api_keys = []
        if api_key:
            self.api_keys.append(api_key)
        if GROQ_API_KEY:
            self.api_keys.append(GROQ_API_KEY)
        
        # Add additional keys from environment if present
        for i in range(2, 6):
            key = os.environ.get(f"GROQ_API_KEY_{i}")
            if key:
                self.api_keys.append(key)
            
        self.model = model
        self._current_key_idx = 0
        self._client = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            if not self.api_keys:
                raise ValueError(
                    "No GROQ_API_KEYs available. "
                    "Set via environment variables or pass to constructor."
                )
            current_key = self.api_keys[self._current_key_idx]
            self._client = OpenAI(
                api_key=current_key,
                base_url=GROQ_BASE_URL,
            )
        return self._client
        
    def _rotate_key(self):
        if len(self.api_keys) > 1:
            self._current_key_idx = (self._current_key_idx + 1) % len(self.api_keys)
            self._client = None
            logger.info(f"Rotated GROQ API key to index {self._current_key_idx}")

    def _call_with_fallback(self, **kwargs):
        import openai
        import time
        max_retries = len(self.api_keys)
        for attempt in range(max_retries):
            try:
                return self.client.chat.completions.create(**kwargs)
            except openai.RateLimitError as e:
                logger.warning(f"RateLimitError encountered on key index {self._current_key_idx}: {e}")
                self._rotate_key()
                time.sleep(1)
            except openai.APIError as e:
                logger.warning(f"APIError encountered on key index {self._current_key_idx}: {e}")
                self._rotate_key()
                time.sleep(1)
        # If we exhausted all keys, try one last time which will naturally bubble up the exception
        return self.client.chat.completions.create(**kwargs)

    # ------------------------------------------------------------------
    # 1. Chain-of-Thought Reasoning
    # ------------------------------------------------------------------

    def generate_reasoning(
        self,
        query: str,
        reflexion_feedback: Optional[str] = None,
    ) -> ReasoningResult:
        """
        Generate step-by-step reasoning for a math/logic query.

        Args:
            query: The problem to solve
            reflexion_feedback: Optional feedback from SSCE drift detection
                                to guide corrected reasoning

        Returns:
            ReasoningResult with reasoning steps and final answer
        """
        system_prompt = (
            "You are a precise mathematical reasoning assistant. "
            "Solve the problem step-by-step. "
            "Show ALL intermediate calculations clearly. "
            "Assign each intermediate value to a named variable. "
            "At the end, state your final answer as: "
            "FINAL ANSWER: <number>"
        )

        user_prompt = f"Problem: {query}"

        if reflexion_feedback:
            user_prompt += (
                f"\n\n--- IMPORTANT CORRECTION ---\n"
                f"{reflexion_feedback}\n"
                f"--- END CORRECTION ---\n\n"
                f"Rewrite your reasoning while maintaining logical consistency. "
                f"Do NOT repeat the same error."
            )

        response = self._call_with_fallback(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=1024,
        )

        raw = response.choices[0].message.content.strip()
        answer = self._extract_answer(raw)

        return ReasoningResult(
            reasoning=raw,
            final_answer=answer,
            raw_response=raw,
        )

    # ------------------------------------------------------------------
    # 2. LLM-Based Predicate Decomposition
    # ------------------------------------------------------------------

    def decompose_to_predicates(
        self,
        reasoning: str,
    ) -> DecompositionResult:
        """
        Convert step-by-step reasoning into structured JSON predicates.

        Returns:
            DecompositionResult with validated predicate list
        """
        system_prompt = (
            "You are a symbolic logic extractor. "
            "Convert the given mathematical reasoning into a JSON array "
            "of predicate objects. "
            "Each predicate must have exactly two keys: "
            '"predicate" (string) and "arguments" (array of strings). '
            "\n\nSupported predicates:\n"
            '- Assign(value, variable): {"predicate": "Assign", "arguments": ["5", "x"]}\n'
            '- Add(a, b, result): {"predicate": "Add", "arguments": ["x", "3", "y"]}\n'
            '- Subtract(a, b, result): {"predicate": "Subtract", "arguments": ["x", "3", "y"]}\n'
            '- Multiply(a, b, result): {"predicate": "Multiply", "arguments": ["x", "3", "y"]}\n'
            '- Divide(a, b, result): {"predicate": "Divide", "arguments": ["x", "3", "y"]}\n'
            "\n\nRules:\n"
            "1. Output ONLY a JSON array. No other text.\n"
            "2. Use variable names from the reasoning.\n"
            "3. Numeric literals should be strings: \"5\" not 5.\n"
            "4. Every calculation step must be a separate predicate.\n"
            "5. The final result must be the last predicate.\n"
            "\nExtract ALL variable assignments in order as they appear.\n"
            "If a variable is assigned multiple times, include ALL assignments.\n"
            "Do NOT overwrite or ignore earlier values.\n"
            "Preserve the full reasoning trace.\n"
        )

        response = self._call_with_fallback(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Reasoning:\n{reasoning}"},
            ],
            temperature=0.0,
            max_tokens=512,
        )

        raw_json = response.choices[0].message.content.strip()
        predicates, valid = self._parse_predicates(raw_json)

        decomp = DecompositionResult(
            predicates=predicates,
            raw_json=raw_json,
            valid=valid,
        )
        print("\nDEBUG PREDICATES:", decomp.predicates)
        return decomp

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _extract_answer(self, text: str) -> float:
        """Extract numeric answer from 'FINAL ANSWER: <number>'."""
        # Try explicit FINAL ANSWER pattern
        m = re.search(r'FINAL\s+ANSWER\s*:\s*([-+]?\d*\.?\d+)', text, re.I)
        if m:
            return float(m.group(1))

        # Fallback: last number in text
        nums = re.findall(r'[-+]?\d*\.?\d+', text)
        if nums:
            return float(nums[-1])

        return float('nan')

    def _parse_predicates(self, raw: str) -> Tuple[List[Dict], bool]:
        """Parse and validate JSON predicates from LLM output."""
        # Extract JSON array from response (may have markdown fence)
        json_str = raw
        m = re.search(r'\[.*\]', raw, re.DOTALL)
        if m:
            json_str = m.group(0)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON from decomposer: {raw[:200]}")
            return [], False

        if not isinstance(data, list):
            return [], False

        # Validate each predicate
        validated = []
        for item in data:
            if not isinstance(item, dict):
                continue
            pred = item.get("predicate", "")
            args = item.get("arguments", [])

            if pred not in SUPPORTED_PREDICATES:
                logger.warning(f"Unsupported predicate: {pred}")
                continue
            if not isinstance(args, list) or len(args) < 2:
                continue

            # Ensure all arguments are strings
            args = [str(a) for a in args]
            validated.append({"predicate": pred, "arguments": args})

        return validated, len(validated) > 0


# ---------------------------------------------------------------------------
# Convenience Accessor
# ---------------------------------------------------------------------------

def get_groq_llm(api_key: str = "") -> GroqLLM:
    return GroqLLM(api_key=api_key)
