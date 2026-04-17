"""
core/nvidia_llm.py

HalluciNOT (LGP) — NVIDIA LLM Wrapper
=======================================
Agentic reasoning layer using NVIDIA API (OpenAI-compatible SDK).

Model: openai/gpt-oss-120b
Rate Limit: 40 requests/minute (1.5s interval, enforced globally)
"""

from __future__ import annotations

import json
import os
import re
import time
import threading
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

logger = logging.getLogger("LGP.NvidiaLLM")
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Data-classes (unchanged interface — backward compatible)
# ---------------------------------------------------------------------------

SUPPORTED_PREDICATES = [
    "Assign", "Add", "Subtract", "Multiply", "Divide",
    "GreaterThan", "LessThan", "Equals", "Conditional",
]


@dataclass
class ReasoningResult:
    reasoning: str
    final_answer: float
    raw_response: str


@dataclass
class DecompositionResult:
    predicates: List[Dict[str, Any]]
    raw_json: str
    valid: bool


# ---------------------------------------------------------------------------
# Safe Response Extraction
# ---------------------------------------------------------------------------

def _safe_extract_text(response):
    """Safely extract text from an OpenAI-style response, never crashes."""
    try:
        if response is None:
            return ""
        if not hasattr(response, "choices") or not response.choices:
            return ""
        msg = response.choices[0].message
        if not hasattr(msg, "content") or msg.content is None:
            return ""
        return str(msg.content).strip()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# NVIDIA LLM Client
# ---------------------------------------------------------------------------

class NvidiaLLM:
    """
    NVIDIA LLM wrapper with:
      - Global rate limiting (40 RPM → 1.5s minimum interval)
      - Thread-safe rate enforcement
      - Retry with exponential backoff (3 attempts)
      - Full backward compatibility with GeminiLLM interface
    """

    # Class-level rate limiter (shared across ALL instances)
    _last_request_time: float = 0.0
    _lock = threading.Lock()

    def __init__(self, model: str = ""):
        api_key = os.getenv("NVIDIA_API_KEY", "")
        if not api_key:
            raise ValueError("NVIDIA_API_KEY missing from environment variables.")

        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key,
        )
        self.model = model or "openai/gpt-oss-120b"

        # Rate limit: 40 RPM → 1 request every 1.5 seconds
        self.min_interval = 1.5

        print(f"[NVIDIA] Initialised with model: {self.model}")

    # ------------------------------------------------------------------
    # Rate Limit Enforcement (thread-safe, global)
    # ------------------------------------------------------------------

    def _enforce_rate_limit(self):
        with NvidiaLLM._lock:
            now = time.time()
            elapsed = now - NvidiaLLM._last_request_time
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            NvidiaLLM._last_request_time = time.time()

    # ------------------------------------------------------------------
    # Core API Call (non-streaming, with retry)
    # ------------------------------------------------------------------

    def _call(self, messages: list, temperature: float = 0.2,
              max_tokens: int = 1024):
        """
        Make a rate-limited, retried call to NVIDIA API.
        Returns the raw OpenAI-style response object.
        """
        for attempt in range(3):
            try:
                self._enforce_rate_limit()

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                logger.info("[NVIDIA] success model=%s", self.model)
                print(f"[NVIDIA] success model={self.model}")
                return response

            except Exception as e:
                wait = 2 ** (attempt + 1)  # 2, 4, 8
                logger.warning(
                    "[NVIDIA] attempt=%d failed (%s) — retrying in %ds",
                    attempt + 1, e, wait,
                )
                print(
                    f"[NVIDIA] attempt {attempt + 1} failed: {e}"
                    f" — waiting {wait}s"
                )
                time.sleep(wait)

        raise RuntimeError("NVIDIA LLM failed after 3 retries")

    # ------------------------------------------------------------------
    # OpenAI-style interface (used by reflexion.py, decomposer.py,
    # factored_verifier.py via llm._call_with_fallback)
    # ------------------------------------------------------------------

    def _call_with_fallback(self, model=None, messages=None,
                            temperature=0.0, max_tokens=1024):
        """
        Backward-compatible interface matching the old Groq/Gemini pattern.

        Callers expect:
            response = llm._call_with_fallback(model=..., messages=..., ...)
            text = response.choices[0].message.content.strip()

        Returns the raw OpenAI response object.
        """
        return self._call(
            messages=messages or [],
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # ------------------------------------------------------------------
    # String-return interface (used by baselines.py via
    # client._call_gemini_with_fallback)
    # ------------------------------------------------------------------

    def _call_gemini_with_fallback(self, prompt: str,
                                    system_prompt: str = "",
                                    temperature: float = 0.2) -> str:
        """
        Backward-compatible interface for evaluation baselines.

        Returns plain text string (not response object).
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self._call(
            messages=messages,
            temperature=temperature,
            max_tokens=1024,
        )
        return _safe_extract_text(response)

    # ------------------------------------------------------------------
    # Public API — generate_reasoning (same signature as GeminiLLM)
    # ------------------------------------------------------------------

    def generate_answer_only(self, query: str) -> float:
        """Generate fast heuristic answer - intentionally approximate to trigger drift detection."""
        system_prompt = """You are solving a math problem quickly using intuition.
Do NOT compute carefully.
Give your best quick estimate.
Output ONLY the number, nothing else.
Do not show work or explanation."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Problem: {query}"},
        ]

        response = self._call(messages=messages, temperature=0.9)
        raw = _safe_extract_text(response)
        
        # Extract numeric answer from response
        nums = re.findall(r'[-+]?\d+\.?\d*', raw.replace(',', ''))
        if nums:
            return float(nums[0])
        return float('nan')

    def generate_reasoning(
        self,
        query: str,
        reflexion_feedback: Optional[str] = None,
    ) -> ReasoningResult:
        system_prompt = """You are generating reasoning for a symbolic execution system.

STRICT RULES:

1. Output ONLY equations.
2. Each line must be: variable = expression
3. No explanations
4. No text
5. No tables
6. No LaTeX
7. No units (miles, dollars, hours)

VALID EXAMPLE:

r = 1616598
o = 1.2 * r
eregular = 40 * r
overtimehours = 45 - 40
eovertime = overtimehours * o
etotal = eregular + eovertime

RULES:

* Every variable must be defined before use
* Use only numbers and variables
* Allowed operators: + - * /
* Keep variable names simple

IMPORTANT: At the END of your output, add a final line:
result = <final computed value>

This line must contain the numeric answer, not a variable reference.

Return ONLY equations."""

        user_prompt = f"Problem: {query}"
        if reflexion_feedback:
            user_prompt += (
                f"\n\n--- IMPORTANT CORRECTION ---\n"
                f"{reflexion_feedback}\n"
                f"--- END CORRECTION ---\n\n"
                f"Rewrite your reasoning while maintaining logical consistency. "
                f"Do NOT repeat the same error."
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = self._call(messages=messages, temperature=0.2)
        raw = _safe_extract_text(response)
        if not raw:
            logger.warning("Empty LLM response — returning fallback")
            return ReasoningResult(reasoning="", final_answer=float('nan'), raw_response="")
        answer = self._extract_answer(raw)
        return ReasoningResult(reasoning=raw, final_answer=answer, raw_response=raw)

    # ------------------------------------------------------------------
    # Public API — decompose_to_predicates (same signature as GeminiLLM)
    # ------------------------------------------------------------------

    def decompose_to_predicates(self, reasoning: str) -> DecompositionResult:
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
            '3. Numeric literals should be strings: "5" not 5.\n'
            "4. Every calculation step must be a separate predicate.\n"
            "5. The final result must be the last predicate.\n"
            "\nExtract ALL variable assignments in order as they appear.\n"
            "If a variable is assigned multiple times, include ALL assignments.\n"
            "Do NOT overwrite or ignore earlier values.\n"
            "Preserve the full reasoning trace.\n"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Reasoning:\n{reasoning}"},
        ]

        response = self._call(messages=messages, temperature=0.0)
        raw_json = _safe_extract_text(response)
        predicates, valid = self._parse_predicates(raw_json)
        return DecompositionResult(predicates=predicates, raw_json=raw_json, valid=valid)

    # ------------------------------------------------------------------
    # Private parsing helpers (identical to GeminiLLM)
    # ------------------------------------------------------------------

    def _extract_answer(self, text: str) -> float:
        lines = [l.strip() for l in text.strip().split('\n') if l.strip() and '=' in l]
        
        for line in reversed(lines):
            rhs = line.split('=', 1)[1].strip()
            
            m = re.search(r'\bresult\s*=\s*([-+]?\d+\.?\d*)', line, re.I)
            if m:
                return float(m.group(1))
            m = re.search(r'\banswer\s*=\s*([-+]?\d+\.?\d*)', line, re.I)
            if m:
                return float(m.group(1))
            m = re.search(r'\boutput\s*=\s*([-+]?\d+\.?\d*)', line, re.I)
            if m:
                return float(m.group(1))
            
            nums = re.findall(r'(?<![a-zA-Z])[-+]?\d+\.?\d*(?![a-zA-Z])', rhs)
            if nums:
                return float(nums[-1])
        
        return float('nan')

    def _parse_predicates(self, raw: str) -> Tuple[List[Dict], bool]:
        json_str = raw
        m = re.search(r'\[.*\]', raw, re.DOTALL)
        if m:
            json_str = m.group(0)
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning("Invalid JSON from decomposer: %s", raw[:200])
            return [], False
        if not isinstance(data, list):
            return [], False

        validated = []
        for item in data:
            if not isinstance(item, dict):
                continue
            pred = item.get("predicate", "")
            args = item.get("arguments", [])
            if pred not in SUPPORTED_PREDICATES:
                continue
            if not isinstance(args, list) or len(args) < 2:
                continue
            args = [str(a) for a in args]
            validated.append({"predicate": pred, "arguments": args})
        return validated, len(validated) > 0


# ---------------------------------------------------------------------------
# Factory (backward compatible with get_gemini_llm)
# ---------------------------------------------------------------------------

def get_nvidia_llm(model: str = "") -> NvidiaLLM:
    return NvidiaLLM(model=model)
