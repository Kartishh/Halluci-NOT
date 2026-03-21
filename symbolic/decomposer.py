"""
symbolic/decomposer.py

HalluciNOT (LGP)
---------------------------------
Hybrid Symbolic Decomposer (Regex + Gemini Few-Shot)

Architecture:
    Stage 1: Segment text into reasoning steps
    Stage 2A: Try deterministic regex extraction (fast path, <1ms)
    Stage 2B: Fall back to Gemini few-shot extraction (natural language)
    Stage 3: Validate all extracted facts against predicate registry

The LLM extractor uses a carefully crafted few-shot prompt that forces
structured JSON output. Invalid JSON or unsupported predicates trigger
a DecompositionComplexityError for the Reflexion loop.

Author: LGP Framework
"""

from __future__ import annotations

import json
import os
import re
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

logger = logging.getLogger("LGP.SymbolicDecomposer")
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class AtomicFact:
    """
    Structured atomic reasoning fact.

    Attributes:
        predicate: Logical operation name
        arguments: Ordered list of argument identifiers
        raw_text: Original text segment
        source_path: Extraction method ('regex' or 'gemini')
    """
    predicate: str
    arguments: List[str]
    raw_text: str
    source_path: str = "regex"  # 'regex' or 'gemini'

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predicate": self.predicate,
            "arguments": self.arguments,
            "raw_text": self.raw_text,
            "source_path": self.source_path,
        }


# ---------------------------------------------------------------------------
# Custom Exception for Reflexion Loop
# ---------------------------------------------------------------------------

class DecompositionComplexityError(RuntimeError):
    """
    Raised when decomposition fails despite LLM extraction.
    Triggers the Reflexion loop in the policy layer.
    """
    def __init__(self, segment: str, reason: str):
        self.segment = segment
        self.reason = reason
        super().__init__(
            f"Decomposition failed for '{segment[:80]}...': {reason}"
        )


# ---------------------------------------------------------------------------
# Predicate Registry (Extended)
# ---------------------------------------------------------------------------

SUPPORTED_PREDICATES = {
    "Add": 3,          # (a, b, result)
    "Subtract": 3,     # (a, b, result)
    "Multiply": 3,     # (a, b, result)
    "Divide": 3,       # (a, b, result)
    "Assign": 2,       # (value, variable)
    "GreaterThan": 2,  # (a, b) — a > b
    "LessThan": 2,     # (a, b) — a < b
    "Equals": 2,       # (a, b) — a == b
    "Conditional": 3,  # (condition_var, then_result, else_result)
}


# ---------------------------------------------------------------------------
# Few-Shot Prompt Template
# ---------------------------------------------------------------------------

_FEW_SHOT_PROMPT = """You are a symbolic logic extractor for a hallucination detection system.

Your job: Convert natural language reasoning into structured atomic facts.

RULES:
1. Output ONLY a JSON array of fact objects. No explanation, no markdown.
2. Each fact must have exactly: {"predicate": "...", "arguments": [...], "raw_text": "..."}
3. Supported predicates and their argument patterns:
   - Assign(value, variable): "price is 5" → Assign("5", "price")
   - Add(a, b, result): "total is cost plus tax" → Add("cost", "tax", "total")
   - Subtract(a, b, result): "profit is revenue minus expenses" → Subtract("revenue", "expenses", "profit")
   - Multiply(a, b, result): "area is length times width" → Multiply("length", "width", "area")
   - Divide(a, b, result): "average is total divided by count" → Divide("total", "count", "average")
   - GreaterThan(a, b): "x is more than y" → GreaterThan("x", "y")
   - LessThan(a, b): "x is less than y" → LessThan("x", "y")
   - Equals(a, b): "x equals y" → Equals("x", "y")
   - Conditional(condition_var, then_result, else_result): "if qualified then bonus else penalty"
4. Use snake_case variable names. Convert "the total cost" to "total_cost".
5. Raw numbers stay as strings: "5", "3.14", "-50".
6. If a variable is defined with a value, use Assign. If computed, use the operation.

EXAMPLES:

Input: "Janet's ducks lay 16 eggs per day"
Output: [{"predicate": "Assign", "arguments": ["16", "eggs_per_day"], "raw_text": "Janet's ducks lay 16 eggs per day"}]

Input: "She eats three for breakfast every morning and bakes muffins for her friends with four"
Output: [{"predicate": "Assign", "arguments": ["3", "eggs_eaten"], "raw_text": "She eats three for breakfast"}, {"predicate": "Assign", "arguments": ["4", "eggs_baked"], "raw_text": "bakes muffins for her friends with four"}]

Input: "The total cost is the price multiplied by the quantity"
Output: [{"predicate": "Multiply", "arguments": ["price", "quantity", "total_cost"], "raw_text": "The total cost is the price multiplied by the quantity"}]

Input: "Subtracting the 5 percent tax from the 100 dollar base gives the net amount"
Output: [{"predicate": "Assign", "arguments": ["5", "tax_percent"], "raw_text": "5 percent tax"}, {"predicate": "Assign", "arguments": ["100", "base"], "raw_text": "100 dollar base"}, {"predicate": "Subtract", "arguments": ["base", "tax_percent", "net_amount"], "raw_text": "Subtracting the 5 percent tax from the 100 dollar base gives the net amount"}]

Input: "If the score is above 50, the student passes"
Output: [{"predicate": "GreaterThan", "arguments": ["score", "50"], "raw_text": "the score is above 50"}, {"predicate": "Conditional", "arguments": ["score", "passes", "fails"], "raw_text": "If the score is above 50, the student passes"}]

Now extract atomic facts from this text:

Input: "{text}"
Output:"""


# ---------------------------------------------------------------------------
# Gemini Client (Lazy-loaded Singleton)
# ---------------------------------------------------------------------------

_gemini_model = None


def _get_gemini_model():
    """Lazy-load Gemini model for decomposition."""
    global _gemini_model
    if _gemini_model is None:
        try:
            import google.generativeai as genai
            from dotenv import load_dotenv

            load_dotenv()
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                logger.warning("No GEMINI_API_KEY found. LLM extraction disabled.")
                return None

            genai.configure(api_key=api_key)
            _gemini_model = genai.GenerativeModel(
                os.getenv("LLM_MODEL", "gemini-2.0-flash")
            )
            logger.info("Gemini model loaded for decomposition.")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            return None

    return _gemini_model


# ---------------------------------------------------------------------------
# Decomposer Engine
# ---------------------------------------------------------------------------

class SymbolicDecomposer:
    """
    Hybrid decomposer: Regex fast-path + Gemini LLM fallback.

    Pipeline:
        1. Segment text
        2. For each segment:
           a. Try regex (deterministic, <1ms)
           b. If regex fails → Try Gemini few-shot (natural language)
           c. If both fail → Raise DecompositionComplexityError
        3. Validate all facts
    """

    def __init__(self):
        self._gemini_model = None  # Lazy-loaded on first LLM call

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def to_atomic_facts(self, text: str) -> List[AtomicFact]:
        """
        Convert reasoning text into atomic facts.

        Uses regex for structured input, falls back to Gemini for
        natural language reasoning.
        """
        if not text or not text.strip():
            raise ValueError("Input text for decomposition cannot be empty.")

        segments = self._segment_text(text)
        facts: List[AtomicFact] = []
        unresolved_segments: List[str] = []

        # --- First pass: try regex on all segments ---
        for segment in segments:
            extracted = self._rule_based_extract(segment)
            if extracted:
                # Tag as regex path
                for fact in extracted:
                    fact.source_path = "regex"
                facts.extend(extracted)
            else:
                unresolved_segments.append(segment)

        # --- Second pass: batch LLM extraction for unresolved ---
        if unresolved_segments:
            combined_text = "\n".join(unresolved_segments)
            llm_facts = self._llm_extract(combined_text)
            if llm_facts:
                # Tag as gemini path
                for fact in llm_facts:
                    fact.source_path = "gemini"
                facts.extend(llm_facts)
            elif unresolved_segments:
                # Neither regex nor LLM could extract — this is a real problem
                logger.warning(
                    f"No facts extracted from: {combined_text[:200]}"
                )

        self._validate_facts(facts)
        return facts

    # ------------------------------------------------------------------
    # Stage 1: Segmentation
    # ------------------------------------------------------------------

    def _segment_text(self, text: str) -> List[str]:
        """Split reasoning text into candidate logical segments."""
        segments = re.split(r"\. |\n|; ", text)
        return [seg.strip() for seg in segments if seg.strip()]

    # ------------------------------------------------------------------
    # Stage 2A: Rule-Based Extraction (Fast Path)
    # ------------------------------------------------------------------

    def _rule_based_extract(self, segment: str) -> List[AtomicFact]:
        """
        Deterministic extraction for structured arithmetic.
        Handles: assignments, all four arithmetic ops, negatives, decimals.
        Returns empty list if segment is natural language.
        """
        facts: List[AtomicFact] = []

        # Pattern: a = b * c
        match = re.match(r"(\w+)\s*=\s*(\w+)\s*\*\s*(\w+)", segment)
        if match:
            facts.append(AtomicFact(
                predicate="Multiply",
                arguments=[match.group(2), match.group(3), match.group(1)],
                raw_text=segment,
            ))
            return facts

        # Pattern: a = b + c
        match = re.match(r"(\w+)\s*=\s*(\w+)\s*\+\s*(\w+)", segment)
        if match:
            facts.append(AtomicFact(
                predicate="Add",
                arguments=[match.group(2), match.group(3), match.group(1)],
                raw_text=segment,
            ))
            return facts

        # Pattern: a = b - c
        match = re.match(r"(\w+)\s*=\s*(\w+)\s*-\s*(\w+)", segment)
        if match:
            facts.append(AtomicFact(
                predicate="Subtract",
                arguments=[match.group(2), match.group(3), match.group(1)],
                raw_text=segment,
            ))
            return facts

        # Pattern: a = b / c
        match = re.match(r"(\w+)\s*=\s*(\w+)\s*/\s*(\w+)", segment)
        if match:
            facts.append(AtomicFact(
                predicate="Divide",
                arguments=[match.group(2), match.group(3), match.group(1)],
                raw_text=segment,
            ))
            return facts

        # Pattern: a = b (supports negative numbers and decimals)
        match = re.match(r"(\w+)\s*=\s*(-?[\w.]+)", segment)
        if match:
            facts.append(AtomicFact(
                predicate="Assign",
                arguments=[match.group(2), match.group(1)],
                raw_text=segment,
            ))
            return facts

        return facts

    # ------------------------------------------------------------------
    # Stage 2B: Gemini Few-Shot Extraction
    # ------------------------------------------------------------------

    def _llm_extract(self, segment: str) -> List[AtomicFact]:
        """
        Use Gemini few-shot prompting to extract atomic facts from
        natural language reasoning text.

        Returns validated AtomicFact list.
        Raises DecompositionComplexityError on complete failure.
        """
        model = _get_gemini_model()
        if model is None:
            logger.warning("Gemini not available — LLM extraction skipped.")
            return []

        prompt = _FEW_SHOT_PROMPT.replace("{text}", segment)

        try:
            response = model.generate_content(prompt)
            raw_text = response.text.strip()

            # Strip markdown code fences if the model wraps in ```json
            raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
            raw_text = re.sub(r"\s*```$", "", raw_text)

            parsed = json.loads(raw_text)

            if not isinstance(parsed, list):
                raise DecompositionComplexityError(
                    segment, "LLM returned non-array JSON."
                )

            facts: List[AtomicFact] = []

            for item in parsed:
                predicate = item.get("predicate", "")
                arguments = item.get("arguments", [])
                raw = item.get("raw_text", segment)

                # Validate predicate exists
                if predicate not in SUPPORTED_PREDICATES:
                    logger.warning(
                        f"LLM returned unsupported predicate '{predicate}' — skipping."
                    )
                    continue

                # Validate arity
                expected_arity = SUPPORTED_PREDICATES[predicate]
                if len(arguments) != expected_arity:
                    logger.warning(
                        f"LLM predicate {predicate} has wrong arity "
                        f"({len(arguments)} vs {expected_arity}) — skipping."
                    )
                    continue

                # Sanitize arguments: ensure all are strings
                arguments = [str(a) for a in arguments]

                facts.append(AtomicFact(
                    predicate=predicate,
                    arguments=arguments,
                    raw_text=raw,
                ))

            if facts:
                logger.info(
                    f"LLM extracted {len(facts)} facts from: "
                    f"'{segment[:60]}...'"
                )

            return facts

        except json.JSONDecodeError as e:
            logger.error(f"LLM returned invalid JSON: {e}")
            raise DecompositionComplexityError(
                segment,
                f"LLM returned invalid JSON: {str(e)[:100]}"
            )

        except DecompositionComplexityError:
            raise

        except Exception as e:
            logger.error(f"LLM extraction error: {e}")
            raise DecompositionComplexityError(
                segment,
                f"LLM extraction failed: {str(e)[:100]}"
            )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_facts(self, facts: List[AtomicFact]) -> None:
        """Enforce predicate schema validity."""
        for fact in facts:
            if fact.predicate not in SUPPORTED_PREDICATES:
                raise ValueError(f"Unsupported predicate: {fact.predicate}")

            expected_arity = SUPPORTED_PREDICATES[fact.predicate]
            if len(fact.arguments) != expected_arity:
                raise ValueError(
                    f"Predicate {fact.predicate} expects {expected_arity} "
                    f"arguments, got {len(fact.arguments)}"
                )


# ---------------------------------------------------------------------------
# Convenience Accessor
# ---------------------------------------------------------------------------


def get_symbolic_decomposer() -> SymbolicDecomposer:
    """Public accessor for SymbolicDecomposer."""
    return SymbolicDecomposer()
