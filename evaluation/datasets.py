"""
evaluation/datasets.py

HalluciNOT (LGP)
---------------------------------
Unified Dataset Loaders for Evaluation

Supported datasets:
    1. HaluEval — Hallucination evaluation benchmark (Li et al., 2023)
    2. GSM-Hard — Grade-school math with harder numerics (reasoning-machines)
    3. PopQA  — Long-tail factual QA (Mallen et al., 2023)

All loaders return a uniform EvalSample schema for pipeline compatibility.

Author: LGP Framework
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from datasets import load_dataset

logger = logging.getLogger("LGP.Eval.Datasets")
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Uniform Sample Schema
# ---------------------------------------------------------------------------

@dataclass
class EvalSample:
    """
    Uniform evaluation sample across all datasets.

    Attributes:
        id: Unique sample identifier
        query: Input to feed into LGP pipeline
        expected_answer: Ground truth for accuracy evaluation
        dataset: Source dataset name
        category: Logic category (arithmetic, factual, logical, etc.)
        metadata: Dataset-specific extra fields
    """
    id: str
    query: str
    expected_answer: Any
    dataset: str
    category: str = "general"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# HaluEval Loader
# ---------------------------------------------------------------------------

def load_halueval(
    split: str = "qa",
    limit: Optional[int] = None,
) -> List[EvalSample]:
    """
    Load HaluEval benchmark dataset.

    Args:
        split: Which HaluEval task — 'qa', 'dialogue', or 'summarization'
        limit: Max samples to load (None = all)

    Returns:
        List of EvalSample with hallucination labels
    """
    logger.info(f"Loading HaluEval (split={split}, limit={limit})")

    config_map = {
        "qa": "qa_samples",
        "dialogue": "dialogue_samples",
        "summarization": "summarization_samples",
    }

    config = config_map.get(split)
    if not config:
        raise ValueError(f"Invalid HaluEval split: {split}. Use: {list(config_map.keys())}")

    ds = load_dataset("pminervini/HaluEval", config, split="data")

    samples: List[EvalSample] = []

    for idx, row in enumerate(ds):
        if limit and idx >= limit:
            break

        if split == "qa":
            query = row.get("question", "")
            expected = row.get("right_answer", "")
            hallucinated = row.get("hallucinated_answer", "")
            knowledge = row.get("knowledge", "")
            category = "factual_qa"
        elif split == "dialogue":
            query = row.get("dialogue_history", "")
            expected = row.get("right_response", "")
            hallucinated = row.get("hallucinated_response", "")
            knowledge = row.get("knowledge", "")
            category = "dialogue"
        else:  # summarization
            query = row.get("document", "")
            expected = row.get("right_summary", "")
            hallucinated = row.get("hallucinated_summary", "")
            knowledge = ""
            category = "summarization"

        samples.append(EvalSample(
            id=f"halueval_{split}_{idx}",
            query=query,
            expected_answer=expected,
            dataset="halueval",
            category=category,
            metadata={
                "hallucinated_answer": hallucinated,
                "knowledge": knowledge,
                "split": split,
            },
        ))

    logger.info(f"Loaded {len(samples)} HaluEval samples.")
    return samples


# ---------------------------------------------------------------------------
# GSM-Hard Loader
# ---------------------------------------------------------------------------

def _extract_numeric_answer(answer_text: str) -> Optional[float]:
    """
    Extract the final numeric answer from GSM-style answer text.
    Handles formats like '#### 42' or plain numbers.
    """
    if not answer_text:
        return None

    # Try #### marker first
    match = re.search(r"####\s*([-+]?\d*\.?\d+)", str(answer_text))
    if match:
        return float(match.group(1))

    # Try last number in text
    numbers = re.findall(r"[-+]?\d*\.?\d+", str(answer_text))
    if numbers:
        return float(numbers[-1])

    return None


def load_gsm_hard(
    limit: Optional[int] = None,
) -> List[EvalSample]:
    """
    Load GSM-Hard benchmark for multi-step arithmetic reasoning.

    This is the primary benchmark for symbolic drift detection
    since every question involves multi-step calculations.

    Returns:
        List of EvalSample with numeric ground truth answers
    """
    logger.info(f"Loading GSM-Hard (limit={limit})")

    ds = load_dataset("reasoning-machines/gsm-hard", split="train")

    samples: List[EvalSample] = []

    for idx, row in enumerate(ds):
        if limit and idx >= limit:
            break

        query = row.get("input", "")
        raw_target = row.get("target", "")

        numeric_answer = _extract_numeric_answer(str(raw_target))

        # Determine logic category based on content
        category = _classify_logic_type(query)

        samples.append(EvalSample(
            id=f"gsm_hard_{idx}",
            query=query,
            expected_answer=numeric_answer,
            dataset="gsm_hard",
            category=category,
            metadata={
                "raw_target": raw_target,
                "code": row.get("code", ""),
            },
        ))

    logger.info(f"Loaded {len(samples)} GSM-Hard samples.")
    return samples


# ---------------------------------------------------------------------------
# PopQA Loader
# ---------------------------------------------------------------------------

def load_popqa(
    limit: Optional[int] = None,
) -> List[EvalSample]:
    """
    Load PopQA long-tail factual QA dataset.

    Tests NLI retrieval triggers and factual grounding when
    knowledge is rare or obscure.

    Returns:
        List of EvalSample with factual answers
    """
    logger.info(f"Loading PopQA (limit={limit})")

    ds = load_dataset("akariasai/PopQA", split="test")

    samples: List[EvalSample] = []

    for idx, row in enumerate(ds):
        if limit and idx >= limit:
            break

        query = row.get("question", "")
        expected = row.get("possible_answers", "")

        # possible_answers is often a JSON-serialized list
        if isinstance(expected, str):
            try:
                expected = json.loads(expected)
            except (json.JSONDecodeError, TypeError):
                expected = [expected]

        samples.append(EvalSample(
            id=f"popqa_{idx}",
            query=query,
            expected_answer=expected,
            dataset="popqa",
            category="factual_qa",
            metadata={
                "subj": row.get("subj", ""),
                "prop": row.get("prop", ""),
                "obj": row.get("obj", ""),
                "s_pop": row.get("s_pop", 0),
            },
        ))

    logger.info(f"Loaded {len(samples)} PopQA samples.")
    return samples


# ---------------------------------------------------------------------------
# Logic Type Classifier
# ---------------------------------------------------------------------------

def _classify_logic_type(query: str) -> str:
    """
    Classify the type of logical reasoning required by a query.

    Categories:
        - arithmetic: Pure math operations
        - comparison: Greater/less than, ranking
        - conditional: If-then logic
        - multi_step: Chained dependencies
        - causal: Cause-effect reasoning
        - temporal: Time-based reasoning
        - general: Unclassified
    """
    q_lower = query.lower()

    # Check for conditional logic
    if any(kw in q_lower for kw in ["if ", "suppose ", "assuming ", "given that", "when "]):
        return "conditional"

    # Check for comparison logic
    if any(kw in q_lower for kw in [
        "more than", "less than", "greater", "fewer", "most", "least",
        "compare", "difference between", "which is bigger", "which is more",
    ]):
        return "comparison"

    # Check for temporal reasoning
    if any(kw in q_lower for kw in [
        "before", "after", "during", "while", "then ", "next ",
        "first ", "last ", "earlier", "later", "yesterday", "tomorrow",
    ]):
        return "temporal"

    # Check for causal reasoning
    if any(kw in q_lower for kw in [
        "because", "therefore", "cause", "result in", "leads to",
        "consequence", "effect of", "due to",
    ]):
        return "causal"

    # Check for multi-step arithmetic indicators
    step_indicators = len(re.findall(
        r"(then|next|after that|also|additionally|finally|total|altogether)",
        q_lower,
    ))
    if step_indicators >= 2:
        return "multi_step"

    # Check for basic arithmetic
    if any(kw in q_lower for kw in [
        "how many", "how much", "calculate", "total", "sum", "product",
        "average", "percent", "ratio", "cost", "price", "earn",
    ]):
        return "arithmetic"

    return "general"


# ---------------------------------------------------------------------------
# Convenience: Load Any Dataset
# ---------------------------------------------------------------------------

def load_dataset_by_name(
    name: str,
    limit: Optional[int] = None,
    **kwargs,
) -> List[EvalSample]:
    """
    Load a dataset by name.

    Args:
        name: One of 'halueval', 'gsm_hard', 'popqa'
        limit: Max samples
        **kwargs: Dataset-specific args (e.g., split for HaluEval)

    Returns:
        List of EvalSample
    """
    loaders = {
        "halueval": load_halueval,
        "gsm_hard": load_gsm_hard,
        "popqa": load_popqa,
    }

    loader = loaders.get(name)
    if not loader:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(loaders.keys())}")

    return loader(limit=limit, **kwargs)
