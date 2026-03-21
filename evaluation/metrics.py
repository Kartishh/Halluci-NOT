"""
evaluation/metrics.py

HalluciNOT (LGP)
---------------------------------
Evaluation Metrics Computation

Implements research-grade metrics for measuring LGP's effectiveness:

    1. Hallucination Reduction Rate
    2. Drift Detection Accuracy
    3. Execution Success Rate
    4. Answer Accuracy (exact match + numeric tolerance)
    5. FFactScore (% atomic facts verified via PoT)
    6. Logic Category Breakdown

Author: LGP Framework
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

from evaluation.runner import EvalResult

logger = logging.getLogger("LGP.Eval.Metrics")
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Answer Comparison Utilities
# ---------------------------------------------------------------------------

def _extract_number(value: Any) -> Optional[float]:
    """Extract a numeric value from various formats."""
    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, dict):
        # Try to extract from dict values (PoT output)
        nums = [v for v in value.values() if isinstance(v, (int, float))]
        if nums:
            return float(nums[-1])

    if isinstance(value, str):
        # Try #### format
        match = re.search(r"####\s*([-+]?\d*\.?\d+)", value)
        if match:
            return float(match.group(1))
        # Try last number
        numbers = re.findall(r"[-+]?\d*\.?\d+", value)
        if numbers:
            return float(numbers[-1])

    return None


def _normalize_text(text: Any) -> str:
    """Normalize text for comparison."""
    if text is None:
        return ""
    return str(text).strip().lower()


def _numeric_match(predicted: Any, expected: Any, tolerance: float = 0.01) -> bool:
    """
    Check if predicted and expected answers match numerically.
    Uses relative tolerance for large numbers, absolute for small.
    """
    pred_num = _extract_number(predicted)
    exp_num = _extract_number(expected)

    if pred_num is None or exp_num is None:
        return False

    if exp_num == 0:
        return abs(pred_num) < tolerance

    relative_error = abs(pred_num - exp_num) / abs(exp_num)
    return relative_error < tolerance


def _exact_match(predicted: Any, expected: Any) -> bool:
    """
    Check exact match between predicted and expected.
    Handles lists (PopQA possible_answers).
    """
    pred_text = _normalize_text(predicted)

    if isinstance(expected, list):
        return any(
            _normalize_text(ans) in pred_text or pred_text in _normalize_text(ans)
            for ans in expected
        )

    return _normalize_text(expected) == pred_text


def check_answer_correct(predicted: Any, expected: Any, dataset: str) -> bool:
    """
    Unified answer correctness check.

    For numeric datasets (gsm_hard): uses numeric tolerance
    For text datasets (halueval, popqa): uses exact match / containment
    """
    if predicted is None:
        return False

    # Try numeric match first for all datasets
    if _numeric_match(predicted, expected):
        return True

    # Fall back to text matching
    return _exact_match(predicted, expected)


# ---------------------------------------------------------------------------
# Core Metrics
# ---------------------------------------------------------------------------

def compute_metrics(results: List[EvalResult]) -> Dict[str, Any]:
    """
    Compute comprehensive evaluation metrics from results.

    Returns:
        Dictionary with overall and per-category metrics
    """
    if not results:
        return {"error": "No results to compute metrics from."}

    total = len(results)

    # --- Overall Metrics ---
    exec_success = sum(1 for r in results if r.execution_success)
    drift_detected = sum(1 for r in results if r.drift_detected)
    nli_triggered = sum(1 for r in results if r.nli_triggered)
    errors = sum(1 for r in results if r.error is not None)

    # --- Answer Accuracy ---
    correct = 0
    evaluated = 0
    for r in results:
        if r.expected_answer is not None:
            evaluated += 1
            if check_answer_correct(r.predicted_answer, r.expected_answer, r.dataset):
                correct += 1

    # --- Latency ---
    latencies = [r.latency_ms for r in results if r.latency_ms > 0]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 1 else avg_latency

    # --- FFactScore ---
    # Percentage of samples that passed PoT verification successfully
    pot_verified = sum(
        1 for r in results
        if r.execution_success and not r.drift_detected
    )
    ffact_score = pot_verified / total if total > 0 else 0

    # --- Per-Category Breakdown ---
    category_metrics = _compute_category_breakdown(results)

    # --- Per-Dataset Breakdown ---
    dataset_metrics = _compute_dataset_breakdown(results)

    overall = {
        "total_samples": total,
        "execution_success_rate": round(exec_success / total, 4),
        "drift_detection_rate": round(drift_detected / total, 4),
        "nli_trigger_rate": round(nli_triggered / total, 4),
        "error_rate": round(errors / total, 4),
        "answer_accuracy": round(correct / evaluated, 4) if evaluated > 0 else None,
        "ffact_score": round(ffact_score, 4),
        "avg_latency_ms": round(avg_latency, 2),
        "p95_latency_ms": round(p95_latency, 2),
        "samples_evaluated_for_accuracy": evaluated,
        "samples_correct": correct,
    }

    return {
        "overall": overall,
        "by_category": category_metrics,
        "by_dataset": dataset_metrics,
    }


# ---------------------------------------------------------------------------
# Category & Dataset Breakdowns
# ---------------------------------------------------------------------------

def _compute_category_breakdown(results: List[EvalResult]) -> Dict[str, Dict[str, Any]]:
    """Compute metrics broken down by logic category."""
    buckets: Dict[str, List[EvalResult]] = defaultdict(list)

    for r in results:
        buckets[r.category].append(r)

    breakdown = {}
    for category, cat_results in sorted(buckets.items()):
        n = len(cat_results)
        exec_ok = sum(1 for r in cat_results if r.execution_success)
        drift = sum(1 for r in cat_results if r.drift_detected)

        correct = 0
        evaluated = 0
        for r in cat_results:
            if r.expected_answer is not None:
                evaluated += 1
                if check_answer_correct(r.predicted_answer, r.expected_answer, r.dataset):
                    correct += 1

        breakdown[category] = {
            "count": n,
            "execution_success_rate": round(exec_ok / n, 4),
            "drift_detection_rate": round(drift / n, 4),
            "answer_accuracy": round(correct / evaluated, 4) if evaluated > 0 else None,
        }

    return breakdown


def _compute_dataset_breakdown(results: List[EvalResult]) -> Dict[str, Dict[str, Any]]:
    """Compute metrics broken down by source dataset."""
    buckets: Dict[str, List[EvalResult]] = defaultdict(list)

    for r in results:
        buckets[r.dataset].append(r)

    breakdown = {}
    for dataset, ds_results in sorted(buckets.items()):
        n = len(ds_results)
        exec_ok = sum(1 for r in ds_results if r.execution_success)
        drift = sum(1 for r in ds_results if r.drift_detected)

        correct = 0
        evaluated = 0
        for r in ds_results:
            if r.expected_answer is not None:
                evaluated += 1
                if check_answer_correct(r.predicted_answer, r.expected_answer, r.dataset):
                    correct += 1

        latencies = [r.latency_ms for r in ds_results if r.latency_ms > 0]

        breakdown[dataset] = {
            "count": n,
            "execution_success_rate": round(exec_ok / n, 4),
            "drift_detection_rate": round(drift / n, 4),
            "answer_accuracy": round(correct / evaluated, 4) if evaluated > 0 else None,
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0,
        }

    return breakdown


# ---------------------------------------------------------------------------
# Comparative Metrics (LGP vs Baseline)
# ---------------------------------------------------------------------------

def compute_comparative_metrics(
    lgp_results: List[EvalResult],
    baseline_results: List[EvalResult],
    baseline_name: str = "baseline",
) -> Dict[str, Any]:
    """
    Compute comparative metrics between LGP and a baseline system.

    Measures:
        - Accuracy improvement
        - Hallucination reduction
        - Consistency improvement
    """
    lgp_metrics = compute_metrics(lgp_results)
    baseline_metrics = compute_metrics(baseline_results)

    lgp_overall = lgp_metrics.get("overall", {})
    baseline_overall = baseline_metrics.get("overall", {})

    lgp_acc = lgp_overall.get("answer_accuracy", 0) or 0
    baseline_acc = baseline_overall.get("answer_accuracy", 0) or 0

    # Count hallucinations: execution failures or drift-detected
    lgp_halluc = sum(1 for r in lgp_results if not r.execution_success or r.drift_detected)
    base_halluc = sum(1 for r in baseline_results if not r.execution_success)

    total_lgp = len(lgp_results) or 1
    total_base = len(baseline_results) or 1

    lgp_halluc_rate = lgp_halluc / total_lgp
    base_halluc_rate = base_halluc / total_base

    halluc_reduction = (
        (base_halluc_rate - lgp_halluc_rate) / base_halluc_rate
        if base_halluc_rate > 0 else 0
    )

    return {
        "lgp": lgp_overall,
        f"{baseline_name}": baseline_overall,
        "comparison": {
            "accuracy_improvement": round(lgp_acc - baseline_acc, 4),
            "accuracy_improvement_pct": round(
                ((lgp_acc - baseline_acc) / baseline_acc) * 100, 2
            ) if baseline_acc > 0 else None,
            "lgp_hallucination_rate": round(lgp_halluc_rate, 4),
            f"{baseline_name}_hallucination_rate": round(base_halluc_rate, 4),
            "hallucination_reduction_pct": round(halluc_reduction * 100, 2),
        },
    }
