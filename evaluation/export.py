"""
evaluation/export.py

HalluciNOT (LGP)
---------------------------------
Results Export for Research Paper

Generates publication-ready output formats:
    1. JSON — Full detailed results
    2. CSV — Summary tables
    3. LaTeX — Paper-ready tables
    4. Markdown — Quick review

Author: LGP Framework
"""

from __future__ import annotations

import csv
import json
import logging
import os
from typing import Any, Dict, List, Optional

from evaluation.runner import EvalResult
from evaluation.metrics import compute_metrics, compute_comparative_metrics

logger = logging.getLogger("LGP.Eval.Export")
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# JSON Export
# ---------------------------------------------------------------------------

def export_json(
    results: List[EvalResult],
    output_path: str,
    include_audit: bool = True,
) -> None:
    """Export full results as JSON."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    data = {
        "metadata": {
            "total_samples": len(results),
            "datasets": list(set(r.dataset for r in results)),
            "categories": list(set(r.category for r in results)),
        },
        "metrics": compute_metrics(results),
        "results": [
            {k: v for k, v in r.to_dict().items()
             if include_audit or k != "audit_trace"}
            for r in results
        ],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

    logger.info(f"JSON results exported to {output_path}")


# ---------------------------------------------------------------------------
# CSV Export
# ---------------------------------------------------------------------------

def export_csv(
    results: List[EvalResult],
    output_path: str,
) -> None:
    """Export results summary as CSV."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fields = [
        "sample_id", "dataset", "category", "query",
        "expected_answer", "predicted_answer",
        "execution_success", "drift_detected", "nli_triggered",
        "latency_ms", "error",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row = r.to_dict()
            # Truncate long fields
            row["query"] = str(row.get("query", ""))[:200]
            row["predicted_answer"] = str(row.get("predicted_answer", ""))[:200]
            row["expected_answer"] = str(row.get("expected_answer", ""))[:200]
            writer.writerow(row)

    logger.info(f"CSV results exported to {output_path}")


# ---------------------------------------------------------------------------
# LaTeX Table Export
# ---------------------------------------------------------------------------

def export_latex_table(
    metrics: Dict[str, Any],
    output_path: str,
    caption: str = "HalluciNOT Evaluation Results",
    label: str = "tab:eval_results",
) -> None:
    """
    Generate a LaTeX table from metrics for direct paper inclusion.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    overall = metrics.get("overall", {})

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"\textbf{Metric} & \textbf{Value} \\",
        r"\midrule",
    ]

    metric_labels = {
        "total_samples": "Total Samples",
        "execution_success_rate": "Execution Success Rate",
        "answer_accuracy": "Answer Accuracy",
        "drift_detection_rate": "Drift Detection Rate",
        "ffact_score": "FFactScore",
        "avg_latency_ms": "Avg Latency (ms)",
        "p95_latency_ms": "P95 Latency (ms)",
        "nli_trigger_rate": "NLI Trigger Rate",
        "error_rate": "Error Rate",
    }

    for key, display_name in metric_labels.items():
        value = overall.get(key, "N/A")
        if isinstance(value, float) and key != "avg_latency_ms" and key != "p95_latency_ms":
            value = f"{value:.2%}"
        elif isinstance(value, float):
            value = f"{value:.1f}"
        lines.append(f"{display_name} & {value} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"LaTeX table exported to {output_path}")


def export_comparative_latex(
    lgp_results: List[EvalResult],
    baseline_results: List[EvalResult],
    baseline_name: str,
    output_path: str,
    caption: str = "LGP vs Baseline Comparison",
    label: str = "tab:comparison",
) -> None:
    """
    Generate a comparative LaTeX table (LGP vs baseline).
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    comp = compute_comparative_metrics(lgp_results, baseline_results, baseline_name)

    lgp = comp.get("lgp", {})
    baseline = comp.get(baseline_name, {})
    comparison = comp.get("comparison", {})

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        f"\\textbf{{Metric}} & \\textbf{{HalluciNOT}} & \\textbf{{{baseline_name.title()}}} \\\\",
        r"\midrule",
    ]

    compare_metrics = [
        ("Answer Accuracy", "answer_accuracy"),
        ("Execution Success", "execution_success_rate"),
        ("Avg Latency (ms)", "avg_latency_ms"),
    ]

    for display_name, key in compare_metrics:
        lgp_val = lgp.get(key, "N/A")
        base_val = baseline.get(key, "N/A")

        if isinstance(lgp_val, float) and key != "avg_latency_ms":
            lgp_val = f"{lgp_val:.2%}"
        elif isinstance(lgp_val, float):
            lgp_val = f"{lgp_val:.1f}"

        if isinstance(base_val, float) and key != "avg_latency_ms":
            base_val = f"{base_val:.2%}"
        elif isinstance(base_val, float):
            base_val = f"{base_val:.1f}"

        lines.append(f"{display_name} & {lgp_val} & {base_val} \\\\")

    # Add improvement row
    improvement = comparison.get("accuracy_improvement_pct", "N/A")
    if isinstance(improvement, (int, float)):
        improvement = f"+{improvement:.1f}\\%"

    halluc_red = comparison.get("hallucination_reduction_pct", "N/A")
    if isinstance(halluc_red, (int, float)):
        halluc_red = f"{halluc_red:.1f}\\%"

    lines.extend([
        r"\midrule",
        f"Accuracy Improvement & \\multicolumn{{2}}{{c}}{{{improvement}}} \\\\",
        f"Hallucination Reduction & \\multicolumn{{2}}{{c}}{{{halluc_red}}} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"Comparative LaTeX table exported to {output_path}")


# ---------------------------------------------------------------------------
# Markdown Export
# ---------------------------------------------------------------------------

def export_markdown_summary(
    results: List[EvalResult],
    output_path: str,
) -> None:
    """Export a Markdown summary for quick review."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    metrics = compute_metrics(results)
    overall = metrics.get("overall", {})

    lines = [
        "# HalluciNOT Evaluation Summary\n",
        "## Overall Metrics\n",
        "| Metric | Value |",
        "|---|---|",
    ]

    for key, value in overall.items():
        display = key.replace("_", " ").title()
        if isinstance(value, float) and "rate" in key or "accuracy" in key or "score" in key:
            value = f"{value:.2%}"
        elif isinstance(value, float):
            value = f"{value:.1f}"
        lines.append(f"| {display} | {value} |")

    # Category breakdown
    by_cat = metrics.get("by_category", {})
    if by_cat:
        lines.extend(["\n## By Logic Category\n",
                       "| Category | Count | Exec Success | Accuracy | Drift Rate |",
                       "|---|---|---|---|---|"])
        for cat, m in by_cat.items():
            acc = f"{m['answer_accuracy']:.2%}" if m.get("answer_accuracy") is not None else "N/A"
            lines.append(
                f"| {cat} | {m['count']} | {m['execution_success_rate']:.2%} | "
                f"{acc} | {m['drift_detection_rate']:.2%} |"
            )

    # Dataset breakdown
    by_ds = metrics.get("by_dataset", {})
    if by_ds:
        lines.extend(["\n## By Dataset\n",
                       "| Dataset | Count | Exec Success | Accuracy | Avg Latency |",
                       "|---|---|---|---|---|"])
        for ds, m in by_ds.items():
            acc = f"{m['answer_accuracy']:.2%}" if m.get("answer_accuracy") is not None else "N/A"
            lines.append(
                f"| {ds} | {m['count']} | {m['execution_success_rate']:.2%} | "
                f"{acc} | {m['avg_latency_ms']:.1f}ms |"
            )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"Markdown summary exported to {output_path}")
