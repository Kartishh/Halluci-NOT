"""
evaluation/runner.py

HalluciNOT (LGP)
---------------------------------
Evaluation Execution Engine

Runs evaluation samples through the LGP pipeline or baseline runners.
Captures per-sample results with full audit traces.

Features:
    - Checkpoint/resume via JSONL
    - Per-sample timeout and error handling
    - Configurable batch processing
    - Full audit trace capture

Author: LGP Framework
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional

from tqdm import tqdm

from evaluation.datasets import EvalSample

logger = logging.getLogger("LGP.Eval.Runner")
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Result Schema
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """
    Result of running a single EvalSample through a pipeline.
    """
    sample_id: str
    query: str
    expected_answer: Any
    predicted_answer: Any
    dataset: str
    category: str

    # Pipeline outcomes
    execution_success: bool
    drift_detected: bool
    nli_triggered: bool

    # Timing
    latency_ms: float

    # Audit
    audit_trace: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# LGP Pipeline Runner
# ---------------------------------------------------------------------------

def run_lgp_pipeline(sample: EvalSample) -> EvalResult:
    """
    Run a single sample through the full LGP pipeline with Reflexion.
    """
    from core.gemini_llm import get_gemini_llm
    from core.reflexion import get_reflexion_engine

    llm = get_gemini_llm()
    engine = get_reflexion_engine(llm)

    start_time = time.time()
    error_msg = None
    drift_detected = False
    nli_triggered = False
    predicted = None
    audit = {}

    try:
        # Get baseline reasoning to force it via forced_reasoning? No, ReflexionEngine will generate it.
        # But we want a direct comparison. Let's just use engine.run(query) so it generates reasoning normally.
        res = engine.run(sample.query)

        predicted = res.final_answer
        drift_detected = res.drift_detected
        audit = {
            "drift_reports": res.drift_reports,
            "correction_applied": res.correction_applied,
            "correction_successful": res.correction_successful,
            "iterations_used": res.iterations_used,
            "execution_trace": res.execution_trace,
        }

    except Exception as e:
        error_msg = str(e)
        logger.warning(f"Pipeline error for {sample.id}: {error_msg}")

    elapsed_ms = (time.time() - start_time) * 1000

    return EvalResult(
        sample_id=sample.id,
        query=sample.query,
        expected_answer=sample.expected_answer,
        predicted_answer=predicted,
        dataset=sample.dataset,
        category=sample.category,
        execution_success=error_msg is None and predicted is not None,
        drift_detected=drift_detected,
        nli_triggered=nli_triggered,
        latency_ms=round(elapsed_ms, 2),
        audit_trace=audit,
        error=error_msg,
    )


# ---------------------------------------------------------------------------
# Evaluation Runner
# ---------------------------------------------------------------------------

class EvaluationRunner:
    """
    Orchestrates evaluation across a dataset.

    Features:
        - Progress tracking via tqdm
        - Checkpoint to JSONL for resume
        - Per-sample error isolation
    """

    def __init__(
        self,
        pipeline_fn: Optional[Callable[[EvalSample], EvalResult]] = None,
        checkpoint_path: Optional[str] = None,
    ):
        self.pipeline_fn = pipeline_fn or run_lgp_pipeline
        self.checkpoint_path = checkpoint_path
        self._completed_ids: set = set()

        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_checkpoint()

    # ------------------------------------------------------------------
    # Checkpoint Management
    # ------------------------------------------------------------------

    def _load_checkpoint(self):
        """Load previously completed sample IDs."""
        try:
            with open(self.checkpoint_path, "r") as f:
                for line in f:
                    data = json.loads(line.strip())
                    self._completed_ids.add(data.get("sample_id"))
            logger.info(
                f"Loaded checkpoint: {len(self._completed_ids)} completed."
            )
        except Exception as e:
            logger.warning(f"Checkpoint load error: {e}")

    def _save_result(self, result: EvalResult):
        """Append result to checkpoint file."""
        if self.checkpoint_path:
            os.makedirs(os.path.dirname(self.checkpoint_path) or ".", exist_ok=True)
            with open(self.checkpoint_path, "a") as f:
                f.write(json.dumps(result.to_dict(), default=str) + "\n")

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(
        self,
        samples: List[EvalSample],
        show_progress: bool = True,
    ) -> List[EvalResult]:
        """
        Run evaluation on all samples.

        Skips already-completed samples if checkpoint exists.
        """
        results: List[EvalResult] = []
        pending = [s for s in samples if s.id not in self._completed_ids]

        if len(pending) < len(samples):
            logger.info(
                f"Resuming: {len(samples) - len(pending)} already done, "
                f"{len(pending)} remaining."
            )

        iterator = tqdm(pending, desc="Evaluating", disable=not show_progress)

        for sample in iterator:
            iterator.set_postfix(dataset=sample.dataset, cat=sample.category)

            try:
                result = self.pipeline_fn(sample)
            except Exception as e:
                # Catch-all for unexpected errors
                result = EvalResult(
                    sample_id=sample.id,
                    query=sample.query,
                    expected_answer=sample.expected_answer,
                    predicted_answer=None,
                    dataset=sample.dataset,
                    category=sample.category,
                    execution_success=False,
                    drift_detected=False,
                    nli_triggered=False,
                    latency_ms=0.0,
                    error=str(e),
                )

            results.append(result)
            self._save_result(result)
            self._completed_ids.add(sample.id)

        return results


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    import argparse
    from evaluation.datasets import load_dataset_by_name
    from evaluation.metrics import compute_metrics

    parser = argparse.ArgumentParser(description="LGP Evaluation Runner")
    parser.add_argument("--dataset", required=True, choices=["halueval", "gsm_hard", "popqa"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split", type=str, default="qa", help="HaluEval split")
    parser.add_argument("--output", type=str, default="results/eval_results.jsonl")
    parser.add_argument("--baseline", type=str, default=None, choices=["vanilla", "cot"])
    args = parser.parse_args()

    # Load dataset
    kwargs = {}
    if args.dataset == "halueval":
        kwargs["split"] = args.split

    samples = load_dataset_by_name(args.dataset, limit=args.limit, **kwargs)
    print(f"Loaded {len(samples)} samples from {args.dataset}")

    # Choose pipeline
    pipeline_fn = run_lgp_pipeline

    if args.baseline:
        from evaluation.baselines import get_baseline_runner
        pipeline_fn = get_baseline_runner(args.baseline)

    # Run evaluation
    runner = EvaluationRunner(
        pipeline_fn=pipeline_fn,
        checkpoint_path=args.output,
    )

    results = runner.run(samples)

    # Compute & print metrics
    metrics = compute_metrics(results)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
