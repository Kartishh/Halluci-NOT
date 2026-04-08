#!/usr/bin/env python3
"""
run_factored_eval.py

HalluciNOT (LGP) — Factored Verification Evaluation Pipeline
================================================================

Runs the upgraded multi-signal factored verification pipeline against:
    - 50 controlled drift cases (known drift types)
    - 20 GSM-8K cases (real-world LLM reasoning)
    - 10 clean cases (correct reasoning → FP measurement)

Total: 80 cases

Metrics computed (NO hardcoding, NO forced success):
    - detection_rate = drift_cases_detected / total_drift_cases
    - correction_success_rate = corrected_cases / drift_cases_detected
    - false_positive_rate = clean_cases_flagged / total_clean_cases

Output:
    - results/final_comparison_table.txt
    - results/factored_eval_metrics.json
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import logging
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load .env
_env_path = os.path.join(BASE_DIR, ".env")
if os.path.exists(_env_path):
    with open(_env_path) as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ[k] = v

# Silence noisy loggers
logging.getLogger("LGP").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

from core.groq_llm import get_groq_llm, GroqLLM
from core.reflexion import get_reflexion_engine, ReflexionEngine


# ---------------------------------------------------------------------------
# GSM-8K Queries (20 cases)
# ---------------------------------------------------------------------------

GSM_QUERIES = [
    {"id": "gsm_01", "query": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells every remaining egg at the farmers' market for $2. How much does she make every day at the farmers' market?", "expected": 18, "logic_type": "multistep"},
    {"id": "gsm_02", "query": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?", "expected": 3, "logic_type": "fraction"},
    {"id": "gsm_03", "query": "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?", "expected": 70000, "logic_type": "percentage"},
    {"id": "gsm_04", "query": "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?", "expected": 540, "logic_type": "multistep"},
    {"id": "gsm_05", "query": "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy. She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed. In the afternoon, she gives her chickens another 25 cups of feed. If Wendi has 20 chickens, how many cups of feed does she need to give her chickens in the final meal of the day?", "expected": 20, "logic_type": "multistep"},
    {"id": "gsm_06", "query": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?", "expected": 72, "logic_type": "fraction"},
    {"id": "gsm_07", "query": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "expected": 10, "logic_type": "fraction"},
    {"id": "gsm_08", "query": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?", "expected": 5, "logic_type": "multistep"},
    {"id": "gsm_09", "query": "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?", "expected": 42, "logic_type": "multistep"},
    {"id": "gsm_10", "query": "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?", "expected": 624, "logic_type": "multistep"},
    {"id": "gsm_11", "query": "Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more of those that are purple. There are only 25% as many green flowers as there are yellow and purple flowers. How many flowers does Mark have in his garden?", "expected": 35, "logic_type": "percentage"},
    {"id": "gsm_12", "query": "Albert is wondering how much pizza he can eat in one day. He buys 2 large pizzas and 2 small pizzas. A large pizza has 16 slices and a small pizza has 8 slices. If he eats it all, how many pizza slices does he eat that day?", "expected": 48, "logic_type": "arithmetic"},
    {"id": "gsm_13", "query": "Ken created a care package to send to his brother, who was away at boarding school. Ken placed a box on a scale, and then he poured in enough jelly beans to bring the weight to 2 pounds. Then, he added enough brownies to cause the weight to triple. Next, he added another 2 pounds of jelly beans. And finally, he added enough gummy worms to double the weight once more. What was the final weight of the box of goodies, in pounds?", "expected": 16, "logic_type": "sequential"},
    {"id": "gsm_14", "query": "Alexis is applying for a new job and bought a new set of business clothes to wear to the interview. She went to a department store with a budget of $200 and spent $30 on a pair of trousers and $46 on a blouse. She also purchased a pair of shoes and a belt. If she has $16 left from her budget, how much did the pair of shoes and belt cost, together?", "expected": 108, "logic_type": "arithmetic"},
    {"id": "gsm_15", "query": "Tina makes $18.00 an hour. If she works more than 8 hours per shift, she is eligible for overtime, which is paid by your hourly wage + 1/2 your hourly wage. If she works 10 hours every day for 5 days, how much money does she make?", "expected": 990, "logic_type": "multistep"},
    {"id": "gsm_16", "query": "A merchant wants to make a choice of purchase between 2 purchase plans: jewelry worth $5,000 or electronic gadgets worth $8,000. His financial advisor speculates that the jewelry market will go up 2.5% while the electronic gadget market will rise 1.2% within the same month. If the merchant wants to maximize profit at the end of the month by making a choice, how much profit would this be?", "expected": 125, "logic_type": "percentage"},
    {"id": "gsm_17", "query": "Two trains leave San Rafael at the same time. They begin traveling westward, both traveling for 80 miles. The next day, they travel northward, covering 150 miles. What's the distance covered by each train in the two days?", "expected": 230, "logic_type": "arithmetic"},
    {"id": "gsm_18", "query": "Eliza's rate per hour for the first 40 hours she works each week is $10. She also receives an overtime pay of 1.2 times her regular hourly rate. If Eliza worked for 45 hours this week, how much are her earnings for this week?", "expected": 460, "logic_type": "multistep"},
    {"id": "gsm_19", "query": "A new program had 60 downloads in the first month. The number of downloads in the second month was three times more than the downloads in the first month, but then reduced by 30% in the third month. How many downloads did the program have total over the three months?", "expected": 366, "logic_type": "percentage"},
    {"id": "gsm_20", "query": "Toula went to the bakery and bought various types of pastries. She bought 3 dozen donuts which cost $68 per dozen, 2 dozen mini cupcakes which cost $80 per dozen, and 6 dozen mini cheesecakes for $55 per dozen. How much was the total cost?", "expected": 694, "logic_type": "arithmetic"},
]


# ---------------------------------------------------------------------------
# Answer Checker
# ---------------------------------------------------------------------------

def is_correct(predicted: float, expected: float, tol: float = 0.5) -> bool:
    if predicted is None or (isinstance(predicted, float) and math.isnan(predicted)):
        return False
    if expected == 0:
        return abs(predicted) < tol
    return math.isclose(predicted, expected, rel_tol=0.05, abs_tol=tol)


# ---------------------------------------------------------------------------
# Run LGP Pipeline (with injected reasoning support)
# ---------------------------------------------------------------------------

def run_lgp(engine: ReflexionEngine, query: str, injected_reasoning: Optional[str] = None):
    """
    LGP: Query → LLM → Decompose → Execute → SSCE + Factored → Reflexion → Answer.
    """
    try:
        if injected_reasoning is not None:
            result = engine.run(query, forced_reasoning=injected_reasoning)
        else:
            result = engine.run(query)

        drift_details = ""
        if result.drift_reports:
            drift_details = "; ".join(
                r.get("reason", "")[:100] for r in result.drift_reports
            )

        return {
            "answer": result.final_answer,
            "reasoning": result.reasoning,
            "drift_detected": result.drift_detected,
            "correction_applied": result.correction_applied,
            "correction_successful": result.correction_successful,
            "iterations": result.iterations_used,
            "drift_details": drift_details,
            "factored_drift": result.factored_drift,
            "factored_reports": result.factored_reports,
            "factored_llm_calls": result.factored_llm_calls,
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "answer": float("nan"),
            "reasoning": f"ERROR: {e}",
            "drift_detected": False,
            "correction_applied": False,
            "correction_successful": False,
            "iterations": 0,
            "drift_details": str(e),
            "factored_drift": False,
            "factored_reports": [],
            "factored_llm_calls": 0,
        }


# ---------------------------------------------------------------------------
# Main Evaluation
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 70)
    print("  HalluciNOT (LGP) — Factored Verification Evaluation")
    print("  Multi-Signal Drift Detection (Signal 1 + 2 + conditional 3)")
    print("=" * 70)

    t0 = time.time()

    llm = get_groq_llm()
    engine = get_reflexion_engine(llm)

    # ------------------------------------------------------------------
    # Load Controlled Drift Dataset (50 cases)
    # ------------------------------------------------------------------
    drift_dataset_path = os.path.join(BASE_DIR, "data", "controlled_drift_dataset.json")
    with open(drift_dataset_path, "r") as f:
        controlled_drift = json.load(f)

    # Take all 50 cases
    drift_cases = controlled_drift[:50]
    # Take first 10 with correct reasoning for FP measurement
    clean_cases = controlled_drift[:10]

    print(f"\n  Drift cases: {len(drift_cases)}")
    print(f"  GSM cases:   {len(GSM_QUERIES)}")
    print(f"  Clean cases:  {len(clean_cases)}")
    print(f"  Total:        {len(drift_cases) + len(GSM_QUERIES) + len(clean_cases)}")
    print()

    # ------------------------------------------------------------------
    # Phase 1: Controlled Drift Cases (inject drifted reasoning)
    # ------------------------------------------------------------------
    drift_results = []
    api_errors = 0

    print("\n--- Phase 1: Controlled Drift Cases ---")
    for i, case in enumerate(drift_cases):
        qid = f"drift_{i+1:02d}"
        query = case["question"]
        expected = case["answer"]
        drifted_reasoning = case["drifted_reasoning"]
        drift_type = case.get("drift_type", "unknown")

        sys.stdout.write(f"\r  [{i+1}/{len(drift_cases)}] {qid}...")
        sys.stdout.flush()

        # Rate limiting
        if i > 0:
            time.sleep(2)

        try:
            result = run_lgp(engine, query, injected_reasoning=drifted_reasoning)
            l_correct = is_correct(result["answer"], expected)

            drift_results.append({
                "case_id": qid,
                "drift_type": drift_type,
                "expected": expected,
                "answer": result["answer"],
                "correct": l_correct,
                "drift_detected": result["drift_detected"],
                "factored_drift": result["factored_drift"],
                "correction_applied": result["correction_applied"],
                "correction_successful": result["correction_successful"],
                "factored_llm_calls": result["factored_llm_calls"],
            })

            d_flag = "🔍" if result["drift_detected"] else "  "
            c_flag = "✓" if l_correct else "✗"
            print(f"\r  [{i+1}/{len(drift_cases)}] {qid} {d_flag} {c_flag} "
                  f"(expected={expected}, got={result['answer']}, type={drift_type})")

        except Exception as e:
            api_errors += 1
            print(f"\r  [{i+1}/{len(drift_cases)}] {qid} ERROR: {str(e)[:60]}")
            if api_errors > 5:
                print("\n  ⚠ Too many API errors. Stopping drift cases.")
                break
            time.sleep(5)
            continue

    # ------------------------------------------------------------------
    # Phase 2: GSM-8K Cases (real-world, no injection)
    # ------------------------------------------------------------------
    gsm_results = []
    api_errors = 0

    print("\n\n--- Phase 2: GSM-8K Cases ---")
    for i, item in enumerate(GSM_QUERIES):
        qid = item["id"]
        query = item["query"]
        expected = item["expected"]

        sys.stdout.write(f"\r  [{i+1}/{len(GSM_QUERIES)}] {qid}...")
        sys.stdout.flush()

        # Rate limiting
        if i > 0:
            time.sleep(2)

        try:
            result = run_lgp(engine, query)
            l_correct = is_correct(result["answer"], expected)

            gsm_results.append({
                "case_id": qid,
                "expected": expected,
                "answer": result["answer"],
                "correct": l_correct,
                "drift_detected": result["drift_detected"],
                "factored_drift": result["factored_drift"],
                "correction_applied": result["correction_applied"],
                "correction_successful": result["correction_successful"],
                "factored_llm_calls": result["factored_llm_calls"],
            })

            c_flag = "✓" if l_correct else "✗"
            d_flag = "🔍" if result["drift_detected"] else "  "
            print(f"\r  [{i+1}/{len(GSM_QUERIES)}] {qid} {d_flag} {c_flag} "
                  f"(expected={expected}, got={result['answer']})")

        except Exception as e:
            api_errors += 1
            print(f"\r  [{i+1}/{len(GSM_QUERIES)}] {qid} ERROR: {str(e)[:60]}")
            if api_errors > 5:
                print("\n  ⚠ Too many API errors. Stopping GSM cases.")
                break
            time.sleep(5)
            continue

    # ------------------------------------------------------------------
    # Phase 3: Clean Cases (correct reasoning → FP measurement)
    # ------------------------------------------------------------------
    clean_results = []
    api_errors = 0

    print("\n\n--- Phase 3: Clean Cases (FP Measurement) ---")
    for i, case in enumerate(clean_cases):
        qid = f"clean_{i+1:02d}"
        query = case["question"]
        expected = case["answer"]
        correct_reasoning = case["correct_reasoning"]

        sys.stdout.write(f"\r  [{i+1}/{len(clean_cases)}] {qid}...")
        sys.stdout.flush()

        # Rate limiting
        if i > 0:
            time.sleep(2)

        try:
            result = run_lgp(engine, query, injected_reasoning=correct_reasoning)
            l_correct = is_correct(result["answer"], expected)

            clean_results.append({
                "case_id": qid,
                "expected": expected,
                "answer": result["answer"],
                "correct": l_correct,
                "drift_detected": result["drift_detected"],
                "factored_drift": result["factored_drift"],
                "false_positive": result["drift_detected"],  # drift flagged on CLEAN input
            })

            fp_flag = "FP!" if result["drift_detected"] else "OK "
            print(f"\r  [{i+1}/{len(clean_cases)}] {qid} {fp_flag} "
                  f"(expected={expected}, got={result['answer']})")

        except Exception as e:
            api_errors += 1
            print(f"\r  [{i+1}/{len(clean_cases)}] {qid} ERROR: {str(e)[:60]}")
            if api_errors > 3:
                print("\n  ⚠ Too many API errors. Stopping clean cases.")
                break
            time.sleep(5)
            continue

    elapsed = time.time() - t0

    # ------------------------------------------------------------------
    # Compute Metrics (NO hardcoding, NO forced success)
    # ------------------------------------------------------------------

    # Drift detection metrics (out of drift cases)
    total_drift_cases = len(drift_results)
    drift_detected_count = sum(1 for r in drift_results if r["drift_detected"])
    detection_rate = (drift_detected_count / total_drift_cases * 100) if total_drift_cases > 0 else 0

    # Correction metrics (out of detected cases)
    correction_attempted = sum(1 for r in drift_results if r["correction_applied"])
    correction_succeeded = sum(1 for r in drift_results if r["correction_successful"])
    correction_success_rate = (correction_succeeded / drift_detected_count * 100) if drift_detected_count > 0 else 0

    # False positive metrics (out of clean cases)
    total_clean_cases = len(clean_results)
    false_positives = sum(1 for r in clean_results if r["false_positive"])
    false_positive_rate = (false_positives / total_clean_cases * 100) if total_clean_cases > 0 else 0

    # GSM accuracy
    gsm_correct = sum(1 for r in gsm_results if r["correct"])
    gsm_accuracy = (gsm_correct / len(gsm_results) * 100) if gsm_results else 0

    # Drift accuracy (after correction)
    drift_correct = sum(1 for r in drift_results if r["correct"])
    drift_accuracy = (drift_correct / total_drift_cases * 100) if total_drift_cases > 0 else 0

    # Per-drift-type breakdown
    drift_by_type = {}
    for r in drift_results:
        dt = r["drift_type"]
        if dt not in drift_by_type:
            drift_by_type[dt] = {"total": 0, "detected": 0, "corrected": 0}
        drift_by_type[dt]["total"] += 1
        if r["drift_detected"]:
            drift_by_type[dt]["detected"] += 1
        if r["correction_successful"]:
            drift_by_type[dt]["corrected"] += 1

    # Total LLM calls for Signal 3
    total_signal3_calls = sum(r.get("factored_llm_calls", 0) for r in drift_results)
    total_signal3_calls += sum(r.get("factored_llm_calls", 0) for r in gsm_results)

    # ------------------------------------------------------------------
    # Build Metrics JSON
    # ------------------------------------------------------------------

    metrics = {
        "system": "LGP (Updated — Factored Verification)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(elapsed, 1),
        "total_cases": total_drift_cases + len(gsm_results) + total_clean_cases,
        "detection": {
            "total_drift_cases": total_drift_cases,
            "drift_detected": drift_detected_count,
            "detection_rate_pct": round(detection_rate, 1),
        },
        "correction": {
            "correction_attempted": correction_attempted,
            "correction_succeeded": correction_succeeded,
            "correction_success_rate_pct": round(correction_success_rate, 1),
        },
        "false_positives": {
            "total_clean_cases": total_clean_cases,
            "false_positives": false_positives,
            "false_positive_rate_pct": round(false_positive_rate, 1),
        },
        "gsm_accuracy": {
            "total": len(gsm_results),
            "correct": gsm_correct,
            "accuracy_pct": round(gsm_accuracy, 1),
        },
        "drift_accuracy_after_correction": {
            "total": total_drift_cases,
            "correct": drift_correct,
            "accuracy_pct": round(drift_accuracy, 1),
        },
        "per_drift_type": drift_by_type,
        "signal3_llm_calls": total_signal3_calls,
    }

    # ------------------------------------------------------------------
    # Save Metrics JSON
    # ------------------------------------------------------------------

    metrics_path = os.path.join(RESULTS_DIR, "factored_eval_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  ✓ {metrics_path}")

    # ------------------------------------------------------------------
    # Generate Comparison Table (Task 8)
    # ------------------------------------------------------------------

    table_lines = []
    table_lines.append("=" * 70)
    table_lines.append("  HalluciNOT (LGP) — Final Comparison Table")
    table_lines.append(f"  Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    table_lines.append("=" * 70)
    table_lines.append("")
    table_lines.append(f"  Total Cases: {metrics['total_cases']}")
    table_lines.append(f"  (Drift: {total_drift_cases}, GSM: {len(gsm_results)}, Clean: {total_clean_cases})")
    table_lines.append(f"  Elapsed: {elapsed:.1f}s")
    table_lines.append("")
    table_lines.append("  ┌─────────────────────┬────────────┬─────────────┬──────────┐")
    table_lines.append("  │ System              │ Detection  │ Correction  │ FP       │")
    table_lines.append("  ├─────────────────────┼────────────┼─────────────┼──────────┤")
    table_lines.append(f"  │ LGP (Updated)       │ {detection_rate:>8.1f}%  │ {correction_success_rate:>9.1f}%  │ {false_positive_rate:>5.1f}%  │")
    table_lines.append("  └─────────────────────┴────────────┴─────────────┴──────────┘")
    table_lines.append("")
    table_lines.append("")
    table_lines.append("  Per-Drift-Type Breakdown:")
    table_lines.append("  ┌────────────────┬───────┬──────────┬───────────┐")
    table_lines.append("  │ Drift Type     │ Total │ Detected │ Corrected │")
    table_lines.append("  ├────────────────┼───────┼──────────┼───────────┤")
    for dt, stats in sorted(drift_by_type.items()):
        det_pct = round(stats["detected"] / stats["total"] * 100, 1) if stats["total"] > 0 else 0
        cor_pct = round(stats["corrected"] / stats["detected"] * 100, 1) if stats["detected"] > 0 else 0
        table_lines.append(f"  │ {dt:<14} │ {stats['total']:>5} │ {stats['detected']:>3} ({det_pct:>5.1f}%)│ {stats['corrected']:>3} ({cor_pct:>5.1f}%)│")
    table_lines.append("  └────────────────┴───────┴──────────┴───────────┘")
    table_lines.append("")
    table_lines.append("")
    table_lines.append(f"  GSM-8K Accuracy: {gsm_accuracy:.1f}% ({gsm_correct}/{len(gsm_results)})")
    table_lines.append(f"  Drift Accuracy (after correction): {drift_accuracy:.1f}% ({drift_correct}/{total_drift_cases})")
    table_lines.append(f"  Signal 3 (LLM) calls: {total_signal3_calls}")
    table_lines.append("")
    table_lines.append("  Key Metrics:")
    table_lines.append(f"    detection_rate      = {detection_rate:.1f}%")
    table_lines.append(f"    correction_success  = {correction_success_rate:.1f}%")
    table_lines.append(f"    false_positive_rate = {false_positive_rate:.1f}%")
    table_lines.append("")

    table_text = "\n".join(table_lines)

    table_path = os.path.join(RESULTS_DIR, "final_comparison_table.txt")
    with open(table_path, "w") as f:
        f.write(table_text)
    print(f"  ✓ {table_path}")

    # ------------------------------------------------------------------
    # Print to console
    # ------------------------------------------------------------------
    print("\n" + table_text)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  EVALUATION COMPLETE")
    print("=" * 70)
    print(f"\n  detection_rate      = {detection_rate:.1f}%")
    print(f"  correction_success  = {correction_success_rate:.1f}%")
    print(f"  false_positive_rate = {false_positive_rate:.1f}%")
    print(f"\n  Elapsed: {elapsed:.1f}s")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
