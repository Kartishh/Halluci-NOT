#!/usr/bin/env python3
"""
lgp_eval.py

HalluciNOT (LGP) — Agentic Research Evaluation Pipeline
==========================================================

Two pipelines compared:

    BASELINE:   Query → LLM → Answer  (no verification)
    HALLUCINOT: Query → LLM → SSCE + Reflexion loop → Corrected Answer

Metrics:
    - Accuracy (baseline vs LGP)
    - Drift Detection Rate
    - Correction Success Rate  ← critical
    - Reflexion iterations used

Output:
    - Paper-ready Markdown table
    - Summary metrics JSON
    - Per-query CSV
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
import logging
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Setup paths
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


# ---------------------------------------------------------------------------
# Imports (after path setup)
# ---------------------------------------------------------------------------

from core.groq_llm import get_groq_llm, GroqLLM
from core.reflexion import get_reflexion_engine, ReflexionEngine
from evaluation.liar_agent import get_liar_agent, LiarAgent


# ---------------------------------------------------------------------------
# Curated Datasets
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

SYNTHETIC_DRIFT_QUERIES = [
    {"id": "syn_01", "query": "Let x = 10. Let y = 5. What is x + y?", "expected": 15, "logic_type": "simple_arithmetic", "drift_test": False},
    {"id": "syn_02", "query": "A shirt costs $25. Tax is 8%. What is the total cost?", "expected": 27, "logic_type": "percentage", "drift_test": False},
    {"id": "syn_03", "query": "Let price = 50. Apply 20% discount. What is the sale price?", "expected": 40, "logic_type": "percentage", "drift_test": False},
    {"id": "syn_04", "query": "Start with 100. Add 50. Subtract 30. Multiply by 2. What is the result?", "expected": 240, "logic_type": "sequential", "drift_test": False},
    {"id": "syn_05", "query": "A car travels 60 mph for 2.5 hours. How far does it go?", "expected": 150, "logic_type": "arithmetic", "drift_test": False},
    {"id": "syn_06", "query": "If x = 8 and y = x * 3, what is y - x?", "expected": 16, "logic_type": "dependency", "drift_test": False},
    {"id": "syn_07", "query": "A rectangle has length 12 and width 5. What is its area?", "expected": 60, "logic_type": "arithmetic", "drift_test": False},
    {"id": "syn_08", "query": "John has $200. He spends 30% on food and 20% on transport. How much does he have left?", "expected": 100, "logic_type": "percentage", "drift_test": False},
    {"id": "syn_09", "query": "A factory produces 150 units per day. It operates 5 days a week. How many units per week?", "expected": 750, "logic_type": "arithmetic", "drift_test": False},
    {"id": "syn_10", "query": "Temperature starts at 20°C. It increases by 5°C, then decreases by 8°C, then increases by 3°C. What is the final temperature?", "expected": 20, "logic_type": "sequential", "drift_test": False},
]


# ---------------------------------------------------------------------------
# Result Schema
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    case_id: str
    query: str
    logic_type: str
    expected: float
    baseline_answer: float
    baseline_correct: bool
    baseline_reasoning: str
    lgp_answer: float
    lgp_correct: bool
    lgp_reasoning: str
    drift_detected: bool
    correction_applied: bool
    correction_successful: bool
    reflexion_iterations: int
    drift_details: str


# ---------------------------------------------------------------------------
# Answer Checker
# ---------------------------------------------------------------------------

def is_correct(predicted: float, expected: float, tol: float = 0.5) -> bool:
    if predicted is None or math.isnan(predicted):
        return False
    if expected == 0:
        return abs(predicted) < tol
    return math.isclose(predicted, expected, rel_tol=0.05, abs_tol=tol)


# ---------------------------------------------------------------------------
# Baseline Pipeline
# ---------------------------------------------------------------------------

def run_baseline(llm: GroqLLM, query: str) -> tuple:
    """Baseline: Query → LLM → Answer (no verification)."""
    try:
        result = llm.generate_reasoning(query)
        return result.final_answer, result.reasoning
    except Exception as e:
        return float('nan'), f"ERROR: {e}"


# ---------------------------------------------------------------------------
# LGP Pipeline
# ---------------------------------------------------------------------------

"""def run_lgp(engine: ReflexionEngine, query: str) -> tuple:
    LGP: Query → LLM → Decompose → Execute → SSCE → Reflexion → Answer.
    try:
        result = engine.run(query)
        drift_details = ""
        if result.drift_reports:
            drift_details = "; ".join(
                r.get("reason", "")[:100] for r in result.drift_reports
            )
        return (
            result.final_answer,
            result.reasoning,
            result.drift_detected,
            result.correction_applied,
            result.correction_successful,
            result.iterations_used,
            drift_details,
        )
    except Exception as e:
        return float('nan'), f"ERROR: {e}", False, False, False, 0, str(e)"""
def run_lgp(engine, query, injected_reasoning=None):
    """
    LGP: Query → LLM → Decompose → Execute → SSCE → Reflexion → Answer.
    """

    try:
        # ⚠️ If we provide injected reasoning, override LLM
        if injected_reasoning is not None:
            result = engine.run(query, forced_reasoning=injected_reasoning)
        else:
            result = engine.run(query)

        drift_details = ""
        if result.drift_reports:
            drift_details = "; ".join(
                r.get("reason", "")[:100] for r in result.drift_reports
            )

        return (
            result.final_answer,
            result.reasoning,
            result.drift_detected,
            result.correction_applied,
            result.correction_successful,
            result.iterations_used,
            drift_details,
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return float('nan'), f"ERROR: {e}", False, False, False, 0, str(e)


# ---------------------------------------------------------------------------
# Liar Agent Stress Test
# ---------------------------------------------------------------------------

def run_liar_test(
    llm: GroqLLM,
    engine: ReflexionEngine,
    liar: LiarAgent,
) -> List[Dict]:
    """
    Stress test: Generate valid reasoning, inject drift, test SSCE.
    """
    test_cases = [
        ("If x = 5 and y = 3, what is x * y?", 15),
        ("A price is $40. Apply 10% discount. What is the sale price?", 36),
        ("Start with 100. Add 20. Subtract 10. What is the result?", 110),
    ]
    results = []

    for query, expected in test_cases:
        # Get valid reasoning
        valid = llm.generate_reasoning(query)

        for drift_type in ["redefinition", "sign_flip", "value_swap"]:
            injection = liar.inject_drift(
                valid.reasoning, valid.final_answer, drift_type
            )

            # Test if SSCE detects the injected drift
            #lgp = engine.run(query)  # Fresh run — SSCE should catch
            l_answer, l_reasoning, l_drift, l_correction, l_corr_success, l_iters, l_drift_details = run_lgp(
    engine, query, injected_reasoning=injection.flawed_reasoning
)

            results.append({
                "query": query[:60],
                "drift_type": drift_type,
                "drift_details": injection.drift_details[:80],
                #"ssce_detected": lgp.drift_detected,
                #"correction_applied": lgp.correction_applied,
                #"lgp_answer": lgp.final_answer,
                "ssce_detected": l_drift,
                "correction_applied": l_correction,
                "lgp_answer": l_answer,
                "correct": is_correct(l_answer, expected),
                "expected": expected,
                #"correct": is_correct(lgp.final_answer, expected),
            })

    return results


# ---------------------------------------------------------------------------
# Main Evaluation
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 70)
    print("  HalluciNOT (LGP) — Agentic Research Evaluation")
    print("  Groq LLM + SSCE + Reflexion Loop")
    print("=" * 70)

    t0 = time.time()

    llm = get_groq_llm()
    engine = get_reflexion_engine(llm)
    liar = get_liar_agent()

    # Combine datasets
    all_queries = []
    for q in GSM_QUERIES:
        all_queries.append({**q, "dataset": "gsm"})
    for q in SYNTHETIC_DRIFT_QUERIES:
        all_queries.append({**q, "dataset": "synthetic"})

    print(f"\n  Total queries: {len(all_queries)} (GSM: {len(GSM_QUERIES)}, Synthetic: {len(SYNTHETIC_DRIFT_QUERIES)})")
    print(f"  Model: {llm.model}")
    print(f"  Max Reflexion iterations: 3")
    print()

    results: List[EvalResult] = []
    api_errors = 0

    for i, item in enumerate(all_queries):
        qid = item["id"]
        query = item["query"]
        expected = item["expected"]
        logic_type = item["logic_type"]

        sys.stdout.write(f"\r  [{i+1}/{len(all_queries)}] {qid}...")
        sys.stdout.flush()

        # Rate limiting — Groq free tier
        if i > 0:
            time.sleep(2)

        try:
            # Baseline
            b_answer, b_reasoning = run_baseline(llm, query)
            b_correct = is_correct(b_answer, expected)

            time.sleep(1)  # Rate limit gap

            # LGP
            lgp_out = run_lgp(engine, query)
            l_answer = lgp_out[0]
            l_reasoning = lgp_out[1]
            l_drift = lgp_out[2]
            l_correction = lgp_out[3]
            l_corr_success = lgp_out[4]
            l_iters = lgp_out[5]
            l_drift_details = lgp_out[6]
            l_correct = is_correct(l_answer, expected)

            results.append(EvalResult(
                case_id=qid,
                query=query[:200],
                logic_type=logic_type,
                expected=expected,
                baseline_answer=b_answer,
                baseline_correct=b_correct,
                baseline_reasoning=b_reasoning[:300],
                lgp_answer=l_answer,
                lgp_correct=l_correct,
                lgp_reasoning=l_reasoning[:300],
                drift_detected=l_drift,
                correction_applied=l_correction,
                correction_successful=l_corr_success,
                reflexion_iterations=l_iters,
                drift_details=l_drift_details[:200],
            ))

            status = "✓" if l_correct else "✗"
            b_status = "✓" if b_correct else "✗"
            drift_flag = " 🔍DRIFT" if l_drift else ""
            corr_flag = " 🔧CORRECTED" if l_corr_success else ""
            print(f"\r  [{i+1}/{len(all_queries)}] {qid}  Base={b_answer}({b_status}) LGP={l_answer}({status}){drift_flag}{corr_flag}")

        except Exception as e:
            api_errors += 1
            print(f"\r  [{i+1}/{len(all_queries)}] {qid}  ERROR: {str(e)[:60]}")
            if api_errors > 5:
                print("\n  ⚠ Too many API errors. Stopping early.")
                break
            time.sleep(5)
            continue

    elapsed = time.time() - t0

    if not results:
        print("\n  No results collected. Exiting.")
        return

    # ------------------------------------------------------------------
    # Compute Metrics
    # ------------------------------------------------------------------
    n = len(results)
    gsm_r = [r for r in results if r.case_id.startswith("gsm")]
    syn_r = [r for r in results if r.case_id.startswith("syn")]

    def _metrics(res: List[EvalResult], label: str) -> Dict:
        if not res:
            return {"label": label, "n": 0}
        total = len(res)
        bc = sum(1 for r in res if r.baseline_correct)
        lc = sum(1 for r in res if r.lgp_correct)
        dd = sum(1 for r in res if r.drift_detected)
        ca = sum(1 for r in res if r.correction_applied)
        cs = sum(1 for r in res if r.correction_successful)
        imp = sum(1 for r in res if r.lgp_correct and not r.baseline_correct)
        reg = sum(1 for r in res if r.baseline_correct and not r.lgp_correct)
        return {
            "label": label, "n": total,
            "baseline_accuracy": round(bc / total * 100, 1),
            "lgp_accuracy": round(lc / total * 100, 1),
            "drift_detection_rate": round(dd / total * 100, 1),
            "correction_attempts": ca,
            "correction_successes": cs,
            "correction_success_rate": round(cs / ca * 100, 1) if ca > 0 else 0,
            "improvements": imp,
            "regressions": reg,
            "net_gain": imp - reg,
        }

    m_gsm = _metrics(gsm_r, "GSM-8K (curated)")
    m_syn = _metrics(syn_r, "Synthetic")
    m_all = _metrics(results, "Overall")

    # ------------------------------------------------------------------
    # Paper-Ready Markdown Table
    # ------------------------------------------------------------------
    md_lines = []
    md_lines.append("# HalluciNOT (LGP) — Evaluation Results\n")
    md_lines.append(f"*Generated: {time.strftime('%Y-%m-%d %H:%M')}*\n")
    md_lines.append(f"*Model: {llm.model} | Queries: {n} | Time: {elapsed:.1f}s*\n\n")

    # Summary table
    md_lines.append("## Summary Metrics\n")
    md_lines.append("| Metric | GSM-8K | Synthetic | Overall |")
    md_lines.append("|--------|--------|-----------|---------|")
    for key in ["baseline_accuracy", "lgp_accuracy", "drift_detection_rate", "correction_success_rate"]:
        label = key.replace("_", " ").title()
        v1 = f"{m_gsm.get(key, 'N/A')}%"
        v2 = f"{m_syn.get(key, 'N/A')}%"
        v3 = f"{m_all.get(key, 'N/A')}%"
        md_lines.append(f"| {label} | {v1} | {v2} | {v3} |")
    md_lines.append(f"| Improvements | {m_gsm.get('improvements', 0)} | {m_syn.get('improvements', 0)} | {m_all.get('improvements', 0)} |")
    md_lines.append(f"| Regressions | {m_gsm.get('regressions', 0)} | {m_syn.get('regressions', 0)} | {m_all.get('regressions', 0)} |")
    md_lines.append("")

    # Per-query table
    md_lines.append("## Per-Query Results\n")
    md_lines.append("| Case ID | Baseline | Drift Detected | Reflexion Correction | LGP Answer | Correct? |")
    md_lines.append("|---------|----------|---------------|---------------------|------------|----------|")
    for r in results:
        b_mark = "✓" if r.baseline_correct else "✗"
        l_mark = "✓" if r.lgp_correct else "✗"
        drift = "🔍 Yes" if r.drift_detected else "—"
        corr = "🔧 Yes" if r.correction_successful else ("⚠ Attempted" if r.correction_applied else "—")
        md_lines.append(
            f"| {r.case_id} | {r.baseline_answer} ({b_mark}) | "
            f"{drift} | {corr} | {r.lgp_answer} | {l_mark} (exp: {r.expected}) |"
        )
    md_lines.append("")

    # Correction cases detail
    corrections = [r for r in results if r.correction_applied]
    if corrections:
        md_lines.append("## Correction Cases (Detailed)\n")
        for r in corrections:
            md_lines.append(f"### {r.case_id}: {r.query[:80]}...")
            md_lines.append(f"- **Expected**: {r.expected}")
            md_lines.append(f"- **Baseline**: {r.baseline_answer} ({'✓' if r.baseline_correct else '✗'})")
            md_lines.append(f"- **LGP**: {r.lgp_answer} ({'✓' if r.lgp_correct else '✗'})")
            md_lines.append(f"- **Drift**: {r.drift_details[:150]}")
            md_lines.append(f"- **Iterations**: {r.reflexion_iterations}")
            md_lines.append(f"- **Correction Successful**: {'Yes' if r.correction_successful else 'No'}")
            md_lines.append("")

    md_text = "\n".join(md_lines)

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    print(f"\n\n  Saving results...\n")

    # Markdown report
    report_path = os.path.join(RESULTS_DIR, "lgp_evaluation_report.md")
    with open(report_path, "w") as f:
        f.write(md_text)
    print(f"  ✓ {report_path}")

    # Summary JSON
    summary = {"gsm": m_gsm, "synthetic": m_syn, "overall": m_all, "elapsed_seconds": round(elapsed, 1)}
    json_path = os.path.join(RESULTS_DIR, "lgp_summary_metrics.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  ✓ {json_path}")

    # CSV
    csv_path = os.path.join(RESULTS_DIR, "lgp_evaluation_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        w.writeheader()
        for r in results:
            w.writerow(asdict(r))
    print(f"  ✓ {csv_path}")

    # ------------------------------------------------------------------
    # Print Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  FINAL RESULTS")
    print(f"{'='*70}")
    print(f"\n  {'Metric':<30} {'GSM':>8} {'Synth':>8} {'All':>8}")
    print(f"  {'-'*56}")
    print(f"  {'Baseline Accuracy':<30} {m_gsm.get('baseline_accuracy','?'):>7}% {m_syn.get('baseline_accuracy','?'):>7}% {m_all.get('baseline_accuracy','?'):>7}%")
    print(f"  {'LGP Accuracy':<30} {m_gsm.get('lgp_accuracy','?'):>7}% {m_syn.get('lgp_accuracy','?'):>7}% {m_all.get('lgp_accuracy','?'):>7}%")
    print(f"  {'Drift Detection Rate':<30} {m_gsm.get('drift_detection_rate','?'):>7}% {m_syn.get('drift_detection_rate','?'):>7}% {m_all.get('drift_detection_rate','?'):>7}%")
    print(f"  {'Correction Success Rate':<30} {m_gsm.get('correction_success_rate','?'):>7}% {m_syn.get('correction_success_rate','?'):>7}% {m_all.get('correction_success_rate','?'):>7}%")
    print(f"  {'Improvements':<30} {m_gsm.get('improvements',0):>8} {m_syn.get('improvements',0):>8} {m_all.get('improvements',0):>8}")
    print(f"  {'Regressions':<30} {m_gsm.get('regressions',0):>8} {m_syn.get('regressions',0):>8} {m_all.get('regressions',0):>8}")
    print(f"\n  Completed in {elapsed:.1f}s")
    print(f"  Results: {RESULTS_DIR}/")
    print(f"{'='*70}\n")

    # Print markdown report to console too
    print(md_text)


"""if __name__ == "__main__":
    main()"""
if __name__ == "__main__":
    llm = get_groq_llm()
    engine = get_reflexion_engine(llm)
    liar = get_liar_agent()

    results = run_liar_test(llm, engine, liar)

    print("\nLIAR TEST RESULTS:\n")
    for r in results:
        print(r)
