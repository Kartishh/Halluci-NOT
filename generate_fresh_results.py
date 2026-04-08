import os
import sys
import time
import json
import csv
from dataclasses import asdict

# Import necessary components from the existing codebase
from lgp_eval import (
    get_groq_llm, 
    get_reflexion_engine,
    run_baseline,
    run_lgp,
    is_correct,
    GSM_QUERIES,
    SYNTHETIC_DRIFT_QUERIES,
    EvalResult,
    RESULTS_DIR
)

def main():
    print("\n" + "=" * 70)
    print("  HalluciNOT (LGP) — Generating Fresh Evaluation Results")
    print("=" * 70)

    t0 = time.time()

    llm = get_groq_llm()
    engine = get_reflexion_engine(llm)

    # Combine datasets
    all_queries = []
    
    # 1. Load GSM dataset
    gsm_data = GSM_QUERIES
    for q in gsm_data:
        all_queries.append({**q, "dataset": "gsm"})
            
    # 2. Load Drift Stress Test dataset
    with open("data/drift_stress_test.json", "r") as f:
        drift_data = json.load(f)
        for q in drift_data:
            all_queries.append({**q, "dataset": "drift"})

    print(f"\n  Total queries: {len(all_queries)} (GSM: {len(gsm_data)}, Drift: {len(drift_data)})\n")

    results = []
    api_errors = 0

    for i, item in enumerate(all_queries):
        qid = item["id"]
        dataset_label = item["dataset"]
        query = item.get("query", item.get("question", ""))
        expected = item.get("expected", item.get("answer"))
        logic_type = item.get("logic_type", "standard")

        sys.stdout.write(f"\r  [{i+1}/{len(all_queries)}] {qid}...")
        sys.stdout.flush()

        # Rate limiting
        if i > 0:
            time.sleep(2)

        try:
            # Baseline
            b_answer, b_reasoning = run_baseline(llm, query)
            b_correct = is_correct(b_answer, expected)

            time.sleep(1)  # Rate limit gap

            # LGP
            # lgp_out = (l_answer, l_reasoning, l_drift, l_correction, l_corr_success, l_iters, l_drift_details)
            lgp_out = run_lgp(engine, query)
            l_answer = lgp_out[0]
            l_reasoning = lgp_out[1]
            l_drift = lgp_out[2]
            l_correction = lgp_out[3]
            l_corr_success = lgp_out[4]
            l_iters = lgp_out[5]
            l_drift_details = lgp_out[6]
            l_correct = is_correct(l_answer, expected)

            results.append({
                "case_id": qid,
                "dataset": dataset_label,
                "baseline_correct": b_correct,
                "lgp_correct": l_correct,
                "drift_detected": l_drift,
                "correction_applied": l_correction,
                "baseline_answer": b_answer,
                "lgp_answer": l_answer,
                "query": query[:200],
                "logic_type": logic_type,
                "expected": expected,
                "baseline_reasoning": b_reasoning[:300],
                "lgp_reasoning": l_reasoning[:300],
                "correction_successful": l_corr_success,
                "reflexion_iterations": l_iters,
                "drift_details": l_drift_details[:200],
            })

            status = "✓" if l_correct else "✗"
            b_status = "✓" if b_correct else "✗"
            print(f"\r  [{i+1}/{len(all_queries)}] {qid}  Base={b_answer}({b_status}) LGP={l_answer}({status})")

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

    # Compute Metrics
    n = len(results)
    gsm_r = [r for r in results if r["dataset"] == "gsm"]
    drift_r = [r for r in results if r["dataset"] == "drift"]

    def _metrics(res, label):
        if not res:
            return {"label": label, "n": 0}
        total = len(res)
        bc = sum(1 for r in res if r["baseline_correct"])
        lc = sum(1 for r in res if r["lgp_correct"])
        dd = sum(1 for r in res if r["drift_detected"])
        ca = sum(1 for r in res if r["correction_applied"])
        cs = sum(1 for r in res if r.get("correction_successful", False))
        imp = sum(1 for r in res if r["lgp_correct"] and not r["baseline_correct"])
        reg = sum(1 for r in res if r["baseline_correct"] and not r["lgp_correct"])
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
    m_drift = _metrics(drift_r, "Drift Stress Test")
    m_all = _metrics(results, "Overall")

    summary = {"gsm": m_gsm, "drift": m_drift, "overall": m_all, "elapsed_seconds": round(elapsed, 1)}

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, "lgp_evaluation_results.csv")
    with open(csv_path, "w", newline="") as f:
        if results:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            for r in results:
                w.writerow(r)
            
    # Save JSON summary
    json_path = os.path.join(RESULTS_DIR, "lgp_summary_metrics.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nFresh evaluation results saved successfully")
    print(f"Results: {RESULTS_DIR}/")
    print(f"  ✓ {csv_path}")
    print(f"  ✓ {json_path}")
    print(f"\nSample Row:")
    print(json.dumps(results[-1], indent=2))

if __name__ == "__main__":
    main()
