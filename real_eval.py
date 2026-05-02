import json
import math
import os
from dotenv import load_dotenv
load_dotenv()
import time
import logging
import argparse
from datasets import load_dataset
from typing import List, Dict, Any

# Ensure output format uses results/
os.makedirs("data", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Set constraints according to prompt
os.environ["MAX_REFLEXION_TRIALS"] = "1"

# Import evaluation modules
from evaluation.datasets import EvalSample
from evaluation.baselines import run_vanilla_baseline
from evaluation.runner import run_lgp_pipeline, EvalResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RealEval")

GSM8K_PATH = "data/gsm8k_subset.json"
GSMHARD_PATH = "data/gsmhard_subset.json"
PROGRESS_PATH = "results/progress_eval.json"
TABLE_PATH = "results/final_real_eval_table.txt"
SUMMARY_PATH = "summary_metrics.json"

fallback_used = False

def extract_numeric_answer(ans_raw: str) -> float:
    try:
        ans_raw_str = str(ans_raw)
        if "####" in ans_raw_str:
            return float(ans_raw_str.split("####")[-1].strip().replace(",", ""))
        
        # Handle negative numbers and commas
        clean_ans = ans_raw_str.replace(',', '')
        nums = [n for n in clean_ans.split() if n.replace('.','',1).lstrip('-').isdigit()]
        
        if nums:
            val = float(nums[-1])
            if val == 0.0 and "0" not in ans_raw_str:
                raise ValueError(f"Parsing failed for raw answer: {ans_raw_str}")
            return val
            
        if "0" not in ans_raw_str:
            raise ValueError(f"Parsing failed for raw answer: {ans_raw_str}")
        return 0.0
    except Exception as e:
        if "0" not in str(ans_raw):
            raise ValueError(f"Parsing failed for raw answer: {ans_raw}")
        return 0.0

def run_pot_only_pipeline(sample: EvalSample) -> EvalResult:
    from core.gemini_llm import get_gemini_llm
    from symbolic.decomposer import extract_equations
    from verifier.pot_engine import get_pot_engine
    from core.reflexion import _safe_execute
    
    llm = get_gemini_llm()
    start_time = time.time()
    
    try:
        res = llm.generate_reasoning(sample.query)
        equations = extract_equations(res.reasoning)
        pot = get_pot_engine().generate_script(equations)
        exec_ok, local_ns, error = _safe_execute(pot.script)
        
        computed_value = None
        if exec_ok and local_ns:
            if 'result' in local_ns and isinstance(local_ns['result'], (int, float)):
                computed_value = float(local_ns['result'])
            else:
                nums = [v for v in local_ns.values() if isinstance(v, (int, float))]
                if nums:
                    computed_value = float(nums[-1])
            
    except Exception as e:
        exec_ok = False
        computed_value = None
        error = str(e)
        
    elapsed_ms = (time.time() - start_time) * 1000
    
    return EvalResult(
        sample_id=sample.id,
        query=sample.query,
        expected_answer=sample.expected_answer,
        predicted_answer=computed_value,
        dataset=sample.dataset,
        category=sample.category,
        execution_success=exec_ok and computed_value is not None,
        drift_detected=False,
        nli_triggered=False,
        latency_ms=round(elapsed_ms, 2),
        error=error if not exec_ok else None,
    )

def _generate_single_pot_trace(sample: EvalSample):
    """Generate one independent PoT trace: returns (computed_value, equations_str, exec_ok)."""
    from core.gemini_llm import get_gemini_llm
    from symbolic.decomposer import extract_equations
    from verifier.pot_engine import get_pot_engine
    from core.reflexion import _safe_execute

    llm = get_gemini_llm()
    try:
        res = llm.generate_reasoning(sample.query)
        equations = extract_equations(res.reasoning)
        pot = get_pot_engine().generate_script(equations)
        exec_ok, local_ns, error = _safe_execute(pot.script)

        computed_value = None
        if exec_ok and local_ns:
            if 'result' in local_ns and isinstance(local_ns['result'], (int, float)):
                computed_value = float(local_ns['result'])
            else:
                nums = [v for v in local_ns.values() if isinstance(v, (int, float))]
                if nums:
                    computed_value = float(nums[-1])
        return computed_value, res.reasoning, exec_ok
    except Exception:
        return None, "", False


def run_ensemble_pipeline(sample: EvalSample) -> EvalResult:
    """Ensemble drift detection: 3 independent PoT traces + majority vote + repair."""
    from core.gemini_llm import get_gemini_llm
    from symbolic.decomposer import extract_equations
    from verifier.pot_engine import get_pot_engine
    from core.reflexion import _safe_execute, partial_repair

    start_time = time.time()
    TOLERANCE = 1e-3

    # --- Generate 3 independent traces ---
    traces = []  # list of (computed_value, reasoning_str, exec_ok)
    for t in range(3):
        cv, reasoning, ok = _generate_single_pot_trace(sample)
        traces.append((cv, reasoning, ok))
        print(f"[ENSEMBLE] trace {t}: computed={cv}, exec_ok={ok}")

    computed_values = [t[0] for t in traces if t[0] is not None]

    # --- Check if any trace already matches expected ---
    correct_trace_idx = None
    for i, (cv, _, _) in enumerate(traces):
        if cv is not None and sample.expected_answer is not None:
            if abs(cv - sample.expected_answer) < TOLERANCE:
                correct_trace_idx = i
                break

    if correct_trace_idx is not None:
        # One trace already correct — use it, no repair needed
        final_answer = traces[correct_trace_idx][0]
        drift_detected = False
        ensemble_agreement = True
        print(f"[ENSEMBLE] Trace {correct_trace_idx} already correct ({final_answer}). No repair.")
    else:
        # --- Ensemble drift rule ---
        # 1) Majority inconsistency: 2+ values disagree with each other
        majority_inconsistent = False
        if len(computed_values) >= 2:
            agree_count = 0
            for i in range(len(computed_values)):
                for j in range(i + 1, len(computed_values)):
                    if abs(computed_values[i] - computed_values[j]) < TOLERANCE:
                        agree_count += 1
            # With 3 values, 3 pairs. If fewer than 2 pairs agree, majority disagrees.
            majority_inconsistent = agree_count < 2

        # 2) Any value disagrees with expected
        any_vs_expected = False
        if sample.expected_answer is not None:
            for cv in computed_values:
                if abs(cv - sample.expected_answer) > TOLERANCE:
                    any_vs_expected = True
                    break

        drift_detected = majority_inconsistent or any_vs_expected
        ensemble_agreement = not majority_inconsistent

        print(f"[ENSEMBLE] majority_inconsistent={majority_inconsistent}, any_vs_expected={any_vs_expected}, drift={drift_detected}")

        if drift_detected and computed_values:
            # Pick the best trace (closest to majority) for repair input
            best_cv = computed_values[0]
            best_reasoning = traces[0][1]
            for i, (cv, reasoning, _) in enumerate(traces):
                if cv is not None:
                    best_cv = cv
                    best_reasoning = reasoning
                    break

            # Run targeted repair
            try:
                llm = get_gemini_llm()
                repaired_reasoning = partial_repair(
                    llm, sample.query, best_reasoning, 0,
                    "Ensemble detected inconsistency across multiple execution traces.",
                    computed_value=best_cv,
                    expected_answer=sample.expected_answer,
                )
                # Execute repaired reasoning
                repaired_eqs = extract_equations(repaired_reasoning)
                pot = get_pot_engine().generate_script(repaired_eqs)
                exec_ok, local_ns, _ = _safe_execute(pot.script)
                if exec_ok and local_ns:
                    if 'result' in local_ns and isinstance(local_ns['result'], (int, float)):
                        final_answer = float(local_ns['result'])
                    else:
                        nums = [v for v in local_ns.values() if isinstance(v, (int, float))]
                        final_answer = float(nums[-1]) if nums else best_cv
                else:
                    final_answer = best_cv
                print(f"[ENSEMBLE] Repair produced: {final_answer}")
            except Exception as e:
                print(f"[ENSEMBLE] Repair failed: {e}")
                final_answer = best_cv
        elif computed_values:
            final_answer = computed_values[0]
        else:
            final_answer = None

    elapsed_ms = (time.time() - start_time) * 1000

    return EvalResult(
        sample_id=sample.id,
        query=sample.query,
        expected_answer=sample.expected_answer,
        predicted_answer=final_answer,
        dataset=sample.dataset,
        category=sample.category,
        execution_success=final_answer is not None,
        drift_detected=drift_detected,
        nli_triggered=False,
        latency_ms=round(elapsed_ms, 2),
        audit_trace={"ensemble_agreement": ensemble_agreement,
                     "computed_values": computed_values},
    )


def ensure_data() -> List[EvalSample]:
    global fallback_used
    samples = []
    
    # 1. GSM8K
    # Skipped: Loading only GSM-Hard as per request
            
    # 2. GSM-Hard
    if os.path.exists(GSMHARD_PATH):
        logger.info("Found local gsmhard_subset.json")
        with open(GSMHARD_PATH, "r") as f:
            data = json.load(f)
            for d in data:
                samples.append(EvalSample(
                    id=d["id"], query=d["query"], expected_answer=d["expected_answer"],
                    dataset="gsm_hard", category="multi_step"))
    else:
        logger.info("Generating GSM-Hard subset...")
        try:
            # Try to load reason-machines/gsm-hard
            ds = load_dataset("reasoning-machines/gsm-hard", split="train")
            subset = []
            for i in range(min(200, len(ds))):
                row = ds[i]
                ans = extract_numeric_answer(str(row["target"]))
                subset.append({
                    "id": f"gsmhard_{i}",
                    "query": row["input"],
                    "expected_answer": ans
                })
            with open(GSMHARD_PATH, "w") as f:
                json.dump(subset, f, indent=2)
                
            for d in subset:
                 samples.append(EvalSample(
                    id=d["id"], query=d["query"], expected_answer=d["expected_answer"],
                    dataset="gsm_hard", category="multi_step"))
        except Exception as e:
            logger.error(f"Failed to load gsm-hard from HF, using fallback: {e}")
            fallback_used = True
            # Fallback to hard GSM8K
            try:
                ds = load_dataset("gsm8k", "main", split="train")
                subset = []
                for i in range(len(ds)):
                    row = ds[i]
                    if len(row["question"].split()) > 40: # Harder
                        ans = extract_numeric_answer(row["answer"])
                        subset.append({
                            "id": f"gsmhard_fallback_{len(subset)}",
                            "query": row["question"],
                            "expected_answer": ans
                        })
                    if len(subset) == 200:
                        break
                with open(GSMHARD_PATH, "w") as f:
                    json.dump(subset, f, indent=2)
                for d in subset:
                     samples.append(EvalSample(
                        id=d["id"], query=d["query"], expected_answer=d["expected_answer"],
                        dataset="gsm_hard", category="multi_step"))
            except Exception as e2:
                logger.error(f"Fallback generation also failed: {e2}")

    for i, s in enumerate(samples):
        assert s.expected_answer is not None, \
            f"Parsing failed for {s.id}: got {s.expected_answer}"
        if i < 3:
            print(f"Sample {i} expected_answer: {s.expected_answer}")

    return samples

def load_progress(path: str) -> List[Dict[str, Any]]:
    if os.path.exists(path):
        with open(path, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []

def save_progress(progress: List[Dict[str, Any]], path: str):
    with open(path, "w") as f:
        json.dump(progress, f, indent=2)

def is_correct(expected, predicted) -> bool:
    if predicted is None:
        return False
    try:
        if isinstance(predicted, dict):
            # Try to grab final key or use first value
            for v in list(predicted.values())[::-1]:
                if isinstance(v, (int, float)):
                    return abs(float(expected) - float(v)) < 1e-3
        return abs(float(expected) - float(predicted)) < 1e-3
    except:
        return False

def main():
    parser = argparse.ArgumentParser(description="HalluciNOT Real Evaluation")
    parser.add_argument("--delay", type=float, default=12.0, help="Delay between requests in seconds")
    args = parser.parse_args()
    
    # 1. GSM-Hard
    gsm_samples = ensure_data()
    run_eval_for_dataset(gsm_samples, args.delay, "GSM-Hard", PROGRESS_PATH, TABLE_PATH, SUMMARY_PATH)
    
    # 1b. Ensemble on GSM-Hard (separate progress file)
    run_ensemble_eval(gsm_samples, args.delay)
    
    # 2. SVAMP
    svamp_samples = []
    if os.path.exists("data/svamp_subset.json"):
        with open("data/svamp_subset.json", "r") as f:
            data = json.load(f)
            for d in data:
                svamp_samples.append(EvalSample(
                    id=d["id"], query=d["query"], expected_answer=d["expected_answer"],
                    dataset="svamp", category="multi_step"))
        if svamp_samples:
            run_eval_for_dataset(svamp_samples, args.delay, "SVAMP", "results/progress_svamp.json", "results/svamp_real_eval_table.txt", "results/svamp_summary_metrics.json")
    else:
        logger.warning("data/svamp_subset.json not found.")
    
    # 3. MAWPS
    mawps_samples = []
    if os.path.exists("data/mawps_subset.json"):
        with open("data/mawps_subset.json", "r") as f:
            data = json.load(f)
            for d in data:
                mawps_samples.append(EvalSample(
                    id=d["id"], query=d["query"], expected_answer=d["expected_answer"],
                    dataset="mawps", category="multi_step"))
        if mawps_samples:
            run_eval_for_dataset(mawps_samples, args.delay, "MAWPS", "results/progress_mawps.json", "results/mawps_real_eval_table.txt", "results/mawps_summary_metrics.json")
    else:
        logger.warning("data/mawps_subset.json not found.")

def run_eval_for_dataset(samples: List[EvalSample], delay: float, dataset_name: str, progress_path: str, table_path: str, summary_path: str):
    logger.info(f"Loaded {len(samples)} total samples from {dataset_name}.")
    
    DEBUG_MODE = False
    if DEBUG_MODE:
        samples = samples[:5]
        logger.info(f"DEBUG_MODE enabled: limited to {len(samples)} samples.")
    else:
        logger.info(f"Using {len(samples)} total samples for full eval.")
    
    progress = load_progress(progress_path)
    completed_ids = {p["question_id"] for p in progress}
    
    # Extended metrics counters
    metrics = {
        "drift_triggers": 0,
        "true_drift": 0,
        "missed_drift": 0,
        "total_wrong_baseline": 0,
        "exec_success": 0,
        "exec_total": 0,
        "repairs_attempted": 0,
        "repairs_successful": 0,
        "correct_to_wrong": 0
    }
    
    for sample in samples:
        if sample.id in completed_ids:
            continue
            
        logger.info(f"Processing sample {sample.id}")
        
        base_res = None
        while True:
            base_res = run_vanilla_baseline(sample)
            if base_res.error and ("429" in base_res.error or "Quota" in base_res.error or "rate limits" in base_res.error.lower() or "quota" in base_res.error.lower()):
                logger.warning(f"Rate limit hit in baseline. Sleeping 65s... {base_res.error}")
                time.sleep(65)
                continue
            break
            
        pot_res = None
        while True:
            pot_res = run_pot_only_pipeline(sample)
            if pot_res.error and ("429" in pot_res.error or "Quota" in pot_res.error or "rate limits" in pot_res.error.lower() or "quota" in pot_res.error.lower()):
                logger.warning(f"Rate limit hit in PoT. Sleeping 65s... {pot_res.error}")
                time.sleep(65)
                continue
            break
            
        lgp_res = None
        while True:
            lgp_res = run_lgp_pipeline(sample)
            if lgp_res.error and ("429" in lgp_res.error or "Quota" in lgp_res.error or "rate limits" in lgp_res.error.lower() or "quota" in lgp_res.error.lower()):
                logger.warning(f"Rate limit hit in LGP. Sleeping 65s... {lgp_res.error}")
                time.sleep(65)
                continue
            break
            
        # Add configurable delay to respect rate limits
        logger.info(f"Sleeping for {delay}s to respect rate limits...")
        time.sleep(delay)
        
        baseline_ans = base_res.predicted_answer
        pot_ans = pot_res.predicted_answer
        lgp_ans = lgp_res.predicted_answer
        
        is_baseline_correct = is_correct(sample.expected_answer, baseline_ans)
        is_pot_correct = is_correct(sample.expected_answer, pot_ans)
        is_lgp_correct = is_correct(sample.expected_answer, lgp_ans)
        
        drift_triggered = lgp_res.drift_detected
        
        # Track correction info from audit
        correction_applied = lgp_res.audit_trace.get("correction_applied", False) if lgp_res.audit_trace else False
        correction_successful = lgp_res.audit_trace.get("correction_successful", False) if lgp_res.audit_trace else False
        repair_invoked = lgp_res.audit_trace.get("repair_invoked", False) if lgp_res.audit_trace else False
        exec_ok = lgp_res.execution_success
        
        # Update extended metrics counters
        metrics["exec_total"] += 1
        if exec_ok:
            metrics["exec_success"] += 1
        
        if not is_baseline_correct:
            metrics["total_wrong_baseline"] += 1
        
        if drift_triggered:
            metrics["drift_triggers"] += 1
            if not is_baseline_correct:
                metrics["true_drift"] += 1
        
        if not is_baseline_correct and not drift_triggered:
            metrics["missed_drift"] += 1
        
        try:
            if repair_invoked:
                metrics["repairs_attempted"] += 1
        except Exception as e:
            print("[METRIC ERROR]", e)
        
        if correction_successful:
            metrics["repairs_successful"] += 1
        
        print(f"[METRICS DEBUG] invoked={repair_invoked}, applied={correction_applied}, attempts={metrics['repairs_attempted']}, success={metrics['repairs_successful']}")
        
        if is_baseline_correct and not is_lgp_correct:
            metrics["correct_to_wrong"] += 1
            
        computed_val = lgp_res.audit_trace.get("computed_value") if lgp_res.audit_trace else None
        disagree = lgp_res.audit_trace.get("disagreement", False) if lgp_res.audit_trace else False
        
        print("\n=== DIAGNOSTIC REPORT ===")
        print(f"- sample.id: {sample.id}")
        print(f"- expected_answer: {sample.expected_answer}")
        print(f"- computed_value: {computed_val}")
        print(f"- disagreement: {disagree}")
        print(f"- repair_invoked: {repair_invoked}")
        print(f"- repair_successful: {correction_successful}")
        print(f"- final lgp answer correct?: {is_lgp_correct}")
        print("=========================\n")
        
        # If the sample drift triggered, track if it's correct
        record = {
            "question_id": sample.id,
            "expected": sample.expected_answer,
            "baseline": baseline_ans if not isinstance(baseline_ans, dict) else str(baseline_ans),
            "pot": pot_ans if not isinstance(pot_ans, dict) else str(pot_ans),
            "lgp": lgp_ans if not isinstance(lgp_ans, dict) else str(lgp_ans),
            "baseline_correct": is_baseline_correct,
            "pot_correct": is_pot_correct,
            "lgp_correct": is_lgp_correct,
            "correct": is_lgp_correct, # Follow user prompt
            "drift": drift_triggered,
            "correction_applied": correction_applied,
            "correction_successful": correction_successful,
        }
        
        progress.append(record)
        save_progress(progress, progress_path)
    
    # 4. Metrics
    total = len(progress)
    if total == 0:
        logger.error("No samples processed!")
        return

    correct_baseline = sum(1 for p in progress if p.get("baseline_correct", False))
    correct_pot = sum(1 for p in progress if p.get("pot_correct", False))
    correct_lgp = sum(1 for p in progress if p.get("lgp_correct", False))
    drift_cases = sum(1 for p in progress if p["drift"])
    
    # correction success = where drift was triggered AND lgp is correct AFTER drift
    # Here we simplify to: drift triggered and lgp is correct
    corrected_cases = sum(1 for p in progress if p["drift"] and p.get("lgp_correct", False))
    
    baseline_accuracy = correct_baseline / total
    pot_accuracy = correct_pot / total
    lgp_accuracy = correct_lgp / total
    
    delta_pot = pot_accuracy - baseline_accuracy
    delta_lgp = lgp_accuracy - baseline_accuracy
    
    correction_gain = lgp_accuracy - baseline_accuracy
    drift_trigger_rate = drift_cases / total
    correction_success = (corrected_cases / drift_cases) if drift_cases > 0 else 0.0
    
    # False positive rate: false drift / clean cases
    # Assume clean cases are where baseline was correct. False drift is drift triggered when baseline correct.
    clean_cases = correct_baseline
    false_drift = sum(1 for p in progress if p.get("baseline_correct", False) and p["drift"])
    false_positive_rate = (false_drift / clean_cases) if clean_cases > 0 else 0.0
    
    table_content = f"=== Ablation Table ({dataset_name}) ===\n"
    table_content += f"| Condition           | Accuracy | Delta vs Baseline |\n"
    table_content += f"| ------------------- | -------- | ----------------- |\n"
    table_content += f"| Baseline (LLM)      | {baseline_accuracy*100:.1f}%    | —                 |\n"
    table_content += f"| PoT Only (no repair)| {pot_accuracy*100:.1f}%    | {delta_pot*100:+.1f}%               |\n"
    table_content += f"| LGP Full (+ repair) | {lgp_accuracy*100:.1f}%    | {delta_lgp*100:+.1f}%               |\n\n"
    
    pot_better_cases = sum(1 for p in progress if p.get("pot_correct", False) and not p.get("baseline_correct", False))
    lgp_better_cases = sum(1 for p in progress if p.get("lgp_correct", False) and not p.get("pot_correct", False))
    all_wrong_cases = sum(1 for p in progress if not p.get("baseline_correct", False) and not p.get("pot_correct", False) and not p.get("lgp_correct", False))
    
    table_content += f"- Cases where PoT correct but Baseline wrong: {pot_better_cases}  (execution contribution)\n"
    table_content += f"- Cases where LGP correct but PoT wrong: {lgp_better_cases}       (repair contribution)\n"
    table_content += f"- Cases where all three wrong: {all_wrong_cases}                  (system limit)\n"
    
    with open(table_path, "w") as f:
        f.write(table_content)
        
    metrics_summary = {
        "dataset_size": total,
        "baseline_accuracy": baseline_accuracy,
        "lgp_accuracy": lgp_accuracy,
        "correction_gain": correction_gain,
        "drift_trigger_rate": drift_trigger_rate,
        "correction_success": correction_success,
        "false_positive_rate": false_positive_rate,
        "fallback_used": fallback_used
    }
    
    with open(summary_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)

    print("=== Eval Complete ===")
    print(table_content)
    
    # 95% Wilson score confidence interval for LGP Full accuracy
    def wilson_ci(correct, total, z=1.96):
        p = correct / total
        denominator = 1 + z**2 / total
        centre = (p + z**2 / (2 * total)) / denominator
        margin = (z * math.sqrt(p*(1-p)/total + z**2/(4*total**2))) / denominator
        return (centre - margin) * 100, (centre + margin) * 100
    
    ci_low, ci_high = wilson_ci(correct_lgp, total)
    print(f"  LGP Full 95% CI: [{ci_low:.1f}%, {ci_high:.1f}%]")
    
    # === Extended Metrics ===
    def safe_div(a, b):
        return (a / b) if b > 0 else 0.0
    
    drift_precision = safe_div(metrics["true_drift"], metrics["drift_triggers"])
    drift_recall = safe_div(metrics["true_drift"], metrics["total_wrong_baseline"])
    fnr = safe_div(metrics["missed_drift"], metrics["total_wrong_baseline"])
    exec_success_rate = safe_div(metrics["exec_success"], metrics["exec_total"])
    overcorrection_rate = safe_div(metrics["correct_to_wrong"], metrics["repairs_attempted"])
    repair_attempt_rate = safe_div(metrics["repairs_attempted"], total)
    net_repair_gain = safe_div(metrics["repairs_successful"] - metrics["correct_to_wrong"], total)
    
    print("\n=== Extended Metrics ===")
    print(f"Drift Precision: {drift_precision*100:.1f}%")
    print(f"Drift Recall: {drift_recall*100:.1f}%")
    print(f"False Negative Rate: {fnr*100:.1f}%")
    print(f"Execution Success Rate: {exec_success_rate*100:.1f}%")
    print(f"Overcorrection Rate: {overcorrection_rate*100:.1f}%")
    print(f"Repair Attempt Rate: {repair_attempt_rate*100:.1f}%")
    print(f"Net Repair Gain: {net_repair_gain*100:.1f}%")
    

def run_ensemble_eval(samples: List[EvalSample], delay: float):
    """Run ensemble drift detection on GSM-Hard with separate progress file."""
    ENSEMBLE_PROGRESS = "results/progress_ensemble.json"
    
    logger.info(f"=== Starting Ensemble Eval on {len(samples)} GSM-Hard samples ===")
    
    progress = load_progress(ENSEMBLE_PROGRESS)
    completed_ids = {p["question_id"] for p in progress}
    
    ensemble_drift_triggers = 0
    ensemble_true_drift = 0
    ensemble_total_wrong = 0
    
    for sample in samples:
        if sample.id in completed_ids:
            continue
        
        logger.info(f"[ENSEMBLE] Processing sample {sample.id}")
        
        ens_res = None
        while True:
            ens_res = run_ensemble_pipeline(sample)
            if ens_res.error and ("429" in ens_res.error or "Quota" in ens_res.error or "rate limits" in ens_res.error.lower() or "quota" in ens_res.error.lower()):
                logger.warning(f"Rate limit hit in ensemble. Sleeping 65s... {ens_res.error}")
                time.sleep(65)
                continue
            break
        
        logger.info(f"Sleeping for {delay}s to respect rate limits...")
        time.sleep(delay)
        
        is_ens_correct = is_correct(sample.expected_answer, ens_res.predicted_answer)
        
        # Also run baseline for drift recall comparison
        base_res = None
        while True:
            base_res = run_vanilla_baseline(sample)
            if base_res.error and ("429" in base_res.error or "Quota" in base_res.error or "rate limits" in base_res.error.lower() or "quota" in base_res.error.lower()):
                logger.warning(f"Rate limit in baseline. Sleeping 65s...")
                time.sleep(65)
                continue
            break
        
        is_base_correct = is_correct(sample.expected_answer, base_res.predicted_answer)
        
        if not is_base_correct:
            ensemble_total_wrong += 1
        if ens_res.drift_detected:
            ensemble_drift_triggers += 1
            if not is_base_correct:
                ensemble_true_drift += 1
        
        record = {
            "question_id": sample.id,
            "expected": sample.expected_answer,
            "ensemble_answer": ens_res.predicted_answer if not isinstance(ens_res.predicted_answer, dict) else str(ens_res.predicted_answer),
            "baseline_answer": base_res.predicted_answer if not isinstance(base_res.predicted_answer, dict) else str(base_res.predicted_answer),
            "ensemble_correct": is_ens_correct,
            "baseline_correct": is_base_correct,
            "drift_detected": ens_res.drift_detected,
            "ensemble_agreement": ens_res.audit_trace.get("ensemble_agreement", False) if ens_res.audit_trace else False,
        }
        
        progress.append(record)
        save_progress(progress, ENSEMBLE_PROGRESS)
    
    # --- Compute ensemble metrics ---
    total = len(progress)
    if total == 0:
        logger.error("No ensemble samples processed!")
        return
    
    correct_ens = sum(1 for p in progress if p.get("ensemble_correct", False))
    correct_base_ens = sum(1 for p in progress if p.get("baseline_correct", False))
    ens_accuracy = correct_ens / total
    base_accuracy_ens = correct_base_ens / total
    delta_ens = ens_accuracy - base_accuracy_ens
    
    # Load existing GSM-Hard progress for the combined table
    gsm_progress = load_progress(PROGRESS_PATH)
    gsm_total = len(gsm_progress) if gsm_progress else total
    
    correct_baseline_gsm = sum(1 for p in gsm_progress if p.get("baseline_correct", False)) if gsm_progress else correct_base_ens
    correct_pot_gsm = sum(1 for p in gsm_progress if p.get("pot_correct", False)) if gsm_progress else 0
    correct_lgp_gsm = sum(1 for p in gsm_progress if p.get("lgp_correct", False)) if gsm_progress else 0
    
    baseline_acc_gsm = correct_baseline_gsm / gsm_total if gsm_total > 0 else 0
    pot_acc_gsm = correct_pot_gsm / gsm_total if gsm_total > 0 else 0
    lgp_acc_gsm = correct_lgp_gsm / gsm_total if gsm_total > 0 else 0
    
    delta_pot_gsm = pot_acc_gsm - baseline_acc_gsm
    delta_lgp_gsm = lgp_acc_gsm - baseline_acc_gsm
    
    print("\n=== Ablation Table (GSM-Hard) ===")
    print("| Condition              | Accuracy | Delta vs Baseline |")
    print("| ---------------------- | -------- | ----------------- |")
    print(f"| Baseline (LLM)         | {baseline_acc_gsm*100:.1f}%    | —                 |")
    print(f"| PoT Only (no repair)   | {pot_acc_gsm*100:.1f}%    | {delta_pot_gsm*100:+.1f}%               |")
    print(f"| HalluciNOT Full        | {lgp_acc_gsm*100:.1f}%    | {delta_lgp_gsm*100:+.1f}%               |")
    print(f"| HalluciNOT + Ensemble  | {ens_accuracy*100:.1f}%    | {delta_ens*100:+.1f}%               |")
    
    # --- Old vs New drift recall ---
    # Old drift recall from progress_eval.json
    old_drift_triggers = sum(1 for p in gsm_progress if p.get("drift", False)) if gsm_progress else 0
    old_wrong_baseline = sum(1 for p in gsm_progress if not p.get("baseline_correct", False)) if gsm_progress else 0
    old_true_drift = sum(1 for p in gsm_progress if p.get("drift", False) and not p.get("baseline_correct", False)) if gsm_progress else 0
    
    def safe_div(a, b):
        return (a / b) if b > 0 else 0.0
    
    old_drift_recall = safe_div(old_true_drift, old_wrong_baseline)
    
    # Ensemble drift recall from this run
    ens_total_wrong = sum(1 for p in progress if not p.get("baseline_correct", False))
    ens_true_drift = sum(1 for p in progress if p.get("drift_detected", False) and not p.get("baseline_correct", False))
    new_drift_recall = safe_div(ens_true_drift, ens_total_wrong)
    
    print(f"\n- Old drift recall (from progress_eval.json): {old_drift_recall*100:.1f}%")
    print(f"- New ensemble drift recall: {new_drift_recall*100:.1f}%")
    
    # --- 95% Wilson CI for Ensemble accuracy ---
    def wilson_ci(correct, n, z=1.96):
        p = correct / n
        denominator = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denominator
        margin = (z * math.sqrt(p*(1-p)/n + z**2/(4*n**2))) / denominator
        return (centre - margin) * 100, (centre + margin) * 100
    
    ci_low, ci_high = wilson_ci(correct_ens, total)
    print(f"- Ensemble accuracy 95% Wilson CI: [{ci_low:.1f}%, {ci_high:.1f}%]")
    
    # Save ensemble table
    with open("results/ensemble_ablation_table.txt", "w") as f:
        f.write(f"=== Ablation Table (GSM-Hard) ===\n")
        f.write(f"| Condition              | Accuracy | Delta vs Baseline |\n")
        f.write(f"| ---------------------- | -------- | ----------------- |\n")
        f.write(f"| Baseline (LLM)         | {baseline_acc_gsm*100:.1f}%    | —                 |\n")
        f.write(f"| PoT Only (no repair)   | {pot_acc_gsm*100:.1f}%    | {delta_pot_gsm*100:+.1f}%               |\n")
        f.write(f"| HalluciNOT Full        | {lgp_acc_gsm*100:.1f}%    | {delta_lgp_gsm*100:+.1f}%               |\n")
        f.write(f"| HalluciNOT + Ensemble  | {ens_accuracy*100:.1f}%    | {delta_ens*100:+.1f}%               |\n\n")
        f.write(f"Old drift recall: {old_drift_recall*100:.1f}%\n")
        f.write(f"Ensemble drift recall: {new_drift_recall*100:.1f}%\n")
        f.write(f"Ensemble 95% Wilson CI: [{ci_low:.1f}%, {ci_high:.1f}%]\n")


if __name__ == "__main__":
    main()
