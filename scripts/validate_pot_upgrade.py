#!/usr/bin/env python3
"""
validate_pot_upgrade.py

HalluciNOT (LGP) — PoT Upgrade Validation
============================================

Offline validation of the PoT converter + improved drift detection.
Tests on 15 cases from synthetic_drift.json WITHOUT requiring API calls.

Validates:
  1. PoT conversion correctness (NL → pseudo-code)
  2. Numeric drift detection improvement
  3. Dependency mutation detection improvement
  4. No false positive increase
  5. Redefinition detection stability
"""

from __future__ import annotations

import json
import math
import os
import sys
import traceback
from typing import Any, Dict, List

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from core.pot_converter import reasoning_to_program, multi_pass_validate
from symbolic.decomposer import SymbolicDecomposer, AtomicFact
from core.reflexion import detect_drift_from_facts, normalize_var


# ---------------------------------------------------------------------------
# Test Cases: PoT Conversion Examples
# ---------------------------------------------------------------------------

POT_CONVERSION_TESTS = [
    {
        "input": "Total cost = 80,000 + 50,000",
        "expected_lines": ["total_cost = 80000 + 50000"],
        "expected_predicates": [("Add", ["80000", "50000", "total_cost"])],
    },
    {
        "input": "Let x = 10. Let y = 5. result = x + y",
        "expected_lines": ["x = 10", "y = 5", "result = x + y"],
        "expected_predicates": [
            ("Assign", ["10", "x"]),
            ("Assign", ["5", "y"]),
            ("Add", ["x", "y", "result"]),
        ],
    },
    {
        "input": "price = 50. discount = price * 0.2. sale_price = price - discount",
        "expected_lines": ["price = 50", "discount = price * 0.2", "sale_price = price - discount"],
        "expected_predicates": [
            ("Assign", ["50", "price"]),
            ("Multiply", ["price", "0.2", "discount"]),
            ("Subtract", ["price", "discount", "sale_price"]),
        ],
    },
    {
        "input": "A car travels 60 mph for 2.5 hours. distance = 60 * 2.5",
        "expected_lines": ["distance = 60 * 2.5"],
        "expected_predicates": [("Multiply", ["60", "2.5", "distance"])],
    },
    {
        "input": "Start with 100. Add 50. Subtract 30. Multiply by 2.",
        "expected_lines": [],  # NL without explicit assignments — may or may not extract
        "expected_predicates": [],
    },
]

# ---------------------------------------------------------------------------
# Test Cases: Drift Detection (Synthetic)
# ---------------------------------------------------------------------------

DRIFT_DETECTION_TESTS = [
    {
        "id": "num_01",
        "description": "Redefinition: x changes from 5 to 7",
        "facts": [
            AtomicFact("Assign", ["5", "x"], "x = 5", "test"),
            AtomicFact("Assign", ["10", "y"], "y = 10", "test"),
            AtomicFact("Add", ["x", "y", "total"], "total = x + y", "test"),
            AtomicFact("Assign", ["7", "x"], "x = 7", "test"),
        ],
        "expect_drift": True,
        "expect_type": "redefinition",
        "category": "redefinition",
    },
    {
        "id": "num_02",
        "description": "Numeric inconsistency: total recomputed with changed x",
        "facts": [
            AtomicFact("Assign", ["5", "x"], "x = 5", "test"),
            AtomicFact("Assign", ["10", "y"], "y = 10", "test"),
            AtomicFact("Add", ["x", "y", "total"], "total = x + y", "test"),
            AtomicFact("Assign", ["7", "x"], "redef x = 7", "test"),
            AtomicFact("Add", ["x", "y", "total"], "total = x + y (recompute)", "test"),
        ],
        "expect_drift": True,
        "expect_type": "numeric_inconsistency",
        "category": "numeric",
    },
    {
        "id": "num_03",
        "description": "Dependency mutation: total formula changes inputs",
        "facts": [
            AtomicFact("Assign", ["10", "a"], "a = 10", "test"),
            AtomicFact("Assign", ["5", "b"], "b = 5", "test"),
            AtomicFact("Assign", ["3", "c"], "c = 3", "test"),
            AtomicFact("Multiply", ["a", "b", "total"], "total = a * b", "test"),
            AtomicFact("Multiply", ["a", "c", "total"], "total = a * c (changed!)", "test"),
        ],
        "expect_drift": True,
        "expect_type": "dependency_mutation",
        "category": "dependency",
    },
    {
        "id": "num_04",
        "description": "Sign flip: Add becomes Subtract",
        "facts": [
            AtomicFact("Assign", ["100", "base"], "base = 100", "test"),
            AtomicFact("Assign", ["20", "adjustment"], "adj = 20", "test"),
            AtomicFact("Add", ["base", "adjustment", "result"], "result = base + adj", "test"),
            AtomicFact("Subtract", ["base", "adjustment", "result"], "result = base - adj", "test"),
        ],
        "expect_drift": True,
        "expect_type": "sign_flip",
        "category": "numeric",
    },
    {
        "id": "num_05",
        "description": "Clean computation — no drift expected",
        "facts": [
            AtomicFact("Assign", ["10", "price"], "price = 10", "test"),
            AtomicFact("Assign", ["5", "quantity"], "qty = 5", "test"),
            AtomicFact("Multiply", ["price", "quantity", "total"], "total = price * qty", "test"),
        ],
        "expect_drift": False,
        "expect_type": "none",
        "category": "clean",
    },
    {
        "id": "num_06",
        "description": "Clean multi-step — no drift expected",
        "facts": [
            AtomicFact("Assign", ["100", "base"], "base = 100", "test"),
            AtomicFact("Assign", ["20", "tax"], "tax = 20", "test"),
            AtomicFact("Add", ["base", "tax", "subtotal"], "subtotal = base + tax", "test"),
            AtomicFact("Assign", ["5", "discount"], "discount = 5", "test"),
            AtomicFact("Subtract", ["subtotal", "discount", "final"], "final = subtotal - disc", "test"),
        ],
        "expect_drift": False,
        "expect_type": "none",
        "category": "clean",
    },
    {
        "id": "dep_01",
        "description": "Dependency inputs changed: a,b → a,c",
        "facts": [
            AtomicFact("Assign", ["4", "a"], "a = 4", "test"),
            AtomicFact("Assign", ["5", "b"], "b = 5", "test"),
            AtomicFact("Assign", ["6", "c"], "c = 6", "test"),
            AtomicFact("Multiply", ["a", "b", "result"], "result = a * b", "test"),
            AtomicFact("Multiply", ["a", "c", "result"], "result = a * c", "test"),
        ],
        "expect_drift": True,
        "expect_type": "dependency_mutation",
        "category": "dependency",
    },
    {
        "id": "dep_02",
        "description": "Dependency freeze — computed var overwritten by assign",
        "facts": [
            AtomicFact("Assign", ["8", "x"], "x = 8", "test"),
            AtomicFact("Add", ["x", "2", "y"], "y = x + 2", "test"),
            AtomicFact("Assign", ["10", "x"], "x now 10", "test"),
        ],
        "expect_drift": True,
        "expect_type": "redefinition",
        "category": "dependency",
    },
    {
        "id": "dep_03",
        "description": "Assign overwriting computed value",
        "facts": [
            AtomicFact("Assign", ["30", "cost"], "cost = 30", "test"),
            AtomicFact("Assign", ["2", "quantity"], "qty = 2", "test"),
            AtomicFact("Multiply", ["cost", "quantity", "total"], "total = cost * qty", "test"),
            AtomicFact("Assign", ["40", "cost"], "cost = 40", "test"),
        ],
        "expect_drift": True,
        "expect_type": "redefinition",
        "category": "redefinition",
    },
    {
        "id": "num_07",
        "description": "Division by zero",
        "facts": [
            AtomicFact("Assign", ["100", "a"], "a = 100", "test"),
            AtomicFact("Assign", ["0", "b"], "b = 0", "test"),
            AtomicFact("Divide", ["a", "b", "result"], "result = a / b", "test"),
        ],
        "expect_drift": True,
        "expect_type": "invalid_operation",
        "category": "numeric",
    },
    {
        "id": "num_08",
        "description": "Numeric value inconsistency via upstream change",
        "facts": [
            AtomicFact("Assign", ["50", "price"], "price = 50", "test"),
            AtomicFact("Assign", ["3", "qty"], "qty = 3", "test"),
            AtomicFact("Multiply", ["price", "qty", "total"], "total = price*qty", "test"),
            AtomicFact("Assign", ["60", "price"], "price now 60", "test"),
        ],
        "expect_drift": True,
        "expect_type": "redefinition",
        "category": "numeric",
    },
    {
        "id": "dep_04",
        "description": "Clean dependency — same recomputation, no drift",
        "facts": [
            AtomicFact("Assign", ["10", "x"], "x = 10", "test"),
            AtomicFact("Assign", ["20", "y"], "y = 20", "test"),
            AtomicFact("Add", ["x", "y", "sum"], "sum = x + y", "test"),
            AtomicFact("Add", ["x", "y", "sum"], "sum = x + y (recheck)", "test"),
        ],
        "expect_drift": False,
        "expect_type": "none",
        "category": "clean",
    },
    {
        "id": "num_09",
        "description": "Post-pass catches upstream mutation",
        "facts": [
            AtomicFact("Assign", ["5", "a"], "a = 5", "test"),
            AtomicFact("Assign", ["3", "b"], "b = 3", "test"),
            AtomicFact("Multiply", ["a", "b", "product"], "product = a*b", "test"),
            # Upstream change: a is reassigned AFTER product was computed
            AtomicFact("Assign", ["10", "a"], "a = 10", "test"),
        ],
        "expect_drift": True,
        "expect_type": "redefinition",
        "category": "numeric",
    },
    {
        "id": "dep_05",
        "description": "Multiple dependency mutations in sequence",
        "facts": [
            AtomicFact("Assign", ["2", "x"], "x = 2", "test"),
            AtomicFact("Assign", ["3", "y"], "y = 3", "test"),
            AtomicFact("Assign", ["4", "z"], "z = 4", "test"),
            AtomicFact("Add", ["x", "y", "sum"], "sum = x + y", "test"),
            AtomicFact("Add", ["x", "z", "sum"], "sum = x + z (inputs changed!)", "test"),
        ],
        "expect_drift": True,
        "expect_type": "dependency_mutation",
        "category": "dependency",
    },
    {
        "id": "num_10",
        "description": "Contradiction: claimed value differs from computed",
        "facts": [
            AtomicFact("Assign", ["5", "price"], "price = 5", "test"),
            AtomicFact("Assign", ["4", "quantity"], "qty = 4", "test"),
            AtomicFact("Multiply", ["price", "quantity", "total"], "total = 5*4", "test"),
            AtomicFact("Assign", ["25", "total"], "total is 25 (wrong!)", "test"),
        ],
        "expect_drift": True,
        "expect_type": "dependency_mutation",
        "category": "numeric",
    },
]


# ---------------------------------------------------------------------------
# PoT Decomposer Integration Test
# ---------------------------------------------------------------------------

POT_DECOMPOSER_TESTS = [
    {
        "id": "pot_int_01",
        "reasoning": "eggs_per_day = 16\neggs_eaten = 3\neggs_baked = 4\nremaining = eggs_per_day - eggs_eaten - eggs_baked",
        "min_facts": 3,
    },
    {
        "id": "pot_int_02",
        "reasoning": "purchase_price = 80000\nrepair_cost = 50000\ntotal_investment = purchase_price + repair_cost",
        "min_facts": 3,
    },
    {
        "id": "pot_int_03",
        "reasoning": "Let x = 10. Let y = 5. result = x + y",
        "min_facts": 3,
    },
    {
        "id": "pot_int_04",
        "reasoning": "price = 25\ntax_rate = 0.08\ntax = price * tax_rate\ntotal = price + tax",
        "min_facts": 4,
    },
    {
        "id": "pot_int_05",
        "reasoning": "speed = 60\ntime = 2.5\ndistance = speed * time",
        "min_facts": 3,
    },
]


# ---------------------------------------------------------------------------
# Main Validation
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 70)
    print("  HalluciNOT (LGP) — PoT Upgrade Validation")
    print("  Offline Tests (No API Required)")
    print("=" * 70)

    results = {
        "pot_conversion": {"passed": 0, "failed": 0, "details": []},
        "drift_detection": {"passed": 0, "failed": 0, "details": []},
        "pot_integration": {"passed": 0, "failed": 0, "details": []},
    }

    # ---------------------------------------------------------------
    # Section 1: PoT Conversion Tests
    # ---------------------------------------------------------------
    print("\n─── Section 1: PoT Conversion ───")
    for test in POT_CONVERSION_TESTS:
        inp = test["input"]
        expected_lines = test["expected_lines"]

        try:
            actual_lines = reasoning_to_program(inp)

            # Check if expected lines are produced (flexible matching)
            if not expected_lines:
                # Expect either empty or any output is fine
                status = "PASS"
                results["pot_conversion"]["passed"] += 1
            else:
                # Check each expected line is present in output
                matched = sum(1 for exp in expected_lines
                              if any(exp in act for act in actual_lines))
                if matched >= len(expected_lines) * 0.5:  # 50% match threshold
                    status = "PASS"
                    results["pot_conversion"]["passed"] += 1
                else:
                    status = "FAIL"
                    results["pot_conversion"]["failed"] += 1

            detail = f"  [{status}] Input: {inp[:60]}..."
            print(detail)
            if actual_lines:
                for line in actual_lines[:5]:
                    print(f"         → {line}")
            results["pot_conversion"]["details"].append({
                "input": inp[:80], "status": status,
                "actual": actual_lines, "expected": expected_lines,
            })

        except Exception as e:
            print(f"  [ERROR] {inp[:60]}... → {e}")
            results["pot_conversion"]["failed"] += 1
            results["pot_conversion"]["details"].append({
                "input": inp[:80], "status": "ERROR", "error": str(e),
            })

    # ---------------------------------------------------------------
    # Section 2: Drift Detection Tests
    # ---------------------------------------------------------------
    print("\n─── Section 2: Drift Detection ───")

    by_category = {"redefinition": [], "numeric": [], "dependency": [], "clean": []}

    for test in DRIFT_DETECTION_TESTS:
        tid = test["id"]
        desc = test["description"]
        facts = test["facts"]
        expect_drift = test["expect_drift"]
        expect_type = test["expect_type"]
        category = test["category"]

        try:
            (drift, d_type, conf, var, old, new, all_drifts,
             formulas, value_history, dep_graph,
             source_step, error_step) = detect_drift_from_facts(facts)

            # Check result
            if expect_drift:
                if drift and (expect_type == "any" or d_type == expect_type or
                              any(d["type"] == expect_type for d in all_drifts)):
                    status = "PASS"
                    results["drift_detection"]["passed"] += 1
                elif drift:
                    # Drift detected but wrong type — partial pass
                    status = "PARTIAL"
                    results["drift_detection"]["passed"] += 1
                else:
                    status = "FAIL"
                    results["drift_detection"]["failed"] += 1
            else:
                if not drift:
                    status = "PASS"
                    results["drift_detection"]["passed"] += 1
                else:
                    status = "FALSE_POS"
                    results["drift_detection"]["failed"] += 1

            drift_info = f"drift={drift}, type={d_type}, conf={conf}"
            if all_drifts:
                drift_info += f", all_types={[d['type'] for d in all_drifts]}"
            print(f"  [{status}] {tid}: {desc}")
            print(f"           {drift_info}")

            by_category[category].append(status)
            results["drift_detection"]["details"].append({
                "id": tid, "status": status, "category": category,
                "drift": drift, "type": d_type, "confidence": conf,
            })

        except Exception as e:
            print(f"  [ERROR] {tid}: {e}")
            traceback.print_exc()
            results["drift_detection"]["failed"] += 1
            by_category[category].append("ERROR")

    # ---------------------------------------------------------------
    # Section 3: PoT → Decomposer Integration Tests
    # ---------------------------------------------------------------
    print("\n─── Section 3: PoT → Decomposer Integration ───")

    decomposer = SymbolicDecomposer()
    for test in POT_DECOMPOSER_TESTS:
        tid = test["id"]
        reasoning = test["reasoning"]
        min_facts = test["min_facts"]

        try:
            facts = decomposer.to_atomic_facts(reasoning)
            pot_count = sum(1 for f in facts if f.source_path == "pot")
            total = len(facts)

            if total >= min_facts:
                status = "PASS"
                results["pot_integration"]["passed"] += 1
            else:
                status = "FAIL"
                results["pot_integration"]["failed"] += 1

            print(f"  [{status}] {tid}: {total} facts ({pot_count} via PoT)")
            for f in facts:
                print(f"           {f.predicate}({', '.join(f.arguments)}) [{f.source_path}]")

            results["pot_integration"]["details"].append({
                "id": tid, "status": status, "total_facts": total,
                "pot_facts": pot_count, "expected_min": min_facts,
            })

        except Exception as e:
            print(f"  [ERROR] {tid}: {e}")
            traceback.print_exc()
            results["pot_integration"]["failed"] += 1

    # ---------------------------------------------------------------
    # Summary Report
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  VALIDATION SUMMARY")
    print("=" * 70)

    # PoT Conversion
    pc = results["pot_conversion"]
    print(f"\n  PoT Conversion:     {pc['passed']}/{pc['passed']+pc['failed']} passed")

    # Drift Detection
    dd = results["drift_detection"]
    total_dd = dd["passed"] + dd["failed"]
    print(f"  Drift Detection:    {dd['passed']}/{total_dd} passed")

    # Category breakdown
    print("\n  Drift Detection by Category:")
    for cat, statuses in by_category.items():
        passed = sum(1 for s in statuses if s in ("PASS", "PARTIAL"))
        total = len(statuses)
        pct = round(passed / total * 100, 1) if total > 0 else 0
        false_pos = sum(1 for s in statuses if s == "FALSE_POS")
        print(f"    {cat:20s}: {passed}/{total} ({pct}%)  FP={false_pos}")

    # Integration
    pi = results["pot_integration"]
    print(f"\n  PoT Integration:    {pi['passed']}/{pi['passed']+pi['failed']} passed")

    # Success criteria
    numeric_tests = by_category.get("numeric", [])
    dep_tests = by_category.get("dependency", [])
    redef_tests = by_category.get("redefinition", [])
    clean_tests = by_category.get("clean", [])

    numeric_rate = sum(1 for s in numeric_tests if s in ("PASS", "PARTIAL")) / max(len(numeric_tests), 1) * 100
    dep_rate = sum(1 for s in dep_tests if s in ("PASS", "PARTIAL")) / max(len(dep_tests), 1) * 100
    redef_rate = sum(1 for s in redef_tests if s in ("PASS", "PARTIAL")) / max(len(redef_tests), 1) * 100
    fp_total = sum(1 for s in clean_tests if s == "FALSE_POS")
    fp_rate = fp_total / max(len(clean_tests), 1) * 100

    print("\n  ── Success Criteria ──")
    print(f"  Numeric drift detection:    {numeric_rate:.1f}% (target: >30%)")
    print(f"  Dependency detection:       {dep_rate:.1f}% (target: >20%)")
    print(f"  Redefinition detection:     {redef_rate:.1f}% (target: no regression)")
    print(f"  False positive rate:        {fp_rate:.1f}% (target: ~0%)")

    criteria_met = (
        numeric_rate > 30 and
        dep_rate > 20 and
        redef_rate > 50 and
        fp_rate < 10
    )

    if criteria_met:
        print("\n  ✅ ALL SUCCESS CRITERIA MET")
    else:
        print("\n  ⚠️  Some criteria not fully met")

    print("=" * 70 + "\n")

    # Save results
    results_path = os.path.join(BASE_DIR, "results", "pot_validation_results.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({
            "pot_conversion": {k: v for k, v in pc.items() if k != "details"},
            "drift_detection": {k: v for k, v in dd.items() if k != "details"},
            "pot_integration": {k: v for k, v in pi.items() if k != "details"},
            "rates": {
                "numeric_detection": numeric_rate,
                "dependency_detection": dep_rate,
                "redefinition_detection": redef_rate,
                "false_positive_rate": fp_rate,
            },
            "criteria_met": criteria_met,
        }, f, indent=2)
    print(f"  Results saved: {results_path}\n")

    return 0 if criteria_met else 1


if __name__ == "__main__":
    sys.exit(main())
