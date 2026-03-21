"""
evaluation/stress_tests.py

HalluciNOT (LGP)
---------------------------------
Drift Stress Testing Suite

Synthetic adversarial test cases designed to stress-test the SSCE engine
across ALL types of logical reasoning, not just arithmetic.

Categories tested:
    1. Direct redefinition (arithmetic)
    2. Silent overwrite after computation
    3. Legitimate updates (should NOT trigger drift)
    4. Chained multi-step computation
    5. Logical condition drift
    6. Comparison drift
    7. Multi-variable relation drift
    8. Temporal reasoning drift
    9. Alias collision
    10. Large chain stress

Author: LGP Framework
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from typing import List, Optional

from symbolic.decomposer import get_symbolic_decomposer
from symbolic.ssce_algorithm import get_ssce_engine, SSCEEnforcementError
from symbolic.table import get_symbolic_table
from verifier.pot_engine import get_pot_engine
from verifier.sandbox import get_sandbox_executor

logger = logging.getLogger("LGP.Eval.StressTest")
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Test Case Schema
# ---------------------------------------------------------------------------

@dataclass
class StressTestCase:
    """
    A single stress test scenario.

    Attributes:
        id: Unique test identifier
        name: Human-readable test name
        description: What this test validates
        query: Input query string (multi-step separated by newlines)
        expect_drift: Whether SSCE should detect drift
        category: Logic category being stressed
    """
    id: str
    name: str
    description: str
    query: str
    expect_drift: bool
    category: str


@dataclass
class StressTestResult:
    """Result of running a stress test case."""
    test_id: str
    test_name: str
    category: str
    expect_drift: bool
    actual_drift: bool
    passed: bool
    error: Optional[str]
    latency_ms: float

    def status_emoji(self) -> str:
        return "✅" if self.passed else "❌"


# ---------------------------------------------------------------------------
# Test Suite Definition
# ---------------------------------------------------------------------------

def get_stress_test_suite() -> List[StressTestCase]:
    """Return the full adversarial stress test suite."""

    return [
        # ---------------------------------------------------------------
        # ARITHMETIC DRIFT TESTS
        # ---------------------------------------------------------------
        StressTestCase(
            id="arith_001",
            name="Direct Redefinition",
            description="Variable redefined to different value without justification",
            query="x = 5\nx = 10",
            expect_drift=True,
            category="arithmetic",
        ),
        StressTestCase(
            id="arith_002",
            name="Silent Overwrite After Computation",
            description="Variable used in computation then silently changed",
            query="price = 5\ntotal = price * 3\nprice = 7",
            expect_drift=True,
            category="arithmetic",
        ),
        StressTestCase(
            id="arith_003",
            name="Legitimate New Variable",
            description="Derived variable from existing — should NOT trigger drift",
            query="x = 5\ny = x + 2",
            expect_drift=False,
            category="arithmetic",
        ),
        StressTestCase(
            id="arith_004",
            name="Chained Computation",
            description="Multi-step chain — all legitimate",
            query="a = 2\nb = a * 3\nc = b + 1",
            expect_drift=False,
            category="arithmetic",
        ),
        StressTestCase(
            id="arith_005",
            name="Triple Redefinition",
            description="Variable redefined multiple times",
            query="x = 5\nx = x + 2\nx = 10",
            expect_drift=True,
            category="arithmetic",
        ),
        StressTestCase(
            id="arith_006",
            name="Same Value Reassignment",
            description="Variable reassigned to same value — should NOT trigger",
            query="x = 5\nx = 5",
            expect_drift=False,
            category="arithmetic",
        ),
        StressTestCase(
            id="arith_007",
            name="Large Chain (10 steps)",
            description="Long computation chain — should be stable",
            query="a = 1\nb = a + 1\nc = b + 1\nd = c + 1\ne = d + 1\nf = e + 1\ng = f + 1\nh = g + 1\ni = h + 1\nj = i + 1",
            expect_drift=False,
            category="arithmetic",
        ),
        StressTestCase(
            id="arith_008",
            name="Subtraction Drift",
            description="Subtraction result overwrites source variable",
            query="balance = 100\nbalance = 50",
            expect_drift=True,
            category="arithmetic",
        ),

        # ---------------------------------------------------------------
        # COMPARISON LOGIC TESTS
        # ---------------------------------------------------------------
        StressTestCase(
            id="comp_001",
            name="Comparison Variable Drift",
            description="Comparison operand changes value silently",
            query="threshold = 10\nvalue = 15\nthreshold = 20",
            expect_drift=True,
            category="comparison",
        ),
        StressTestCase(
            id="comp_002",
            name="Derived Comparison",
            description="New variable from comparison — legitimate",
            query="a = 10\nb = 20\nmax_val = b",
            expect_drift=False,
            category="comparison",
        ),

        # ---------------------------------------------------------------
        # CONDITIONAL LOGIC TESTS
        # ---------------------------------------------------------------
        StressTestCase(
            id="cond_001",
            name="Condition Variable Drift",
            description="Variable used in condition changes later",
            query="discount_rate = 10\ntotal = 100\ndiscount_rate = 25",
            expect_drift=True,
            category="conditional",
        ),
        StressTestCase(
            id="cond_002",
            name="Independent Conditional Variables",
            description="Non-overlapping variables — no drift",
            query="flag = 1\nresult = flag * 100",
            expect_drift=False,
            category="conditional",
        ),

        # ---------------------------------------------------------------
        # MULTI-VARIABLE RELATION TESTS
        # ---------------------------------------------------------------
        StressTestCase(
            id="multi_001",
            name="Cross-Variable Contamination",
            description="Two related variables, one drifts affecting relationship",
            query="width = 5\nheight = 10\narea = width * height\nwidth = 8",
            expect_drift=True,
            category="multi_variable",
        ),
        StressTestCase(
            id="multi_002",
            name="Independent Multi-Variable",
            description="Multiple independent variables — no drift",
            query="x = 5\ny = 10\nz = 15",
            expect_drift=False,
            category="multi_variable",
        ),
        StressTestCase(
            id="multi_003",
            name="Cascading Computation",
            description="Each variable depends on previous — all legitimate",
            query="base = 100\ntax = base * 20\nfinal = base + tax",
            expect_drift=False,
            category="multi_variable",
        ),

        # ---------------------------------------------------------------
        # TEMPORAL REASONING TESTS
        # ---------------------------------------------------------------
        StressTestCase(
            id="temp_001",
            name="Temporal Value Override",
            description="Time-dependent value changes without temporal justification",
            query="count = 10\ncount = 5",
            expect_drift=True,
            category="temporal",
        ),
        StressTestCase(
            id="temp_002",
            name="Sequential Accumulation",
            description="Value legitimately grows over steps",
            query="total = 0\nadded = total + 5",
            expect_drift=False,
            category="temporal",
        ),

        # ---------------------------------------------------------------
        # EDGE CASES
        # ---------------------------------------------------------------
        StressTestCase(
            id="edge_001",
            name="Zero to Non-Zero",
            description="Variable goes from 0 to non-zero — drift",
            query="x = 0\nx = 5",
            expect_drift=True,
            category="edge_case",
        ),
        StressTestCase(
            id="edge_002",
            name="Negative Value Drift",
            description="Positive to negative without operation",
            query="balance = 100\nbalance = -50",
            expect_drift=True,
            category="edge_case",
        ),
        StressTestCase(
            id="edge_003",
            name="Single Variable Only",
            description="Only one assignment — no drift possible",
            query="x = 42",
            expect_drift=False,
            category="edge_case",
        ),
    ]


# ---------------------------------------------------------------------------
# Test Runner
# ---------------------------------------------------------------------------

def run_stress_tests(
    verbose: bool = True,
) -> List[StressTestResult]:
    """
    Run all stress tests and return results.

    Each test:
        1. Resets symbolic state
        2. Runs query through decomposer → PoT → sandbox → SSCE
        3. Checks whether drift was detected
        4. Compares against expected result
    """
    suite = get_stress_test_suite()
    results: List[StressTestResult] = []
    decomposer = get_symbolic_decomposer()
    pot_engine = get_pot_engine()
    sandbox = get_sandbox_executor()
    table = get_symbolic_table()

    for test in suite:
        ssce = get_ssce_engine()
        table.clear()

        start_time = time.time()
        actual_drift = False
        error_msg = None

        try:
            facts = decomposer.to_atomic_facts(test.query)

            for fact in facts:
                # Build context from current table
                context_script = ""
                for var, record in table.snapshot().items():
                    context_script += f"{var} = {repr(record['value'])}\n"

                pot_script = pot_engine.generate_script([fact])
                full_script = context_script + "\n" + pot_script.script

                sandbox_result = sandbox.execute(full_script)

                if not sandbox_result.success:
                    error_msg = f"Sandbox error: {sandbox_result.error}"
                    break

                # SSCE check
                try:
                    ssce.enforce(sandbox_result.output)
                except SSCEEnforcementError:
                    actual_drift = True
                    break

                # Commit to table if no drift
                for var, value in sandbox_result.output.items():
                    table.set(var, value, f"stress_test_{test.id}")

        except Exception as e:
            error_msg = str(e)

        elapsed_ms = (time.time() - start_time) * 1000

        passed = (actual_drift == test.expect_drift) and error_msg is None

        result = StressTestResult(
            test_id=test.id,
            test_name=test.name,
            category=test.category,
            expect_drift=test.expect_drift,
            actual_drift=actual_drift,
            passed=passed,
            error=error_msg,
            latency_ms=round(elapsed_ms, 2),
        )

        results.append(result)

        if verbose:
            status = result.status_emoji()
            drift_info = f"drift={'YES' if actual_drift else 'NO'}"
            expected_info = f"expected={'YES' if test.expect_drift else 'NO'}"
            err_info = f" [{error_msg}]" if error_msg else ""
            print(
                f"  {status} [{test.id}] {test.name} "
                f"({drift_info}, {expected_info}){err_info}"
            )

    # Summary
    passed_count = sum(1 for r in results if r.passed)
    total = len(results)
    print(f"\n{'='*60}")
    print(f"Stress Test Results: {passed_count}/{total} passed")
    print(f"{'='*60}")

    # Per-category breakdown
    categories = sorted(set(r.category for r in results))
    for cat in categories:
        cat_results = [r for r in results if r.category == cat]
        cat_passed = sum(1 for r in cat_results if r.passed)
        print(f"  {cat}: {cat_passed}/{len(cat_results)}")

    return results


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("HalluciNOT — SSCE Drift Stress Test Suite")
    print("=" * 60)
    print()

    results = run_stress_tests(verbose=True)

    # Exit code reflects test status
    all_passed = all(r.passed for r in results)
    sys.exit(0 if all_passed else 1)
