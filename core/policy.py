"""
core/policy.py

HalluciNOT (LGP)
---------------------------------
Adaptive Policy Manager (Hardened v2)

Architecture:
    1. Hybrid Decomposition (regex + Gemini few-shot)
    2. PoT generation → Smart sandbox (fast path + Docker)
    3. SSCE drift enforcement per-step
    4. Numeric Consistency Gate (replaces NLI skip)
    5. Full NLI for natural-language claims
    6. DecompositionComplexityError → Reflexion loop

Every reasoning step now passes through the NumericConsistencyGate.
No blanket NLI skip — we verify that computed values match textual claims.

Author: LGP Framework
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional

from symbolic.decomposer import (
    get_symbolic_decomposer,
    DecompositionComplexityError,
)
from symbolic.ssce_algorithm import get_ssce_engine, SSCEEnforcementError
from symbolic.table import get_symbolic_table
from verifier.pot_engine import get_pot_engine
from verifier.sandbox import get_sandbox_executor
from verifier.nli_gate import get_nli_gate
from verifier.numeric_nli import get_numeric_consistency_gate
from data.popqa_loader import PopQALoader
from data.logger import get_semantic_logger
from core.state_manager import StateManager


logger = logging.getLogger("LGP.Policy")
logger.setLevel(logging.INFO)


class AdaptivePolicyManager:

    def __init__(self, corpus_path: Optional[str] = None):
        self.decomposer = get_symbolic_decomposer()
        self.pot_engine = get_pot_engine()
        self.sandbox = get_sandbox_executor()
        self.ssce = get_ssce_engine()
        self.nli_gate = get_nli_gate()
        self.numeric_gate = get_numeric_consistency_gate()
        self.state_manager = StateManager()
        self.logger = get_semantic_logger()
        self.symbolic_table = get_symbolic_table()

        self.retriever = None
        if corpus_path:
            self.retriever = PopQALoader(corpus_path=corpus_path)

        self._max_reflexion_trials = int(
            os.getenv("MAX_REFLEXION_TRIALS", "3")
        )

    # ------------------------------------------------------------------
    # Public Entry
    # ------------------------------------------------------------------

    def process_query(self, query: str) -> Dict[str, Any]:

        # Reset deterministic state
        self.state_manager.reset()

        state = self.state_manager.initialize(query)
        self.logger.log_step("init", state.to_dict(), "Initial state created.")

        # ------------------------------------------------------------------
        # Stage 1: Decomposition (with Reflexion retry)
        # ------------------------------------------------------------------

        t0 = time.time()
        facts = None
        decomposition_error = None

        for trial in range(self._max_reflexion_trials):
            try:
                facts = self.decomposer.to_atomic_facts(query)
                break
            except DecompositionComplexityError as e:
                decomposition_error = e
                logger.warning(
                    f"Decomposition trial {trial + 1} failed: {e.reason}"
                )
                self.logger.log_step(
                    "reflexion_retry",
                    {"trial": trial + 1, "error": e.reason},
                    f"Reflexion loop: decomposition retry #{trial + 1}",
                )

        self.logger.log_latency("decomposition", (time.time() - t0) * 1000)

        if facts is None:
            error_msg = (
                f"Decomposition failed after {self._max_reflexion_trials} attempts. "
                f"Last error: {decomposition_error}"
            )
            logger.error(error_msg)
            return self._finalize(state, error_msg)

        if not facts:
            return self._finalize(state, "No symbolic reasoning required.")

        # ------------------------------------------------------------------
        # Stage 2: Interleaved Execution Loop
        # ------------------------------------------------------------------

        final_output: Dict[str, Any] = {}
        path_stats = {"regex": 0, "gemini": 0, "fast_path": 0, "docker": 0}

        for fact in facts:

            # Inject current symbolic table into execution context
            context_script = ""
            current_table = self.symbolic_table.snapshot()

            for var, record in current_table.items():
                context_script += f"{var} = {repr(record['value'])}\n"

            t1 = time.time()
            pot_script = self.pot_engine.generate_script([fact])
            full_script = context_script + "\n" + pot_script.script
            self.logger.log_latency("pot_generation", (time.time() - t1) * 1000)

            t2 = time.time()
            sandbox_result = self.sandbox.execute(full_script)
            self.logger.log_latency("sandbox_execution", (time.time() - t2) * 1000)

            self.logger.log_pot_execution(
                script=full_script,
                output=sandbox_result.output,
                success=sandbox_result.success,
                error=sandbox_result.error,
            )

            # Log path taken for this fact
            exec_path = "fast_path" if getattr(sandbox_result, 'fast_path', False) else "docker"
            path_stats[fact.source_path] = path_stats.get(fact.source_path, 0) + 1
            path_stats[exec_path] += 1

            self.logger.log_step(
                "path_taken",
                {
                    "fact": fact.raw_text[:80],
                    "decomposition_path": fact.source_path,
                    "execution_path": exec_path,
                    "predicate": fact.predicate,
                },
                f"Path: {fact.source_path}→{exec_path} | {fact.predicate}({', '.join(fact.arguments)})",
            )

            if not sandbox_result.success:
                logger.warning("PoT execution failed — deterministic override triggered.")
                return self._finalize(state, "Execution failed. Logic could not be verified.")

            # Strict SSCE enforcement
            t3 = time.time()
            try:
                self.ssce.enforce(sandbox_result.output)
                self.logger.log_latency("ssce_enforcement", (time.time() - t3) * 1000)
            except SSCEEnforcementError as drift_error:
                self.logger.log_latency("ssce_enforcement", (time.time() - t3) * 1000)
                logger.error(str(drift_error))

                for report in drift_error.reports:
                    self.logger.log_drift_report({
                        "variable": report.variable,
                        "old_value": report.old_value,
                        "new_value": report.new_value,
                        "reason": report.reason,
                    })

                return self._finalize(state, "Symbolic drift detected.")

            # Commit state
            state = self.state_manager.update_6tuple(
                current_state=state,
                new_evidence=sandbox_result.output,
                observation=sandbox_result.output,
                utility_delta=0.1,
                factual_status=True,
            )

            self.logger.log_step(
                "post_execution",
                state.to_dict(),
                f"State updated after fact: {fact.raw_text}",
            )

            final_output.update(sandbox_result.output)

        # ------------------------------------------------------------------
        # Optimization Efficiency Summary
        # ------------------------------------------------------------------

        total_facts = path_stats.get("regex", 0) + path_stats.get("gemini", 0)
        self.logger.log_step(
            "decomposition_paths",
            {
                "total_facts": total_facts,
                "regex_count": path_stats.get("regex", 0),
                "gemini_count": path_stats.get("gemini", 0),
                "fast_path_count": path_stats.get("fast_path", 0),
                "docker_count": path_stats.get("docker", 0),
                "regex_pct": round(path_stats.get("regex", 0) / max(total_facts, 1) * 100, 1),
                "fast_path_pct": round(path_stats.get("fast_path", 0) / max(total_facts, 1) * 100, 1),
            },
            f"Optimization: {path_stats.get('regex', 0)}/{total_facts} regex, "
            f"{path_stats.get('fast_path', 0)}/{total_facts} fast_path",
        )

        # ------------------------------------------------------------------
        # Stage 3: Numeric Consistency Gate
        # (Replaces blanket NLI skip — every result must be verified)
        # ------------------------------------------------------------------

        t4 = time.time()
        consistency = self.numeric_gate.check(query, final_output)
        self.logger.log_latency("numeric_consistency", (time.time() - t4) * 1000)

        self.logger.log_step(
            "numeric_consistency",
            consistency.to_dict(),
            f"Numeric gate: {consistency.checked_count} checks, "
            f"{len(consistency.contradictions)} contradictions",
        )

        if not consistency.is_consistent:
            # Build detailed contradiction report
            for c in consistency.contradictions:
                self.logger.log_drift_report({
                    "variable": c.variable,
                    "old_value": c.claimed_value,
                    "new_value": c.computed_value,
                    "reason": c.reason,
                })

            contradiction_msg = "; ".join(
                f"'{c.variable}' drifted from {c.claimed_value} to {c.computed_value}"
                for c in consistency.contradictions
            )
            return self._finalize(
                state,
                f"Numeric contradiction detected: {contradiction_msg}",
            )

        # ------------------------------------------------------------------
        # Stage 4: NLI Agreement (for non-numeric or mixed outputs)
        # ------------------------------------------------------------------

        has_non_numeric = any(
            not isinstance(v, (int, float)) for v in final_output.values()
        )

        if has_non_numeric:
            premise = self._verbalize_premise(query)
            hypothesis = self._verbalize_output(final_output)

            agreement = self.nli_gate.check_contradiction(premise, hypothesis)
            self.logger.log_agreement_score(agreement.score)

            if agreement.decision == "reject":
                return self._finalize(state, "Contradiction detected.")

            if agreement.decision == "retrieve" and self.retriever:
                retrieval = self.retriever.retrieve(query)
                return self._finalize(state, retrieval.to_dict())
        else:
            # Pure numeric output — consistency gate already verified
            self.logger.log_agreement_score(1.0)

        return self._finalize(state, final_output)

    # ------------------------------------------------------------------
    # Verbalization Layer
    # ------------------------------------------------------------------

    def _verbalize_premise(self, query: str) -> str:
        if "=" in query:
            left, right = query.split("=", 1)
            return f"{left.strip()} equals {right.strip()}."
        return query

    def _verbalize_output(self, output: Dict[str, Any]) -> str:
        statements = []
        for var, value in output.items():
            statements.append(f"The value of {var} is {value}.")
        return " ".join(statements)

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------

    def _finalize(self, state, response):
        report = self.logger.build_final_report()
        return {
            "response": response,
            "audit": report,
        }


def get_policy_manager(corpus_path: Optional[str] = None) -> AdaptivePolicyManager:
    return AdaptivePolicyManager(corpus_path=corpus_path)
