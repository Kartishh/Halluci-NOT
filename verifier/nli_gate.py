"""
verifier/nli_gate.py

Logic-Grounded Pelican (LGP)
---------------------------------
Agreement Model (NLI Gate)

Purpose:
Provide post-hoc contradiction detection between:
    p = original query (premise)
    h = grounded hypothesis / verified output

Uses lightweight MNLI model for deterministic semantic validation.

Decision Logic:
    contradiction_prob > 0.6  → reject
    entailment_prob > 0.6     → accept
    otherwise                 → retrieve

Design Guarantees:
- Lazy model loading
- Deterministic softmax scoring
- No stochastic sampling
- Explicit label mapping
- Batch-compatible
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("LGP.NLIGate")
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Agreement Result Schema
# ---------------------------------------------------------------------------

@dataclass
class AgreementResult:
    score: float
    label: str  # "contradiction" | "neutral" | "entailment"
    decision: str  # "reject" | "retrieve" | "accept"

    def to_dict(self):
        return {
            "score": self.score,
            "label": self.label,
            "decision": self.decision,
        }


# ---------------------------------------------------------------------------
# NLI Gate
# ---------------------------------------------------------------------------

class NLIGate:
    """
    Deterministic Agreement Model wrapper.
    """

    DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-small"

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or self.DEFAULT_MODEL
        self._tokenizer = None
        self._model = None

    # ------------------------------------------------------------------
    # Lazy Loading
    # ------------------------------------------------------------------

    def _load_model(self):
        if self._model is None or self._tokenizer is None:
            logger.info(f"Loading NLI model: {self.model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            )
            self._model.eval()

    # ------------------------------------------------------------------
    # Core Scoring
    # ------------------------------------------------------------------

    def check_contradiction(self, premise: str, hypothesis: str) -> AgreementResult:
        """
        Compute semantic agreement between premise and hypothesis.
        """
        if not premise or not hypothesis:
            raise ValueError("Premise and hypothesis must be non-empty.")

        self._load_model()

        inputs = self._tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
        )

        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0]

        # MNLI label mapping: 0=contradiction, 1=neutral, 2=entailment
        contradiction_prob = probs[0].item()
        neutral_prob = probs[1].item()
        entailment_prob = probs[2].item()

        score = entailment_prob  # alignment confidence for logging

        if contradiction_prob > 0.6:
            decision = "reject"
            label = "contradiction"
        elif entailment_prob > 0.6:
            decision = "accept"
            label = "entailment"
        else:
            decision = "retrieve"
            label = "neutral"

        return AgreementResult(
            score=score,
            label=label,
            decision=decision,
        )

    # ------------------------------------------------------------------
    # Batch Support
    # ------------------------------------------------------------------

    def batch_check(
        self,
        pairs: List[Tuple[str, str]]
    ) -> List[AgreementResult]:
        """
        Batch NLI scoring.
        """
        if not pairs:
            return []

        self._load_model()

        premises = [p for p, _ in pairs]
        hypotheses = [h for _, h in pairs]

        inputs = self._tokenizer(
            premises,
            hypotheses,
            return_tensors="pt",
            truncation=True,
            padding=True,
        )

        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

        results: List[AgreementResult] = []

        for prob in probs:
            contradiction_prob = prob[0].item()
            neutral_prob = prob[1].item()
            entailment_prob = prob[2].item()

            score = entailment_prob

            if contradiction_prob > 0.6:
                decision = "reject"
                label = "contradiction"
            elif entailment_prob > 0.6:
                decision = "accept"
                label = "entailment"
            else:
                decision = "retrieve"
                label = "neutral"

            results.append(
                AgreementResult(score=score, label=label, decision=decision)
            )

        return results


# ---------------------------------------------------------------------------
# Convenience Accessor
# ---------------------------------------------------------------------------


def get_nli_gate() -> NLIGate:
    return NLIGate()