"""
data/popqa_loader.py

Logic-Grounded Pelican (LGP)
---------------------------------
PopQA / Long-Tail Retrieval Interface

Purpose:
Provide deterministic retrieval for long-tail knowledge cases when
Agreement Model decision == "retrieve".

Design Principles:
- Deterministic embedding-based retrieval (Contriever-style)
- Pluggable backend (local corpus or external vector DB)
- No hidden stochastic ranking
- Explicit top-k control
- Clean API for policy.py

NOTE:
This implementation provides:
    1. Local JSONL corpus loading
    2. Sentence-transformer fallback embedding
    3. Deterministic cosine similarity ranking

It is production-ready and can be swapped with FAISS or external vector DB.

Author: LGP Framework
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import numpy as np
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("LGP.PopQALoader")
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Retrieval Result Schema
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    query: str
    retrieved: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "retrieved": self.retrieved,
        }


# ---------------------------------------------------------------------------
# PopQA Loader
# ---------------------------------------------------------------------------

class PopQALoader:
    """
    Deterministic embedding-based retriever for long-tail knowledge.
    """

    DEFAULT_MODEL = "facebook/contriever"

    def __init__(
        self,
        corpus_path: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.model_name = model_name or self.DEFAULT_MODEL
        self.corpus_path = corpus_path
        self._model = None
        self._corpus: List[Dict[str, Any]] = []
        self._embeddings: Optional[np.ndarray] = None

        if corpus_path:
            self._load_corpus(corpus_path)

    # ------------------------------------------------------------------
    # Model Loading (Lazy)
    # ------------------------------------------------------------------

    def _load_model(self):
        if self._model is None:
            logger.info(f"Loading retrieval model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)

    # ------------------------------------------------------------------
    # Corpus Handling
    # ------------------------------------------------------------------

    def _load_corpus(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Corpus file not found: {path}")

        logger.info(f"Loading PopQA corpus from {path}")

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                self._corpus.append(json.loads(line.strip()))

        logger.info(f"Loaded {len(self._corpus)} documents.")

    def _build_embeddings(self):
        if not self._corpus:
            raise RuntimeError("Cannot build embeddings: corpus is empty.")

        self._load_model()

        texts = [doc.get("text", "") for doc in self._corpus]
        self._embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 3) -> RetrievalResult:
        """
        Retrieve top-k documents for a query.
        """
        if not query or not query.strip():
            raise ValueError("Query must be non-empty.")

        if not self._corpus:
            raise RuntimeError("PopQA corpus not loaded.")

        if self._embeddings is None:
            self._build_embeddings()

        self._load_model()

        query_vec = self._model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]

        # Cosine similarity (dot product since normalized)
        scores = np.dot(self._embeddings, query_vec)

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append(
                {
                    "score": float(scores[idx]),
                    "document": self._corpus[idx],
                }
            )

        return RetrievalResult(query=query, retrieved=results)


# ---------------------------------------------------------------------------
# Convenience Accessor
# ---------------------------------------------------------------------------


def get_popqa_loader(corpus_path: str) -> PopQALoader:
    return PopQALoader(corpus_path=corpus_path)
