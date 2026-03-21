"""
main.py

Logic-Grounded Pelican (LGP)
---------------------------------
System Entry Point

Responsibilities:
- Wire full LGP pipeline
- Provide CLI interface
- Provide programmatic API
- Support batch execution
- Ensure deterministic resets between runs

This version provides a production-ready synchronous execution layer.
LangGraph integration hooks are included for extension.

Author: LGP Framework
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import List, Optional

from core.policy import get_policy_manager


# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger("LGP.Main")


# ---------------------------------------------------------------------------
# LGP Application Wrapper
# ---------------------------------------------------------------------------

class LGPApplication:
    """
    High-level application wrapper for Logic-Grounded Pelican.
    """

    def __init__(self, corpus_path: Optional[str] = None):
        self.policy_manager = get_policy_manager(corpus_path=corpus_path)

    # ------------------------------------------------------------------
    # Single Query Execution
    # ------------------------------------------------------------------

    def run(self, query: str) -> dict:
        """
        Execute a single query through LGP pipeline.
        """
        if not query or not query.strip():
            raise ValueError("Query must be non-empty.")

        # Reset state for deterministic runs
        self.policy_manager.state_manager.reset()
        self.policy_manager.logger.reset()

        result = self.policy_manager.process_query(query)
        return result

    # ------------------------------------------------------------------
    # Batch Execution
    # ------------------------------------------------------------------

    def run_batch(self, queries: List[str]) -> List[dict]:
        """
        Execute multiple queries sequentially.
        """
        results = []
        for query in queries:
            try:
                results.append(self.run(query))
            except Exception as e:
                logger.error(f"Batch query failed: {str(e)}")
                results.append({
                    "response": None,
                    "audit": {"error": str(e)}
                })
        return results


# ---------------------------------------------------------------------------
# CLI Interface
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Logic-Grounded Pelican (LGP) Execution CLI"
    )

    parser.add_argument(
        "--query",
        type=str,
        help="Single query to process",
    )

    parser.add_argument(
        "--batch",
        type=str,
        help="Path to JSON file containing list of queries",
    )

    parser.add_argument(
        "--corpus",
        type=str,
        help="Optional PopQA corpus JSONL path",
    )

    parser.add_argument(
        "--export",
        type=str,
        help="Optional path to export final result JSON",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    app = LGPApplication(corpus_path=args.corpus)

    if args.query:
        result = app.run(args.query)

    elif args.batch:
        try:
            with open(args.batch, "r", encoding="utf-8") as f:
                queries = json.load(f)

            if not isinstance(queries, list):
                raise ValueError("Batch file must contain a JSON list of queries.")

            result = app.run_batch(queries)
        except Exception as e:
            logger.critical(f"Failed to process batch file: {str(e)}")
            sys.exit(1)
    else:
        logger.error("Either --query or --batch must be provided.")
        sys.exit(1)

    # Output to console
    print(json.dumps(result, indent=2))

    # Optional export
    if args.export:
        try:
            with open(args.export, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            logger.info(f"Results exported to {args.export}")
        except Exception as e:
            logger.error(f"Failed to export results: {str(e)}")


if __name__ == "__main__":
    main()