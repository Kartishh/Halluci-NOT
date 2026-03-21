"""
verifier/sandbox.py

HalluciNOT (LGP)
---------------------------------
Sandboxed Executor with Early Exit Fast Path

ARCHITECTURE:
    1. fast_execute: AST-checks the script for safety. If it contains
       only math, assignments, and allowed imports → executes in-process
       with restricted __builtins__. (~2ms)
    2. execute: Full Docker container isolation for complex/unsafe
       scripts. (~5000ms)
    3. execute_smart: Tries fast path first, falls back to Docker.

Author: LGP Framework
"""

from __future__ import annotations

import ast
import json
import logging
import tempfile
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set

import docker
from docker.errors import DockerException


logger = logging.getLogger("LGP.Sandbox")
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Result Schema
# ---------------------------------------------------------------------------

@dataclass
class SandboxResult:
    success: bool
    output: Optional[Dict[str, Any]]
    error: Optional[str]
    timeout: bool
    fast_path: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "timeout": self.timeout,
            "fast_path": self.fast_path,
        }


# ---------------------------------------------------------------------------
# AST Safety Checker
# ---------------------------------------------------------------------------

# Allowed AST node types for fast-path execution
_SAFE_NODES: Set[type] = {
    ast.Module, ast.Expr, ast.Assign, ast.AugAssign,
    ast.BinOp, ast.UnaryOp, ast.Compare,
    # Operators
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv,
    ast.Mod, ast.Pow, ast.USub, ast.UAdd,
    # Comparison operators
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    # Values
    ast.Constant, ast.Name, ast.Load, ast.Store,
    # Collections (for __result__ dict)
    ast.Dict, ast.List, ast.Tuple,
    ast.Subscript, ast.Index,
    # Import (only math/sympy allowed)
    ast.Import, ast.ImportFrom, ast.alias,
}

_ALLOWED_IMPORTS = {"math", "sympy", "json"}


def is_safe_script(script: str) -> bool:
    """
    AST-walk the script to determine if it's safe for in-process execution.

    Safe means:
        - Only basic math operations and assignments
        - No function calls (except dict construction)
        - No loops, no comprehensions
        - No file I/O, no exec/eval
        - Only allowed imports (math, sympy, json)

    Returns True if safe for fast path, False if Docker is needed.
    """
    try:
        tree = ast.parse(script)
    except SyntaxError:
        return False

    for node in ast.walk(tree):
        node_type = type(node)

        # Allow Call only for dict() and specific safe builtins
        if node_type == ast.Call:
            if isinstance(node.func, ast.Name):
                if node.func.id in ("dict", "int", "float", "str", "abs",
                                     "round", "min", "max", "len", "repr"):
                    continue
            return False

        # Check imports are allowed
        if node_type == ast.Import:
            for alias in node.names:
                if alias.name not in _ALLOWED_IMPORTS:
                    return False
            continue

        if node_type == ast.ImportFrom:
            if node.module not in _ALLOWED_IMPORTS:
                return False
            continue

        # Allow attribute access (e.g., math.sqrt)
        if node_type == ast.Attribute:
            continue

        # Reject anything not in safe set
        if node_type not in _SAFE_NODES:
            return False

    return True


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class SandboxExecutor:

    DEFAULT_IMAGE = "lgp-sandbox:latest"
    DEFAULT_TIMEOUT = 5  # seconds
    MEMORY_LIMIT = "128m"
    CPU_QUOTA = 50000

    def __init__(self) -> None:
        try:
            self.client = docker.from_env()
        except DockerException as e:
            raise RuntimeError("Docker must be running for SandboxExecutor.") from e

    # ------------------------------------------------------------------
    # Smart Execute (Fast Path + Docker Fallback)
    # ------------------------------------------------------------------

    def execute(self, script: str, timeout: Optional[int] = None) -> SandboxResult:
        """
        Smart execution: tries fast path first, falls back to Docker.
        """
        # Try fast path for simple scripts
        if is_safe_script(script):
            result = self.fast_execute(script)
            if result.success:
                return result
            # If fast path failed (e.g., runtime error), fall through to Docker
            logger.info("Fast path execution failed, falling back to Docker.")

        return self._docker_execute(script, timeout)

    # ------------------------------------------------------------------
    # Fast Path (In-Process, Restricted)
    # ------------------------------------------------------------------

    def fast_execute(self, script: str) -> SandboxResult:
        """
        Execute a safe script in-process with restricted builtins.

        Security:
            - __builtins__ restricted to safe math functions only
            - No file I/O, no network, no eval/exec
            - AST pre-validated by is_safe_script()
        """
        # Append __result__ extraction
        full_script = script + "\nimport json\n"

        # Restricted builtins — only safe math/type functions
        safe_builtins = {
            "abs": abs, "round": round, "int": int, "float": float,
            "str": str, "len": len, "min": min, "max": max,
            "repr": repr, "dict": dict, "list": list, "tuple": tuple,
            "True": True, "False": False, "None": None,
            "print": lambda *a, **kw: None,  # suppress print
            "__import__": __import__,  # needed for import math/sympy
        }

        restricted_globals = {"__builtins__": safe_builtins}
        local_ns: Dict[str, Any] = {}

        try:
            exec(full_script, restricted_globals, local_ns)

            result = local_ns.get("__result__", {})

            if not isinstance(result, dict):
                return SandboxResult(
                    success=False,
                    output=None,
                    error="Fast path: __result__ is not a dict.",
                    timeout=False,
                    fast_path=True,
                )

            return SandboxResult(
                success=True,
                output=result,
                error=None,
                timeout=False,
                fast_path=True,
            )

        except Exception as e:
            return SandboxResult(
                success=False,
                output=None,
                error=f"Fast path error: {str(e)}",
                timeout=False,
                fast_path=True,
            )

    # ------------------------------------------------------------------
    # Docker Execute (Full Isolation)
    # ------------------------------------------------------------------

    def _docker_execute(self, script: str, timeout: Optional[int] = None) -> SandboxResult:

        timeout = timeout or self.DEFAULT_TIMEOUT

        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = os.path.join(tmpdir, "script.py")

            safe_script = script + "\nimport json\nprint(json.dumps(__result__))"

            with open(script_path, "w", encoding="utf-8") as f:
                f.write(safe_script)

            container = None

            try:
                container = self.client.containers.create(
                    image=self.DEFAULT_IMAGE,
                    command="python /app/script.py",
                    volumes={tmpdir: {"bind": "/app", "mode": "ro"}},
                    network_disabled=True,
                    mem_limit=self.MEMORY_LIMIT,
                    cpu_quota=self.CPU_QUOTA,
                )

                container.start()

                try:
                    container.wait(timeout=timeout)
                except Exception:
                    container.kill()
                    return SandboxResult(
                        success=False,
                        output=None,
                        error="Execution timed out.",
                        timeout=True,
                    )

                logs = container.logs(stdout=True, stderr=True)
                result = logs.decode("utf-8").strip()

                if not result:
                    return SandboxResult(
                        success=False,
                        output=None,
                        error="No output produced.",
                        timeout=False,
                    )

                try:
                    parsed = json.loads(result)
                except json.JSONDecodeError:
                    return SandboxResult(
                        success=False,
                        output=None,
                        error=f"Invalid JSON output from sandbox: {result}",
                        timeout=False,
                    )

                return SandboxResult(
                    success=True,
                    output=parsed,
                    error=None,
                    timeout=False,
                )

            except Exception as e:
                return SandboxResult(
                    success=False,
                    output=None,
                    error=str(e),
                    timeout=False,
                )

            finally:
                if container is not None:
                    try:
                        container.remove(force=True)
                    except Exception:
                        pass


# ---------------------------------------------------------------------------
# Accessor
# ---------------------------------------------------------------------------


def get_sandbox_executor() -> SandboxExecutor:
    return SandboxExecutor()

