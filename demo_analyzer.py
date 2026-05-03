from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_ENDPOINT = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_API_VERSION = "2023-06-01"
RESULTS_DIR = Path("results/demo_runs")


class AnalyzerError(RuntimeError):
    def __init__(self, message: str, retry_without_schema: bool = False) -> None:
        super().__init__(message)
        self.retry_without_schema = retry_without_schema


class ReasoningAnalyzer:
    """
    Provider-neutral reasoning audit client for the working demo.

    The public UI calls this a "reasoning analyzer"; endpoint and model details
    stay server-side.
    """

    def __init__(self) -> None:
        load_env_file()
        self.api_key_env = os.getenv("REASONING_ANALYZER_API_KEY_ENV", "REASONING_ANALYZER_API_KEY")
        self.model_env = os.getenv("REASONING_ANALYZER_MODEL_ENV", "REASONING_ANALYZER_MODEL")
        self.endpoint_env = os.getenv("REASONING_ANALYZER_URL_ENV", "REASONING_ANALYZER_URL")
        self.endpoint = os.getenv(self.endpoint_env, DEFAULT_ENDPOINT)
        self.model = os.getenv(self.model_env, DEFAULT_MODEL)
        self.api_version = os.getenv("REASONING_ANALYZER_API_VERSION", DEFAULT_API_VERSION)
        self.max_tokens = int(os.getenv("REASONING_ANALYZER_MAX_TOKENS", "2200"))
        self.temperature = float(os.getenv("REASONING_ANALYZER_TEMPERATURE", "0"))
        self.timeout_sec = int(os.getenv("REASONING_ANALYZER_TIMEOUT_SEC", "90"))
        self.use_structured_output = os.getenv("REASONING_ANALYZER_STRUCTURED_OUTPUT", "true").lower() != "false"

    def is_configured(self) -> bool:
        return bool(self._api_key())

    def analyze(self, problem: str, draft_reasoning: str = "") -> Dict[str, Any]:
        problem = problem.strip()
        draft_reasoning = draft_reasoning.strip()
        if not problem:
            raise ValueError("Problem text is required.")
        api_key = self._api_key()
        if not api_key:
            raise AnalyzerError(f"Reasoning analyzer key is missing. Set {self.api_key_env} in the server environment.")

        started = time.time()
        payload: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "system": (
                "You are a rigorous logic-grounded reasoning audit engine. "
                "Decompose claims into symbolic steps, verify arithmetic and factual consistency from visible information, "
                "detect symbolic drift, and provide a corrected final answer when possible. "
                "Do not mention the model, vendor, provider, or hidden implementation."
            ),
            "messages": [
                {
                    "role": "user",
                    "content": self._build_prompt(problem=problem, draft_reasoning=draft_reasoning),
                }
            ],
        }
        if self.use_structured_output:
            payload["output_config"] = {"format": {"type": "json_schema", "schema": analysis_schema()}}

        try:
            raw = self._send(payload=payload, api_key=api_key)
        except AnalyzerError as exc:
            if "output_config" not in payload or not exc.retry_without_schema:
                raise
            payload.pop("output_config", None)
            raw = self._send(payload=payload, api_key=api_key)

        result = normalize_analysis(raw)
        result["analysis_mode"] = "logic_grounded_review"
        result["latency_ms"] = round((time.time() - started) * 1000, 2)
        result["created_at"] = datetime.utcnow().isoformat()
        result["input"] = {
            "problem": problem,
            "draft_reasoning": draft_reasoning,
            "has_draft_reasoning": bool(draft_reasoning),
        }
        write_demo_result(result)
        return result

    def _api_key(self) -> str:
        return os.getenv(self.api_key_env) or os.getenv("ANTHROPIC_API_KEY", "")

    def _send(self, payload: Dict[str, Any], api_key: str) -> Dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=data,
            method="POST",
            headers={
                "content-type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": self.api_version,
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500].lower()
            retry_without_schema = exc.code == 400 and (
                "output_config" in detail or "schema" in detail or "structured" in detail
            )
            raise AnalyzerError("Reasoning analyzer request failed.", retry_without_schema=retry_without_schema) from exc
        except urllib.error.URLError as exc:
            raise AnalyzerError("Reasoning analyzer could not be reached.") from exc

        try:
            response_json = json.loads(body)
        except json.JSONDecodeError as exc:
            raise AnalyzerError("Reasoning analyzer returned invalid JSON.") from exc

        text = extract_response_text(response_json)
        parsed = parse_json_object(text)
        if not parsed:
            raise AnalyzerError("Reasoning analyzer returned an unreadable response.")
        return parsed

    @staticmethod
    def _build_prompt(problem: str, draft_reasoning: str) -> str:
        draft_block = draft_reasoning or "(No draft reasoning supplied. Generate a baseline answer, then audit it.)"
        return (
            "Analyze this reasoning task as a Halluci-NOT style symbolic drift audit.\n\n"
            f"Problem:\n{problem}\n\n"
            f"Draft reasoning or answer:\n{draft_block}\n\n"
            "Return only JSON with keys: verdict, confidence, summary, baseline_answer, verified_answer, "
            "corrected_reasoning, symbolic_steps, drift_reports, audit_notes. "
            "Use verdict as one of: Supported, Drift Detected, Corrected, Uncertain. "
            "For symbolic_steps, include ordered claim/operation/computed_value/status rows. "
            "For drift_reports, include claimed_value and verified_value when a contradiction is found."
        )


def analysis_schema() -> Dict[str, Any]:
    step_schema = {
        "type": "object",
        "properties": {
            "step": {"type": "integer"},
            "claim": {"type": "string"},
            "operation": {"type": "string"},
            "computed_value": {"type": "string"},
            "status": {"type": "string"},
        },
        "required": ["step", "claim", "operation", "computed_value", "status"],
        "additionalProperties": False,
    }
    drift_schema = {
        "type": "object",
        "properties": {
            "claim": {"type": "string"},
            "claimed_value": {"type": "string"},
            "verified_value": {"type": "string"},
            "severity": {"type": "string"},
            "explanation": {"type": "string"},
        },
        "required": ["claim", "claimed_value", "verified_value", "severity", "explanation"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "verdict": {"type": "string"},
            "confidence": {"type": "number"},
            "summary": {"type": "string"},
            "baseline_answer": {"type": "string"},
            "verified_answer": {"type": "string"},
            "corrected_reasoning": {"type": "string"},
            "symbolic_steps": {"type": "array", "items": step_schema},
            "drift_reports": {"type": "array", "items": drift_schema},
            "audit_notes": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "verdict",
            "confidence",
            "summary",
            "baseline_answer",
            "verified_answer",
            "corrected_reasoning",
            "symbolic_steps",
            "drift_reports",
            "audit_notes",
        ],
        "additionalProperties": False,
    }


def normalize_analysis(raw: Dict[str, Any]) -> Dict[str, Any]:
    verdict = clean_verdict(str(raw.get("verdict") or "Uncertain"))
    confidence = clamp_float(raw.get("confidence"), default=0.5)
    symbolic_steps = normalize_steps(raw.get("symbolic_steps"))
    drift_reports = normalize_drifts(raw.get("drift_reports"))
    if drift_reports and verdict == "Supported":
        verdict = "Drift Detected"
    return {
        "verdict": verdict,
        "confidence": confidence,
        "summary": str(raw.get("summary") or "Reasoning audit completed."),
        "baseline_answer": str(raw.get("baseline_answer") or ""),
        "verified_answer": str(raw.get("verified_answer") or ""),
        "corrected_reasoning": str(raw.get("corrected_reasoning") or ""),
        "symbolic_steps": symbolic_steps,
        "drift_reports": drift_reports,
        "audit_notes": normalize_string_list(raw.get("audit_notes")),
        "aggregate": {
            "reasoning_depth": len(symbolic_steps),
            "drift_frequency": len(drift_reports),
            "supported_steps": sum(1 for step in symbolic_steps if step["status"].lower() == "supported"),
        },
    }


def normalize_steps(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    rows = []
    for idx, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "step": coerce_int(item.get("step"), default=idx),
                "claim": str(item.get("claim") or ""),
                "operation": str(item.get("operation") or ""),
                "computed_value": str(item.get("computed_value") or ""),
                "status": str(item.get("status") or "Uncertain"),
            }
        )
    return rows


def normalize_drifts(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    rows = []
    for item in value:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "claim": str(item.get("claim") or ""),
                "claimed_value": str(item.get("claimed_value") or ""),
                "verified_value": str(item.get("verified_value") or ""),
                "severity": str(item.get("severity") or "medium"),
                "explanation": str(item.get("explanation") or ""),
            }
        )
    return rows


def write_demo_result(result: Dict[str, Any]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    path = RESULTS_DIR / f"demo_{stamp}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    result["artifact_path"] = str(path)
    return path


def load_env_file(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def extract_response_text(response_json: Dict[str, Any]) -> str:
    texts = []
    for item in response_json.get("content", []):
        if isinstance(item, dict) and item.get("type") == "text":
            texts.append(str(item.get("text", "")))
    return "\n".join(texts).strip()


def parse_json_object(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}


def clean_verdict(value: str) -> str:
    lower = value.strip().lower()
    if "drift" in lower or "contradict" in lower:
        return "Drift Detected"
    if "correct" in lower:
        return "Corrected"
    if "support" in lower or "valid" in lower:
        return "Supported"
    return "Uncertain"


def clamp_float(value: Any, default: float = 0.5) -> float:
    try:
        val = float(value)
    except (TypeError, ValueError):
        val = default
    return max(0.0, min(1.0, val))


def coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]
