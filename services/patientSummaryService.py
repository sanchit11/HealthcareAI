"""
services/patientSummaryService.py
===================================
Patient Summary orchestrator — mirrors the pattern of vitalTrendsService.py.

Receives a comprehensive patient snapshot payload, sends it to the clinical AI
agent, and returns a structured summary with risk stratification and prioritised
action items.

Called from main.py at the /patient_summary endpoint.
Never raises — always returns a result dict (or graceful error).
"""

import os
import json
import re
import httpx
from typing import Any
from dotenv import load_dotenv
from agents import Agent, Runner
from aiprompts.patient_summary import get_system_prompt, generate_patient_summary_prompt

load_dotenv(override=True)

USE_OPENAI      = os.getenv("USE_OPENAI", "true").lower() in ("true", "1", "yes")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3")


# ---------------------------------------------------------------------------
# PUBLIC ENTRY POINT
# ---------------------------------------------------------------------------

async def run_patient_summary(payload: dict) -> dict:
    """
    Generate a clinical patient summary for the current encounter.

    Args:
        payload: Validated PatientSummaryRequest as a plain dict.

    Returns:
        A structured patient summary dict ready to be returned by the
        /patient_summary endpoint.

    Error handling:
        Never raises. Any failure returns a structured error dict.
    """
    pid = payload.get("patient_info", {}).get("pid", "unknown")

    try:
        print(f"📋 Patient summary analysis starting for pid={pid}.")

        user_prompt = generate_patient_summary_prompt(payload)

        if USE_OPENAI:
            print(f"  [PatientSummary] Routing to OpenAI (gpt-4o-mini) for pid={pid}...")
            agent = Agent(
                name="Clinical Patient Summary Analyst",
                instructions=get_system_prompt(),
                model="gpt-4o-mini",
            )
            run_result = await Runner.run(agent, user_prompt)
            raw_output = run_result.final_output

        else:
            print(f"  [PatientSummary] Routing to Ollama ({OLLAMA_MODEL}) for pid={pid}...")
            timeout_cfg = httpx.Timeout(180.0, read=None)
            async with httpx.AsyncClient(timeout=timeout_cfg) as http_client:
                resp = await http_client.post(
                    f"{OLLAMA_BASE_URL}/api/chat",
                    json={
                        "model": OLLAMA_MODEL,
                        "messages": [
                            {"role": "system", "content": get_system_prompt()},
                            {"role": "user",   "content": user_prompt},
                        ],
                        "format": "json",
                        "stream": False,
                        "options": {
                            "temperature": 0.1,
                            "num_predict": 2048,
                        },
                    },
                )
                resp.raise_for_status()
                raw_output = resp.json()["message"]["content"]

        # Parse — strip accidental markdown fences, repair truncation
        if isinstance(raw_output, str):
            report = _parse_json_robust(raw_output, pid)
        else:
            report = raw_output  # already a dict (OpenAI structured output)

        engine_label = "OpenAI/gpt-4o-mini" if USE_OPENAI else f"Ollama/{OLLAMA_MODEL}"
        risk_label = report.get("risk_level", "unknown") if isinstance(report, dict) else "unknown"
        print(
            f"✅ Patient summary complete for pid={pid} — "
            f"risk level: {risk_label}. [engine: {engine_label}]"
        )

        return report

    except json.JSONDecodeError as je:
        print(f"  ✗ Patient summary JSON parse error for pid={pid}: {je}")
        return _error_response(pid, f"AI agent returned non-JSON output: {str(je)}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  ✗ Patient summary analysis failed for pid={pid}: {e}")
        return _error_response(pid, str(e))


# ---------------------------------------------------------------------------
# JSON HELPERS  (mirrors vitalTrendsService.py pattern)
# ---------------------------------------------------------------------------

def _extract_json_block(s: str) -> str:
    """Extract the outermost {...} block from a string."""
    start = s.find("{")
    if start == -1:
        return s
    depth = 0
    in_string = False
    escape_next = False
    end = start
    for i, ch in enumerate(s[start:], start=start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
        elif not in_string:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
    return s[start:end + 1]


def _repair_json(s: str) -> str:
    """Best-effort repair of trailing commas and truncated output."""
    s = _extract_json_block(s).strip()
    s = re.sub(r',\s*([}\]])', r'\1', s)
    if not s.endswith("}"):
        last_open = s.rfind("{")
        if last_open != -1:
            tail = s[last_open:]
            unescaped_q = sum(
                1 for i, c in enumerate(tail)
                if c == '"' and (i == 0 or tail[i - 1] != '\\')
            )
            if unescaped_q % 2 != 0:
                s += '"'
        open_braces   = s.count("{") - s.count("}")
        open_brackets = s.count("[") - s.count("]")
        s += "]" * max(open_brackets, 0)
        s += "}" * max(open_braces, 0)
    s = re.sub(r',\s*([}\]])', r'\1', s)
    return s


def _parse_json_robust(raw: str, pid: Any = "?") -> dict:
    """Try plain json.loads first; if it fails, extract + repair and retry."""
    raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    repaired = _repair_json(raw)
    try:
        result = json.loads(repaired)
        print(f"  [PatientSummary] JSON repaired successfully for pid={pid}.")
        return result
    except json.JSONDecodeError as je:
        raise json.JSONDecodeError(
            f"Could not parse agent output even after repair: {je.msg}",
            je.doc,
            je.pos,
        )


def _error_response(pid: Any, reason: str) -> dict:
    return {
        "status":            "error",
        "reason":            reason,
        "clinical_narrative": "Summary could not be completed due to an internal error.",
        "risk_level":        "unknown",
        "risk_score":        0,
        "risk_factors":      [],
        "priority_actions":  [{"priority": "high", "text": "Review patient data manually — automated summary unavailable."}],
        "confidence":        0.0,
    }
