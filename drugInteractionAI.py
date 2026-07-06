"""
drugInteractionAI.py
====================
Drug Interaction orchestrator — mirrors the pattern of soapAI.py.

Two-layer architecture (per CDS document):
  Layer 1 — rxnav_service.build_rxnav_report()
             RxNorm  → resolves drug names to RxCUI codes
             RxNav   → authoritative interaction pairs (NLM database)
             OpenFDA → FDA-approved label warnings and adverse reactions

  Layer 2 — Local Ollama clinical pharmacology agent
             Adds patient-specific context to Layer 1 results
             Scores relevance, flags special populations, checks allergies
             Frames output as "Consider..." for the AI Alert window

Called from main.py immediately after generate_soap completes.
Never blocks or raises — always returns a result (or graceful error).
"""

import os
import json
import re
import httpx
from dotenv import load_dotenv
from agents import Agent, Runner
from ioschemas.SoapSchemas import SoapGenerationPayload, SoapNoteOutput
from services.rxnav_service import build_rxnav_report
from aiprompts.druginteraction import (
    get_system_prompt,
    generate_drug_interaction_prompt,
    extract_medications_from_plan,
)

load_dotenv(override=True)

# Mirror the same flag used in main.py
USE_OPENAI      = os.getenv("USE_OPENAI", "true").lower() in ("true", "1", "yes")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3")


async def run_drug_interaction_check(
    soap_result: SoapNoteOutput,
    validated_request: SoapGenerationPayload
) -> dict:
    """
    Full two-layer drug interaction check. Called after generate_soap completes.

    Args:
        soap_result:       The completed SoapNoteOutput from soapAI / soapOllama.
        validated_request: The original validated SoapGenerationPayload (provides
                           patient_context → active_medications and allergies).

    Returns:
        A drug interaction report dict ready to be placed in the generate_soap
        response under the key "drug_interaction".

    Error handling:
        This function NEVER raises. Any failure returns a structured error dict
        so the SOAP response is never blocked.
    """
    session_id = validated_request.ambient_details.session_id or "unknown"

    try:
        # ── Quick guard: are there any medications in the plan? ──────────────
        new_meds = extract_medications_from_plan(soap_result.plan)

        if not new_meds:
            print(f"💊 Drug interaction: no medications identified in plan for session {session_id}.")
            return {
                "status":  "skipped",
                "reason":  "No medications identified in this SOAP plan.",
                "data":    _empty_report()
            }

        print(
            f"💊 Drug interaction check starting for session {session_id} "
            f"— {len(new_meds)} newly prescribed med(s)."
        )

        # ── LAYER 1: RxNav + OpenFDA (authoritative DB lookup) ───────────────
        print(f"  [Layer 1] Querying RxNav / OpenFDA for session {session_id}...")
        rxnav_report = await build_rxnav_report(new_meds)
        print(
            f"  [Layer 1] Complete — status: {rxnav_report['layer1_status']}, "
            f"{len(rxnav_report['interactions'])} pair(s)."
        )

        # ── LAYER 2: Clinical context agent (OpenAI or Ollama) ───────────────
        print(f"  [Layer 2] Building AI context prompt for session {session_id}...")
        patient_ctx = validated_request.patient_context

        user_prompt = generate_drug_interaction_prompt(
            soap_plan=soap_result.plan,
            rxnav_report=rxnav_report,
            active_medications=patient_ctx.active_medications,
            allergies=patient_ctx.allergies,
            patient_context=patient_ctx.model_dump()
        )

        if USE_OPENAI:
            print(f"  [Layer 2] Routing to OpenAI (gpt-4o-mini) for session {session_id}...")
            drug_agent = Agent(
                name="Clinical Pharmacology Safety Agent",
                instructions=get_system_prompt(),
                model="gpt-4o-mini",
            )
            run_result = await Runner.run(drug_agent, user_prompt)
            raw_output = run_result.final_output
        else:
            print(
                f"  [Layer 2] Routing to Ollama ({OLLAMA_MODEL}) for session {session_id}..."
            )
            timeout_cfg = httpx.Timeout(180.0, read=None)
            async with httpx.AsyncClient(timeout=timeout_cfg) as http_client:
                resp = await http_client.post(
                    f"{OLLAMA_BASE_URL}/api/chat",
                    json={
                        "model":   OLLAMA_MODEL,
                        "messages": [
                            {"role": "system", "content": get_system_prompt()},
                            {"role": "user",   "content": user_prompt}
                        ],
                        "format": "json",
                        "stream": False,
                        "options": {
                            "temperature": 0.1,
                            "num_predict": 4096  # Drug interaction output is large — needs room for all sections
                        }
                    }
                )
                resp.raise_for_status()
                raw_output = resp.json()["message"]["content"]

        # Strip any accidental markdown fences the model may emit, then repair
        if isinstance(raw_output, str):
            interaction_report = _parse_json_robust(raw_output, session_id)
        else:
            interaction_report = raw_output  # already a dict (OpenAI structured output)

        engine_label = "OpenAI/gpt-4o-mini" if USE_OPENAI else f"Ollama/{OLLAMA_MODEL}"
        print(
            f"✅ Drug interaction complete for session {session_id} — "
            f"safety level: {interaction_report.get('overall_safety_level', 'unknown')}, "
            f"active alerts: {interaction_report.get('active_alerts_count', '?')}."
            f" [engine: {engine_label}]"
        )

        return {
            "status":            "success",
            "layer1_status":     rxnav_report["layer1_status"],
            "medications_found": len(new_meds),
            "unresolved_drugs":  rxnav_report.get("unresolved_drugs", []),
            "data":              interaction_report
        }

    except json.JSONDecodeError as je:
        print(f"  ✗ Drug interaction JSON parse error for session {session_id}: {je}")
        return {
            "status": "parse_error",
            "reason": f"AI agent returned non-JSON output: {str(je)}",
            "data":   _empty_report()
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  ✗ Drug interaction check failed for session {session_id}: {e}")
        return {
            "status": "error",
            "reason": str(e),
            "data":   _empty_report()
        }


def _extract_json_block(s: str) -> str:
    """
    Extract the outermost {...} block from a string.
    Handles stray prose before/after the JSON that small models sometimes emit.
    """
    start = s.find("{")
    if start == -1:
        return s
    # Walk from end to find matching closing brace
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
    """
    Best-effort repair of common small-model JSON errors:
      1. Extract outermost {} block (removes stray prose)
      2. Trailing commas before } or ]
      3. Truncated output — close open strings, arrays, and objects
    """
    # 0. Extract just the JSON block
    s = _extract_json_block(s).strip()

    # 1. Remove trailing commas (,  followed only by whitespace then } or ])
    s = re.sub(r',\s*([}\]])', r'\1', s)

    # 2. Repair truncation: figure out what's still open
    if not s.endswith("}"):
        # Close any open string
        last_open = s.rfind("{")
        if last_open != -1:
            tail = s[last_open:]
            unescaped_q = sum(
                1 for i, c in enumerate(tail)
                if c == '"' and (i == 0 or tail[i - 1] != '\\')
            )
            if unescaped_q % 2 != 0:
                s += '"'
        # Close open arrays / objects
        open_braces   = s.count("{") - s.count("}")
        open_brackets = s.count("[") - s.count("]")
        s += "]" * max(open_brackets, 0)
        s += "}" * max(open_braces, 0)

    # 3. Re-apply trailing comma removal after closing brackets were added
    s = re.sub(r',\s*([}\]])', r'\1', s)

    return s


def _parse_json_robust(raw: str, session_id: str = "?") -> dict:
    """Try plain json.loads first; if it fails, extract+repair and retry; else raise."""
    # Pass 1: direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Pass 2: strip markdown fences, then extract block + repair
    cleaned = (
        raw.strip()
        .lstrip("```json").lstrip("```")
        .rstrip("```").strip()
    )
    repaired = _repair_json(cleaned)
    try:
        result = json.loads(repaired)
        print(f"  ⚠️  JSON repaired for session {session_id} (small-model truncation/trailing-comma).")
        return result
    except json.JSONDecodeError as je:
        raise je  # re-raise so the caller's except block logs it properly


def _empty_report() -> dict:
    """Returns a safe empty report structure for error / skip cases."""
    return {
        "medications_analyzed":    [],
        "drug_drug_interactions":  [],
        "side_effects":            [],
        "dose_warnings":           [],
        "allergy_alerts":          [],
        "special_population_flags":[],
        "overall_safety_level":    "SAFE",
        "active_alerts_count":     0,
        "summary":                 "Drug interaction check was not completed.",
        "report_confidence":       0.0
    }
