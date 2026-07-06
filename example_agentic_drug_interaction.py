"""
example_agentic_drug_interaction.py
=====================================
THIS IS AN EXAMPLE ONLY — it does not replace or modify drugInteractionAI.py.

It shows how the existing drug interaction service could be converted from a
fixed pipeline into a true agentic pattern using the OpenAI Agents SDK.

──────────────────────────────────────────────────────────────────────────────
CURRENT PATTERN (drugInteractionAI.py)
──────────────────────────────────────────────────────────────────────────────
    run_drug_interaction_check()
        │
        ├─► Layer 1: build_rxnav_report()   ← always called, in fixed order
        │       ├─ resolve_rxcui()
        │       ├─ get_interactions()
        │       └─ get_fda_label_warnings()
        │
        └─► Layer 2: LLM (gpt-4o-mini)     ← always called after Layer 1

    Problem: The LLM is just a text processor at the end of a hardcoded chain.
    It cannot decide to skip a step, retry a failed lookup, or call a tool again
    with different parameters.

──────────────────────────────────────────────────────────────────────────────
AGENTIC PATTERN (this file)
──────────────────────────────────────────────────────────────────────────────
    DrugSafetyAgent (gpt-4o)
        │
        │  Has access to 3 registered tools:
        ├─► tool: resolve_drug_to_rxcui()        ← wraps resolve_rxcui()
        ├─► tool: check_known_interactions()     ← wraps get_interactions()
        └─► tool: get_fda_warnings()             ← wraps get_fda_label_warnings()

    The agent autonomously:
      1. Decides which drugs need RxCUI resolution
      2. Decides whether to look up interactions (skips if only 1 drug resolves)
      3. Fetches FDA labels in parallel for resolved drugs
      4. Writes the final safety report — informed by real tool results
      5. Can retry a failed lookup with a different drug name spelling
      6. Can reason: "Drug X didn't resolve — I'll flag it as unverified"

    Key difference: the LLM drives the workflow, not the Python code.
"""

import asyncio
import json
import os
import httpx
from typing import Optional
from dotenv import load_dotenv

# OpenAI Agents SDK — already in your requirements.txt (openai-agents 0.17.3)
from agents import Agent, Runner, function_tool

# ── Reuse your existing service functions exactly as-is ──────────────────────
# Nothing is changed in the original files. We just import the building blocks.
from services.rxnav_service import (
    resolve_rxcui,
    get_interactions,
    get_fda_label_warnings,
)
from aiprompts.druginteraction import get_system_prompt

load_dotenv(override=True)

# ── Shared HTTP client (module-level, reused across tool calls) ───────────────
_http_client: Optional[httpx.AsyncClient] = None

async def _get_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=5.0))
    return _http_client


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Wrap existing service functions as @function_tool
#
# @function_tool is the only addition. The underlying logic (resolve_rxcui,
# get_interactions, get_fda_label_warnings) is UNCHANGED — we call them as-is.
# ══════════════════════════════════════════════════════════════════════════════

@function_tool
async def resolve_drug_to_rxcui(drug_name: str) -> str:
    """
    Resolves a drug name to its RxNorm Concept Unique Identifier (RxCUI).
    Use this first for every drug before checking interactions or FDA labels.
    Returns the RxCUI string, or 'NOT_FOUND' if the drug cannot be resolved.

    Args:
        drug_name: Generic or brand drug name (e.g. 'amoxicillin', 'Tylenol')
    """
    client = await _get_client()
    rxcui = await resolve_rxcui(drug_name, client)          # ← your existing function
    return rxcui if rxcui else "NOT_FOUND"


@function_tool
async def check_known_interactions(rxcuis_json: str) -> str:
    """
    Checks known drug-drug interaction pairs for a list of RxCUI codes.
    Requires at least 2 resolved RxCUI codes to return meaningful results.
    Returns a JSON array of interaction pairs from the NLM RxNav database.

    Args:
        rxcuis_json: JSON array of RxCUI strings e.g. '["1049502", "41493"]'
    """
    rxcuis = json.loads(rxcuis_json)
    client = await _get_client()
    interactions = await get_interactions(rxcuis, client)   # ← your existing function
    return json.dumps(interactions)


@function_tool
async def get_fda_warnings(rxcui: str, drug_name: str) -> str:
    """
    Fetches FDA-approved label data for a single drug: boxed warnings,
    adverse reactions, contraindications, and drug interaction sections.
    Call this for each resolved drug to get authoritative safety information.

    Args:
        rxcui:     The RxCUI code (from resolve_drug_to_rxcui)
        drug_name: Human-readable drug name (used as fallback if rxcui fails)
    """
    client = await _get_client()
    label = await get_fda_label_warnings(rxcui, drug_name, client)  # ← your existing function
    return json.dumps(label)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Create the agent with the tools registered
# ══════════════════════════════════════════════════════════════════════════════

drug_safety_agent = Agent(
    name="Drug Safety Agent",
    model="gpt-4o",
    instructions=f"""
{get_system_prompt()}

You are an autonomous drug safety agent. You have three tools available:
  - resolve_drug_to_rxcui    — resolves a drug name to its RxCUI code
  - check_known_interactions — fetches known interaction pairs from RxNav
  - get_fda_warnings         — fetches FDA label warnings per drug

Your workflow for a given list of medications:
  1. Call resolve_drug_to_rxcui for EACH drug. Note any that return NOT_FOUND.
  2. If 2 or more drugs resolved, call check_known_interactions with their RxCUIs.
  3. Call get_fda_warnings for EACH resolved drug.
  4. Synthesize all tool results into a final structured JSON safety report.

Important rules:
  - If a drug returns NOT_FOUND, flag it as "unverified" in your report — do not guess.
  - If a drug name looks like a typo or shorthand, try an alternate spelling before giving up.
  - Never fabricate interaction data. Only report what the tools returned.
  - Your final output must be valid JSON matching the report schema below.

Final output schema:
{{
  "medications_analyzed":     [{{ "name": str, "rxcui": str, "status": "resolved"|"unverified" }}],
  "drug_drug_interactions":   [{{ "drug_a": str, "drug_b": str, "description": str, "severity_label": str }}],
  "fda_warnings_summary":     [{{ "drug": str, "boxed_warning": str|null, "key_warnings": [str] }}],
  "overall_safety_level":     "SAFE" | "MONITOR" | "MAJOR" | "CONTRAINDICATED",
  "active_alerts_count":      int,
  "summary":                  str,
  "report_confidence":        float  // 0.0–1.0; lower if drugs were unresolved
}}
""",
    tools=[
        resolve_drug_to_rxcui,
        check_known_interactions,
        get_fda_warnings,
    ],
)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Entry point: replace run_drug_interaction_check() with the agent
# ══════════════════════════════════════════════════════════════════════════════

async def run_agentic_drug_check(medications: list[dict]) -> dict:
    """
    Drop-in replacement for run_drug_interaction_check() in drugInteractionAI.py.

    Instead of a hardcoded pipeline, the agent autonomously:
      - decides which tools to call
      - handles partial resolution (e.g. one drug not found in RxNorm)
      - retries with alternate drug name spellings if needed
      - synthesizes results into the same report structure

    Args:
        medications: List of dicts with at least 'generic_name' key.
                     Same format as extract_medications_from_plan() returns.

    Returns:
        Same dict structure as the original run_drug_interaction_check().
    """
    if not medications:
        return {"status": "skipped", "reason": "No medications provided.", "data": {}}

    # Build a plain-text prompt — the agent will call tools on its own
    med_list = "\n".join(
        f"  - {m.get('generic_name', 'unknown')} "
        f"({m.get('dose', '')} {m.get('route', '')} {m.get('frequency', '')})"
        for m in medications
    )

    prompt = f"""
Please perform a complete drug safety check for the following newly prescribed medications:

{med_list}

Use your tools to resolve each drug to an RxCUI, check for known interactions between
them, and retrieve FDA label warnings for each. Then produce a final JSON safety report.
"""

    try:
        result = await Runner.run(drug_safety_agent, prompt)

        # The agent's final output is its JSON report
        raw = result.final_output
        report = json.loads(raw) if isinstance(raw, str) else raw

        return {
            "status": "success",
            "data": report,
        }

    except Exception as e:
        return {
            "status": "error",
            "reason": str(e),
            "data": {},
        }
    finally:
        # Clean up the shared HTTP client
        global _http_client
        if _http_client and not _http_client.is_closed:
            await _http_client.aclose()
            _http_client = None


# ══════════════════════════════════════════════════════════════════════════════
# Quick smoke test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Same medication list format that extract_medications_from_plan() produces
    test_meds = [
        {"generic_name": "warfarin",    "dose": "5mg",  "route": "oral", "frequency": "daily"},
        {"generic_name": "aspirin",     "dose": "81mg", "route": "oral", "frequency": "daily"},
        {"generic_name": "amoxicillin", "dose": "500mg","route": "oral", "frequency": "TID"},
    ]

    result = asyncio.run(run_agentic_drug_check(test_meds))
    print(json.dumps(result, indent=2))
