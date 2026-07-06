"""
services/rxnav_service.py
=========================
Layer 1 — Authoritative Drug Interaction Lookup via NLM RxNav APIs.

APIs used (all free, no auth required):
  RxNorm  — resolves a drug name to its RxCUI code
    GET https://rxnav.nlm.nih.gov/REST/rxcui.json?name={name}&search=1

  RxNav Interaction List — returns known interaction pairs for a set of RxCUIs
    GET https://rxnav.nlm.nih.gov/REST/interaction/list.json?rxcuis={cui1}+{cui2}+...

  OpenFDA Drug Label — fetches FDA-approved label (warnings, adverse reactions)
    GET https://api.fda.gov/drug/label.json?search=openfda.rxcui:{rxcui}&limit=1

This service is called BEFORE the LLM (Layer 2). The LLM receives the
authoritative interaction data from here and only adds patient-specific context.

Architecture:
    SOAP plan → extract_medications_from_plan()
        → resolve_rxcui()          [RxNorm API — name to code]
        → get_interactions()       [RxNav API  — known pairs]
        → get_fda_label_warnings() [OpenFDA    — side effects / black box]
        → structured RxNavReport dict
        → GPT-4o context layer (drugInteractionAI.py)
"""

import asyncio
import httpx
from typing import List, Dict, Any, Optional

# ---------------------------------------------------------------------------
# API Base URLs
# ---------------------------------------------------------------------------
RXNORM_BASE   = "https://rxnav.nlm.nih.gov/REST"
RXNAV_BASE    = "https://rxnav.nlm.nih.gov/REST"
OPENFDA_BASE  = "https://api.fda.gov/drug"

# Shared async HTTP client settings
_TIMEOUT = httpx.Timeout(10.0, connect=5.0)
_HEADERS = {"Accept": "application/json"}


# ---------------------------------------------------------------------------
# CLINICAL TERM ALIASES
# Maps common free-text / shorthand names to their canonical RxNorm drug name.
# Terms mapped to None are too generic for RxCUI lookup and are skipped.
# ---------------------------------------------------------------------------

_RXNORM_ALIASES: Dict[str, Optional[str]] = {
    "iv fluids":          "sodium chloride",
    "intravenous fluids": "sodium chloride",
    "normal saline":      "sodium chloride",
    "saline":             "sodium chloride",
    "lactated ringer":    "lactated ringers solution",
    "ringer":             "lactated ringers solution",
    "dextrose":           "dextrose",
    "potassium":          "potassium chloride",
    "bicarbonate":        "sodium bicarbonate",
    "magnesium":          "magnesium sulfate",
    "calcium":            "calcium gluconate",
    "fluid":              None,   # too generic — skip
    "oxygen":             None,   # not an RxNorm drug
}


# ---------------------------------------------------------------------------
# STEP 1 — Resolve drug name → RxCUI
# ---------------------------------------------------------------------------

async def resolve_rxcui(drug_name: str, client: httpx.AsyncClient) -> Optional[str]:
    """
    Resolves a drug name to its RxNorm Concept Unique Identifier (RxCUI).

    Args:
        drug_name: Generic or brand drug name (e.g., "amoxicillin", "Tylenol")
        client:    Shared httpx.AsyncClient for connection pooling

    Returns:
        RxCUI string if found, None otherwise.
    """
    # Normalise clinical shorthand to canonical RxNorm name before lookup
    lookup_name = _RXNORM_ALIASES.get(drug_name.lower(), drug_name)
    if lookup_name is None:
        print(f"  ⏭ Skipping RxCUI lookup for generic term: {drug_name}")
        return None
    if lookup_name != drug_name:
        print(f"  🔁 Alias: '{drug_name}' → '{lookup_name}'")

    try:
        url = f"{RXNORM_BASE}/rxcui.json"
        resp = await client.get(url, params={"name": lookup_name, "search": "1"}, headers=_HEADERS)
        resp.raise_for_status()
        data = resp.json()

        ids = data.get("idGroup", {}).get("rxnormId", [])
        if ids:
            rxcui = ids[0]
            print(f"  ✓ RxCUI resolved: {drug_name} → {rxcui}")
            return rxcui

        print(f"  ⚠ No RxCUI found for: {drug_name}")
        return None

    except Exception as e:
        print(f"  ✗ RxCUI resolution failed for '{drug_name}': {e}")
        return None


async def resolve_all_rxcuis(
    medications: List[Dict[str, str]],
    client: httpx.AsyncClient
) -> List[Dict[str, Any]]:
    """
    Resolves RxCUIs for all medications concurrently.

    Args:
        medications: List of medication dicts (from extract_medications_from_plan)
        client:      Shared httpx.AsyncClient

    Returns:
        Same list with `rxcui` key added to each dict. Unresolved drugs get rxcui=None.
    """
    tasks = [resolve_rxcui(med["generic_name"], client) for med in medications]
    rxcuis = await asyncio.gather(*tasks)

    enriched = []
    for med, rxcui in zip(medications, rxcuis):
        enriched.append({**med, "rxcui": rxcui})

    return enriched


# ---------------------------------------------------------------------------
# STEP 2 — Fetch known drug-drug interaction pairs from RxNav
# ---------------------------------------------------------------------------

async def get_interactions(
    rxcuis: List[str],
    client: httpx.AsyncClient
) -> List[Dict[str, Any]]:
    """
    Fetches all known interaction pairs for a list of RxCUI codes from
    the RxNav Interaction API.

    Args:
        rxcuis: List of resolved RxCUI strings (None values are filtered out)
        client: Shared httpx.AsyncClient

    Returns:
        List of interaction dicts, each containing:
            drug_a, drug_b, description, severity_label, source
    """
    valid_cuis = [c for c in rxcuis if c]
    if len(valid_cuis) < 2:
        print("  ℹ Less than 2 resolved RxCUIs — skipping interaction lookup.")
        return []

    try:
        url  = f"{RXNAV_BASE}/interaction/list.json"
        params = {"rxcuis": " ".join(valid_cuis)}
        resp = await client.get(url, params=params, headers=_HEADERS)
        resp.raise_for_status()
        data = resp.json()

        interactions = []
        full_groups = data.get("fullInteractionTypeGroup", [])

        for group in full_groups:
            source_name = group.get("sourceName", "RxNav")
            for interaction_type in group.get("fullInteractionType", []):
                for pair in interaction_type.get("interactionPair", []):
                    concepts = pair.get("interactionConcept", [])
                    if len(concepts) < 2:
                        continue

                    drug_a = concepts[0].get("minConceptItem", {}).get("name", "Unknown")
                    drug_b = concepts[1].get("minConceptItem", {}).get("name", "Unknown")
                    description = pair.get("description", "")
                    severity    = pair.get("severity", "unknown")

                    interactions.append({
                        "drug_a":         drug_a.lower(),
                        "drug_b":         drug_b.lower(),
                        "description":    description,
                        "severity_label": severity,   # e.g. "N/A", "high", "moderate"
                        "source":         source_name  # e.g. "ONCHigh", "DrugBank"
                    })

        print(f"  ✓ RxNav returned {len(interactions)} interaction pair(s).")
        return interactions

    except Exception as e:
        print(f"  ✗ RxNav interaction fetch failed: {e}")
        return []


# ---------------------------------------------------------------------------
# STEP 3 — Fetch FDA label warnings and adverse reactions via OpenFDA
# ---------------------------------------------------------------------------

async def get_fda_label_warnings(
    rxcui: str,
    drug_name: str,
    client: httpx.AsyncClient
) -> Dict[str, Any]:
    """
    Fetches FDA-approved label data for a single drug, extracting:
      - warnings / black box warnings
      - adverse reactions
      - contraindications
      - drug interactions section from the label

    Args:
        rxcui:     RxCUI code for the drug
        drug_name: Human-readable name (used as fallback search)
        client:    Shared httpx.AsyncClient

    Returns:
        Dict with keys: drug, warnings, adverse_reactions, contraindications,
                        drug_interactions_label, boxed_warning
    """
    result = {
        "drug":                    drug_name,
        "rxcui":                   rxcui,
        "warnings":                [],
        "adverse_reactions":       [],
        "contraindications":       [],
        "drug_interactions_label": [],
        "boxed_warning":           None
    }

    try:
        # Search by RxCUI first, fall back to drug name
        search_query = (
            f"openfda.rxcui:{rxcui}"
            if rxcui
            else f"openfda.generic_name:{drug_name}"
        )
        url = f"{OPENFDA_BASE}/label.json"
        resp = await client.get(
            url,
            params={"search": search_query, "limit": "1"},
            headers=_HEADERS
        )
        resp.raise_for_status()
        data = resp.json()

        results = data.get("results", [])
        if not results:
            print(f"  ⚠ No FDA label found for: {drug_name} (rxcui={rxcui})")
            return result

        label = results[0]

        def _extract(field: str) -> List[str]:
            """Safely pull a label field (always a list in OpenFDA)."""
            return label.get(field, [])

        result["warnings"]                = _extract("warnings")
        result["adverse_reactions"]       = _extract("adverse_reactions")
        result["contraindications"]       = _extract("contraindications")
        result["drug_interactions_label"] = _extract("drug_interactions")
        result["boxed_warning"]           = (
            label["boxed_warning"][0] if label.get("boxed_warning") else None
        )

        print(f"  ✓ FDA label fetched for: {drug_name}")
        return result

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            print(f"  ⚠ FDA label not found for: {drug_name}")
        else:
            print(f"  ✗ FDA label fetch error for '{drug_name}': {e}")
        return result

    except Exception as e:
        print(f"  ✗ Unexpected error fetching FDA label for '{drug_name}': {e}")
        return result


# ---------------------------------------------------------------------------
# MAIN ENTRY POINT — Full Layer 1 report
# ---------------------------------------------------------------------------

async def build_rxnav_report(
    medications: List[Dict[str, str]]
) -> Dict[str, Any]:
    """
    Full Layer 1 pipeline. Takes extracted medications from the SOAP plan
    and returns a complete authoritative report from RxNorm + RxNav + OpenFDA.

    Args:
        medications: Output of extract_medications_from_plan() — list of dicts
                     with keys: generic_name, brand_name, dose, route,
                                frequency, duration, indication, source

    Returns:
        RxNavReport dict:
        {
            "resolved_medications": [...],    # with rxcui added
            "interactions": [...],            # RxNav known pairs
            "fda_labels": [...],              # OpenFDA warnings per drug
            "unresolved_drugs": [...],        # drugs with no RxCUI found
            "layer1_status": "ok" | "partial" | "no_meds"
        }
    """
    if not medications:
        return {
            "resolved_medications": [],
            "interactions":         [],
            "fda_labels":           [],
            "unresolved_drugs":     [],
            "layer1_status":        "no_meds"
        }

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:

        # Step 1 — resolve all drug names to RxCUIs in parallel
        print("🔍 Layer 1: Resolving RxCUIs...")
        resolved_meds = await resolve_all_rxcuis(medications, client)

        resolved   = [m for m in resolved_meds if m["rxcui"]]
        unresolved = [m["generic_name"] for m in resolved_meds if not m["rxcui"]]

        # Step 2 — fetch interaction pairs for all resolved RxCUIs in parallel
        print("🔍 Layer 1: Fetching interaction pairs from RxNav...")
        rxcui_list   = [m["rxcui"] for m in resolved]
        interactions = await get_interactions(rxcui_list, client)

        # Step 3 — fetch FDA labels for each resolved drug in parallel
        print("🔍 Layer 1: Fetching FDA labels from OpenFDA...")
        fda_tasks = [
            get_fda_label_warnings(m["rxcui"], m["generic_name"], client)
            for m in resolved
        ]
        fda_labels = await asyncio.gather(*fda_tasks)

    layer1_status = (
        "ok"      if resolved and not unresolved else
        "partial" if resolved and unresolved     else
        "no_meds"
    )

    report = {
        "resolved_medications": resolved,
        "interactions":         interactions,
        "fda_labels":           list(fda_labels),
        "unresolved_drugs":     unresolved,
        "layer1_status":        layer1_status
    }

    print(
        f"✅ Layer 1 complete — "
        f"{len(resolved)} resolved, "
        f"{len(unresolved)} unresolved, "
        f"{len(interactions)} interaction pair(s) found."
    )
    return report
