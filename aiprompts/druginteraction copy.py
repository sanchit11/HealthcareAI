"""
aiprompts/druginteraction.py
=============================
Layer 2 — AI Clinical Context Prompts for Drug Interaction Analysis.

This module provides the system prompt and user prompt builder for the
GPT-4o clinical pharmacology agent. It is called AFTER Layer 1 (rxnav_service)
has already fetched authoritative interaction pairs and FDA label data.

The LLM's role here is NOT to independently identify interactions — that is
Layer 1's job. The LLM adds:
  - Patient-specific clinical context  (e.g., "eGFR 42 raises bleeding risk")
  - Relevance scoring per alert         (suppress low-relevance alerts per CDS doc)
  - Plain-language "Consider also..."   framing for the AI Alert window
  - Special population flags            (renal, hepatic, elderly, pregnancy)
  - Allergy cross-reactivity reasoning  (penicillin → cephalosporin, etc.)
  - Overall safety level                (SAFE / CAUTION / UNSAFE)

CDS Hooks note (future):
  Maps to the `medication-prescribe` hook.
  context.draftOrders     → newly_prescribed_medications
  prefetch.medications    → existing_active_medications
  prefetch.allergies      → documented_allergies
  Layer 1 RxNav results   → rxnav_interaction_data (injected into prefetch)
"""

import re
import json
from typing import List, Dict, Any, Optional


# ---------------------------------------------------------------------------
# SYSTEM PROMPT — Layer 2 AI context agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a board-certified clinical pharmacist AI integrated into an EHR system.

Your role is to add patient-specific clinical context and relevance scoring to a set of
drug interaction pairs already identified by an authoritative FDA-sourced database (RxNav / OpenFDA).
You are NOT the primary interaction detector — the database results provided to you are ground truth.

Your responsibilities:
1. Interpret and contextualise each known interaction pair in light of this specific patient's
   clinical data (vitals, labs, problem list, age, allergies, active medications).
2. Score every alert with a relevance_score (0.0–1.0). Suppress alerts below 0.3 by marking
   them suppressed: true. This prevents alert fatigue as per the CDS document guidelines.
3. Frame every alert as "Consider..." or "Monitor for..." — never as a definitive clinical order.
   This keeps the product outside FDA Class II SaMD territory.
4. Check each newly prescribed drug against the patient's allergy list, including cross-reactive
   drug families (e.g., penicillin → cephalosporin ~1–2% cross-reactivity).
5. Flag special population concerns: renal dosing (use eGFR from labs if available), hepatic
   risk (check problem list for liver disease), elderly Beers Criteria (age ≥ 65),
   pregnancy / lactation (from demographics).
6. Assess each drug's prescribed dose against its standard therapeutic range.
7. Provide a 2–4 sentence plain-language summary for the ordering physician.
8. Assign an overall safety level: SAFE | CAUTION | UNSAFE.

Tone: Neutral, professional, physician-facing. Never alarmist, never dismissive.

CRITICAL RULES:
- Do NOT invent interactions not present in the provided RxNav data.
- If RxNav found no interactions, state that explicitly — do not fabricate pairs.
- If FDA label data is available, use it as the source for side effects and warnings.
- If a drug could not be resolved to RxCUI, note it as "unverified in RxNav" and
  use your training knowledge cautiously, clearly labelled as AI-generated (not DB-verified).
- Return ONLY valid JSON. No markdown, no prose outside the JSON structure.

Response JSON structure:
{
  "medications_analyzed": [
    {
      "generic_name": "string",
      "brand_name": "string",
      "dose": "string",
      "route": "string",
      "frequency": "string",
      "duration": "string",
      "indication": "string",
      "rxcui": "string or null",
      "source": "newly_prescribed | existing_active"
    }
  ],
  "drug_drug_interactions": [
    {
      "drug_a": "string",
      "drug_b": "string",
      "severity": "CONTRAINDICATED | MAJOR | MODERATE | MINOR | NONE",
      "mechanism": "string",
      "clinical_effect": "string",
      "patient_context_note": "string — why this matters for THIS patient specifically",
      "recommendation": "string — framed as Consider... or Monitor for...",
      "relevance_score": <float 0.0–1.0>,
      "suppressed": <bool — true if relevance_score < 0.3>,
      "db_verified": <bool — true if sourced from RxNav Layer 1>,
      "source": "string — e.g. RxNav/ONCHigh, RxNav/DrugBank, AI-generated"
    }
  ],
  "side_effects": [
    {
      "drug": "string",
      "common": ["string"],
      "serious": ["string"],
      "boxed_warning": "string or null",
      "relevant_to_patient": ["string — side effects that overlap with problem list"]
    }
  ],
  "dose_warnings": [
    {
      "drug": "string",
      "prescribed_dose": "string",
      "standard_range": "string",
      "status": "WITHIN_RANGE | CAUTION_HIGH_END | EXCEEDS_MAX_DOSE",
      "toxicity_profile": "string"
    }
  ],
  "allergy_alerts": [
    {
      "drug": "string",
      "allergen_matched": "string",
      "reaction_type": "string",
      "severity": "CONTRAINDICATED | CAUTION_CROSS_REACTIVITY",
      "recommendation": "string"
    }
  ],
  "special_population_flags": [
    {
      "flag_type": "RENAL | HEPATIC | PREGNANCY | LACTATION | ELDERLY_BEERS",
      "drug": "string",
      "detail": "string",
      "recommendation": "string — framed as Consider... or Monitor for..."
    }
  ],
  "overall_safety_level": "SAFE | CAUTION | UNSAFE",
  "active_alerts_count": <int — count of non-suppressed alerts>,
  "summary": "string — 2–4 sentence plain-language summary for the ordering physician",
  "report_confidence": <float 0.0–1.0>
}
"""


# ---------------------------------------------------------------------------
# MEDICATION EXTRACTOR — universal free-text plan parser
# ---------------------------------------------------------------------------

# Comprehensive medication terms (generic names, drug classes, clinical names).
# Multi-word terms are listed first so they are tried before single-word substrings.
_MEDICATION_TERMS: List[str] = [
    # Fluids / electrolytes (multi-word)
    "normal saline", "lactated ringer", "iv fluids", "intravenous fluids",
    "sodium bicarbonate", "potassium chloride", "calcium gluconate",
    "calcium chloride", "magnesium sulfate", "ferrous sulfate", "folic acid",
    # Antibiotics
    "amoxicillin", "azithromycin", "doxycycline", "ciprofloxacin",
    "levofloxacin", "ceftriaxone", "vancomycin", "metronidazole", "clindamycin",
    "trimethoprim", "sulfamethoxazole", "piperacillin", "tazobactam",
    "meropenem", "imipenem", "ampicillin", "nafcillin", "oxacillin",
    "penicillin", "cephalexin", "cefdinir", "cefazolin", "cefepime",
    "linezolid", "daptomycin", "gentamicin", "tobramycin", "nitrofurantoin",
    # Diabetes / metabolic
    "insulin", "metformin", "glipizide", "glimepiride", "glyburide",
    "sitagliptin", "saxagliptin", "linagliptin", "empagliflozin",
    "dapagliflozin", "canagliflozin", "liraglutide", "semaglutide",
    "dulaglutide", "pioglitazone", "rosiglitazone", "acarbose",
    "repaglinide", "nateglinide",
    # Fluids / electrolytes (single-word)
    "saline", "dextrose", "potassium", "bicarbonate", "magnesium",
    "albumin", "hetastarch",
    # Cardiovascular
    "aspirin", "clopidogrel", "warfarin", "heparin", "enoxaparin",
    "rivaroxaban", "apixaban", "dabigatran", "atorvastatin", "rosuvastatin",
    "simvastatin", "metoprolol", "carvedilol", "bisoprolol", "atenolol",
    "lisinopril", "enalapril", "ramipril", "losartan", "valsartan",
    "amlodipine", "diltiazem", "verapamil", "hydralazine", "nitroprusside",
    "nitroglycerin", "furosemide", "torsemide", "spironolactone",
    "hydrochlorothiazide", "digoxin", "amiodarone", "adenosine", "bumetanide",
    # Pulmonary
    "albuterol", "ipratropium", "tiotropium", "salmeterol", "formoterol",
    "budesonide", "fluticasone", "beclomethasone", "montelukast",
    "theophylline", "roflumilast", "oxygen",
    # Pain / Anti-inflammatory / Steroids
    "acetaminophen", "ibuprofen", "naproxen", "ketorolac", "morphine",
    "oxycodone", "hydrocodone", "hydromorphone", "fentanyl", "tramadol",
    "codeine", "methylprednisolone", "prednisone", "dexamethasone",
    "prednisolone", "celecoxib", "indomethacin", "diclofenac",
    # GI
    "omeprazole", "pantoprazole", "esomeprazole", "lansoprazole",
    "ranitidine", "famotidine", "ondansetron", "metoclopramide",
    "promethazine", "sucralfate", "docusate", "bisacodyl", "lactulose",
    # Neuro / Psych
    "diazepam", "midazolam", "lorazepam", "alprazolam", "clonazepam",
    "haloperidol", "olanzapine", "quetiapine", "risperidone", "sertraline",
    "fluoxetine", "paroxetine", "escitalopram", "citalopram", "venlafaxine",
    "duloxetine", "bupropion", "mirtazapine", "lithium", "valproate",
    "carbamazepine", "phenytoin", "levetiracetam", "gabapentin", "pregabalin",
    # Thyroid
    "levothyroxine", "methimazole", "propylthiouracil",
    # Emergency / Vasopressors
    "epinephrine", "norepinephrine", "dopamine", "dobutamine", "vasopressin",
    "naloxone", "flumazenil", "atropine",
    # Vitamins / supplements
    "thiamine", "cyanocobalamin", "zinc",
]

# Build a single compiled pattern from all terms (longest-first to avoid
# partial matches, e.g. "normal saline" before "saline").
_MEDS_PATTERN = re.compile(
    r'\b(?:' + '|'.join(
        re.escape(m) for m in sorted(_MEDICATION_TERMS, key=len, reverse=True)
    ) + r')\b',
    re.IGNORECASE
)

# Extracts dose / route abbreviation / frequency / duration from nearby text.
_DOSE_DETAIL_PATTERN = re.compile(
    r"(?P<dose>[\d.]+\s*(?:mg|mcg|g|mEq|units?|IU|ml|%)[^\s]*)\s*"
    r"(?P<route>PO|IV|IM|SC|SQ|SL|PR|TOP|INH|NAS|OPH|OTI|TRANS)?\s*"
    r"(?P<frequency>QD|QHS|BID|TID|QID|Q\d+H|PRN|once daily|twice daily|"
    r"three times daily|four times daily|every \d+ hours?|daily|weekly)?\s*"
    r"(?:x\s*(?P<duration>\d+\s*(?:day|week|month)s?))?",
    re.IGNORECASE
)

# Matches prose route descriptions that appear near the medication name.
_ROUTE_KEYWORDS_PATTERN = re.compile(
    r'\b(intravenous(?:ly)?|IV\b|oral(?:ly)?|PO\b|intramuscular(?:ly)?|IM\b|'
    r'subcutaneous(?:ly)?|SC\b|SQ\b|inhal(?:ed?|ation)|nebulized?|'
    r'topical(?:ly)?)\b',
    re.IGNORECASE
)

_ROUTE_MAP: Dict[str, str] = {
    "intravenous": "IV", "intravenously": "IV", "iv": "IV",
    "oral": "PO", "orally": "PO", "po": "PO",
    "intramuscular": "IM", "intramuscularly": "IM", "im": "IM",
    "subcutaneous": "SC", "subcutaneously": "SC", "sc": "SC", "sq": "SC",
    "inhale": "INH", "inhaled": "INH", "inhalation": "INH",
    "nebulize": "INH", "nebulized": "INH",
    "topical": "TOP", "topically": "TOP",
}

# Matches indication phrases such as "for dehydration" or "to correct blood sugar".
_INDICATION_PATTERN = re.compile(
    r'(?:for|to treat|to correct|to manage|due to|secondary to|indicated for)'
    r'\s+([^.,;\n]{3,80})',
    re.IGNORECASE
)


def extract_medications_from_plan(plan_text: str) -> List[Dict[str, str]]:
    """
    Universal medication extractor — works on any free-text SOAP plan format.

    Scans the plan for known medication terms (generics, drug classes, clinical
    names) and extracts dose, route, frequency, and indication from the
    surrounding sentence context.  No specific line format or prefix required —
    handles prose steps ("Administer insulin IV"), numbered lists, and legacy
    structured "Rx:" entries equally.

    Returns a deduplicated list of medication dicts tagged source='newly_prescribed'.
    """
    if not plan_text:
        return []

    medications: List[Dict[str, str]] = []
    seen: set = set()

    for match in _MEDS_PATTERN.finditer(plan_text):
        generic = match.group(0).strip().lower()

        # Deduplicate — skip if already captured
        if generic in seen:
            continue
        seen.add(generic)

        # Context windows used for attribute extraction
        pre_start  = max(0, match.start() - 40)
        post_end   = min(len(plan_text), match.end() + 160)
        pre_text   = plan_text[pre_start:match.start()]
        post_text  = plan_text[match.end():post_end]
        full_ctx   = plan_text[pre_start:post_end]

        dose = route = frequency = duration = indication = ""

        # 1. Structured dose/route/frequency from text immediately after the match
        detail = _DOSE_DETAIL_PATTERN.search(post_text)
        if detail:
            dose      = (detail.group("dose")      or "").strip()
            frequency = (detail.group("frequency") or "").strip()
            duration  = (detail.group("duration")  or "").strip()
            route_raw = (detail.group("route")     or "").strip()
            route     = _ROUTE_MAP.get(route_raw.lower(), route_raw.upper()) if route_raw else ""

        # 2. Prose route keyword if no abbreviation found above
        if not route:
            # Check post-text first, then pre-text (e.g. "administer IV insulin")
            for search_zone in (post_text, pre_text):
                rm = _ROUTE_KEYWORDS_PATTERN.search(search_zone)
                if rm:
                    route = _ROUTE_MAP.get(rm.group(0).lower(), rm.group(0).upper())
                    break

        # 3. Indication from surrounding sentence
        ind_match = _INDICATION_PATTERN.search(full_ctx)
        if ind_match:
            indication = ind_match.group(1).strip().rstrip(".,;")

        medications.append({
            "generic_name": generic,
            "brand_name":   "",
            "dose":         dose,
            "route":        route,
            "frequency":    frequency,
            "duration":     duration,
            "indication":   indication,
            "source":       "newly_prescribed",
        })

    return medications


# ---------------------------------------------------------------------------
# HELPERS — format patient context fields
# ---------------------------------------------------------------------------

def _format_active_medications(active_meds: List[Any]) -> List[Dict[str, str]]:
    formatted = []
    for m in active_meds:
        if hasattr(m, "model_dump"):
            m = m.model_dump()
        formatted.append({
            "generic_name": m.get("drug_name", "").lower(),
            "brand_name":   "",
            "dose":         m.get("dosage", ""),
            "route":        m.get("route", ""),
            "frequency":    m.get("frequency", ""),
            "duration":     "ongoing",
            "indication":   m.get("indication", ""),
            "source":       "existing_active"
        })
    return formatted


def _format_allergies(allergies: List[Any]) -> List[Dict[str, str]]:
    formatted = []
    for a in allergies:
        if hasattr(a, "model_dump"):
            a = a.model_dump()
        formatted.append({
            "allergen":      a.get("allergen", ""),
            "allergen_type": a.get("allergen_type", ""),
            "reaction":      a.get("reaction", ""),
            "severity":      a.get("severity", "")
        })
    return formatted


def _extract_special_population_context(patient_context: Dict[str, Any]) -> Dict[str, Any]:
    """Pulls age, sex, problem list, and relevant labs for special-population flags."""
    if not patient_context:
        return {}
    demo = patient_context.get("demographics", {})
    return {
        "age":          demo.get("age"),
        "sex":          demo.get("sex", ""),
        "problem_list": [p.get("display", "") for p in patient_context.get("problem_list", [])],
        "recent_labs":  [
            {
                "test":     l.get("test", ""),
                "value":    l.get("value"),
                "unit":     l.get("unit", ""),
                "abnormal": l.get("abnormal", "")
            }
            for l in patient_context.get("labs_recent", [])
        ]
    }


# ---------------------------------------------------------------------------
# PROMPT BUILDERS
# ---------------------------------------------------------------------------

def get_system_prompt() -> str:
    """Returns the Layer 2 system instructions for the drug interaction agent."""
    return SYSTEM_PROMPT


def generate_drug_interaction_prompt(
    soap_plan: str,
    rxnav_report: Optional[Dict[str, Any]] = None,
    active_medications: Optional[List[Any]] = None,
    allergies: Optional[List[Any]] = None,
    patient_context: Optional[Dict[str, Any]] = None
) -> str:
    """
    Builds the user-turn prompt for the Layer 2 drug interaction agent.

    Args:
        soap_plan:          The `plan` field from a completed SoapNoteOutput.
        rxnav_report:       Output of rxnav_service.build_rxnav_report() — Layer 1 results.
                            If None, the LLM will note it and use training knowledge only
                            (clearly labelled as unverified).
        active_medications: Patient's existing active meds from patient_context.
        allergies:          Patient's documented allergies from patient_context.
        patient_context:    Full patient_context dict for special-population checks.

    Returns:
        Fully assembled user prompt string for LLM submission.
    """
    # Extract newly prescribed meds from plan
    new_meds = extract_medications_from_plan(soap_plan)

    if not new_meds:
        return (
            "No Rx entries were identified in the SOAP plan. "
            "Return JSON with all interaction lists empty and "
            "overall_safety_level set to 'SAFE'. "
            "Set summary to: 'No pharmacologic therapy was prescribed in this encounter.'"
        )

    # Format supporting data
    existing_meds  = _format_active_medications(active_medications or [])
    allergy_list   = _format_allergies(allergies or [])
    special_ctx    = _extract_special_population_context(patient_context or {})

    # Summarise Layer 1 results for the prompt
    layer1_summary = None
    if rxnav_report:
        layer1_summary = {
            "layer1_status": rxnav_report.get("layer1_status", "unknown"),
            "unresolved_drugs": rxnav_report.get("unresolved_drugs", []),
            "rxnav_interaction_pairs": rxnav_report.get("interactions", []),
            "fda_label_data": [
                {
                    "drug":             lbl.get("drug"),
                    "rxcui":            lbl.get("rxcui"),
                    "boxed_warning":    lbl.get("boxed_warning"),
                    # Truncate long label text to avoid token overflow
                    "warnings":         [w[:500] for w in lbl.get("warnings", [])[:3]],
                    "adverse_reactions":[a[:500] for a in lbl.get("adverse_reactions", [])[:3]],
                    "contraindications":[c[:400] for c in lbl.get("contraindications", [])[:2]],
                }
                for lbl in rxnav_report.get("fda_labels", [])
            ],
            "resolved_medications": rxnav_report.get("resolved_medications", [])
        }

    prompt_payload = {
        "task": (
            "Using the Layer 1 RxNav database results provided below as ground truth, "
            "add patient-specific clinical context to each interaction, score relevance, "
            "check allergies and special populations, and return a structured "
            "DrugInteractionReport JSON as defined in your system instructions. "
            "Do NOT invent interactions not present in the RxNav data."
        ),
        "newly_prescribed_medications":  new_meds,
        "existing_active_medications":   existing_meds,
        "documented_allergies":          allergy_list,
        "patient_special_population_context": special_ctx,
        "layer1_rxnav_results":          layer1_summary  # None if Layer 1 was skipped
    }

    return (
        "Analyse the following clinical and drug data. "
        "The RxNav Layer 1 results are the authoritative source for known interactions. "
        "Your role is to add clinical context, score relevance, and produce the JSON report.\n\n"
        f"{json.dumps(prompt_payload, indent=2)}"
    )
