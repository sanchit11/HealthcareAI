"""
aiprompts/druginteraction.py
=============================
Layer 2 & Layer 3 — AI Clinical Context and Fallback Prompts for Drug Interaction Analysis.

This module provides the system prompt and user prompt builder for the
clinical pharmacology agent. It handles Layer 2 context addition when Layer 1 
database results are present, and seamlessly transitions into Layer 3 fallback 
using internal knowledge when database results are missing or unresolved.
"""

import re
import json
from typing import List, Dict, Any, Optional  # <--- Explicitly importing Optional to prevent NameError


# ---------------------------------------------------------------------------
# SYSTEM PROMPT — Layer 2 AI context / Layer 3 Fallback Agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a board-certified clinical pharmacist AI integrated into an EHR system.

Your primary role (Layer 2) is to add patient-specific clinical context and relevance scoring to a set of
drug interaction pairs already identified by an authoritative FDA-sourced database (RxNav / OpenFDA).

CRITICAL LAYER 3 FALLBACK RULE:
If the Layer 1 database results (`layer1_rxnav_results`) are missing, empty, or a drug could not be resolved (e.g. 'oxygen'),
you MUST activate "Layer 3 Fallback Mode". Use your internal medical training knowledge to analyze the 
medications, identify standard drug-drug interactions, and extract well-known side effects for basic drug names 
(e.g., insulin, iv fluids, common generic medications). 
When executing Layer 3 fallback sections, you MUST set "db_verified": false and "source": "AI-generated Fallback" 
inside the relevant JSON structures so the clinician knows it came from internal training rather than an external database.

CRITICAL JSON NESTING RULES (DO NOT DEVIATE):
1. `medications_analyzed` MUST ONLY contain basic drug identity details (generic_name, brand_name, dose, route, frequency, duration, indication, rxcui, source). 
2. DO NOT nest `drug_drug_interactions`, `side_effects`, `dose_warnings`, `allergy_alerts`, `special_population_flags`, `overall_safety_level`, or `summary` inside individual items of the `medications_analyzed` list. 
3. ALL evaluation arrays, safety levels, and summaries MUST be keys at the root level of the JSON object.

Tone: Neutral, professional, physician-facing. Never alarmist, never dismissive.

CRITICAL RULES:
- If RxNav database data is present, treat it as ground truth. Do not invent interactions contradicting it.
- IF RXNAV DATA IS MISSING OR EMPTY, DO NOT RETURN EMPTY ARRAYS. Activate Layer 3 instructions immediately 
  and populate the JSON fields (such as side_effects, drug_drug_interactions, or dose_warnings) based on 
  your core medical knowledge so basic drug safety information is never omitted.
- Return ONLY valid JSON. No markdown, no prose outside the JSON structure.

Response JSON structure (FOLLOW THIS ROOT LEVEL STRUCTURE EXACTLY):
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
      "db_verified": <bool — true if sourced from RxNav Layer 1, FALSE if generated via Layer 3 fallback>,
      "source": "string — e.g. RxNav/ONCHigh, RxNav/DrugBank, AI-generated Fallback"
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

_MEDS_PATTERN = re.compile(
    r'\b(?:' + '|'.join(
        re.escape(m) for m in sorted(_MEDICATION_TERMS, key=len, reverse=True)
    ) + r')\b',
    re.IGNORECASE
)

_DOSE_DETAIL_PATTERN = re.compile(
    r"(?P<dose>[\d.]+\s*(?:mg|mcg|g|mEq|units?|IU|ml|%)[^\s]*)\s*"
    r"(?P<route>PO|IV|IM|SC|SQ|SL|PR|TOP|INH|NAS|OPH|OTI|TRANS)?\s*"
    r"(?P<frequency>QD|QHS|BID|TID|QID|Q\d+H|PRN|once daily|twice daily|"
    r"three times daily|four times daily|every \d+ hours?|daily|weekly)?\s*"
    r"(?:x\s*(?P<duration>\d+\s*(?:day|week|month)s?))?",
    re.IGNORECASE
)

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

_INDICATION_PATTERN = re.compile(
    r'(?:for|to treat|to correct|to manage|due to|secondary to|indicated for)'
    r'\s+([^.,;\n]{3,80})',
    re.IGNORECASE
)


def extract_medications_from_plan(plan_text: str) -> List[Dict[str, str]]:
    """Scan free text SOAP plan for known medications and parse surrounding details."""
    if not plan_text:
        return []

    medications: List[Dict[str, str]] = []
    seen: set = set()

    for match in _MEDS_PATTERN.finditer(plan_text):
        generic = match.group(0).strip().lower()

        if generic in seen:
            continue
        seen.add(generic)

        pre_start  = max(0, match.start() - 40)
        post_end   = min(len(plan_text), match.end() + 160)
        pre_text   = plan_text[pre_start:match.start()]
        post_text  = plan_text[match.end():post_end]
        full_ctx   = plan_text[pre_start:post_end]

        dose = route = frequency = duration = indication = ""

        detail = _DOSE_DETAIL_PATTERN.search(post_text)
        if detail:
            dose      = (detail.group("dose")      or "").strip()
            frequency = (detail.group("frequency") or "").strip()
            duration  = (detail.group("duration")  or "").strip()
            route_raw = (detail.group("route")     or "").strip()
            route     = _ROUTE_MAP.get(route_raw.lower(), route_raw.upper()) if route_raw else ""

        if not route:
            for search_zone in (post_text, pre_text):
                rm = _ROUTE_KEYWORDS_PATTERN.search(search_zone)
                if rm:
                    route = _ROUTE_MAP.get(rm.group(0).lower(), rm.group(0).upper())
                    break

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
    """Returns the Layer 2 system instructions with active Layer 3 fallback rules."""
    return SYSTEM_PROMPT


def generate_drug_interaction_prompt(
    soap_plan: str,
    rxnav_report: Optional[Dict[str, Any]] = None,
    active_medications: Optional[List[Any]] = None,
    allergies: Optional[List[Any]] = None,
    patient_context: Optional[Dict[str, Any]] = None
) -> str:
    """
    Builds the user-turn prompt for the Layer 2/3 drug interaction agent.
    """
    new_meds = extract_medications_from_plan(soap_plan)

    if not new_meds:
        return (
            "No Rx entries were identified in the SOAP plan. "
            "Return JSON with all interaction lists empty and "
            "overall_safety_level set to 'SAFE'. "
            "Set summary to: 'No pharmacologic therapy was prescribed in this encounter.'"
        )

    existing_meds  = _format_active_medications(active_medications or [])
    allergy_list   = _format_allergies(allergies or [])
    special_ctx    = _extract_special_population_context(patient_context or {})

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
            "Analyze the provided clinical medications data. If Layer 1 results are provided, "
            "rely on them as ground truth. IF THE RXNAV LAYER 1 RESULTS ARE MISSING OR EMPTY, "
            "activate Layer 3 Fallback Mode: Evaluate the medications (e.g., insulin, iv fluids) using "
            "your internal clinical knowledge. Identify standard clinical warnings, typical "
            "drug-drug interactions, and extract standard side effects, setting 'db_verified' to false."
        ),
        "newly_prescribed_medications":  new_meds,
        "existing_active_medications":   existing_meds,
        "documented_allergies":          allergy_list,
        "patient_special_population_context": special_ctx,
        "layer1_rxnav_results":          layer1_summary  
    }

    return (
        "Analyse the following clinical and drug data. "
        "If RxNav database results are empty or missing, rely safely on your Layer 3 AI fallback instructions.\n\n"
        f"{json.dumps(prompt_payload, indent=2)}"
    )