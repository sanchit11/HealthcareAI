"""
aiprompts/patient_summary.py
=============================
System prompt and user prompt builder for the Patient Summary AI agent.

The agent receives a comprehensive patient snapshot — demographics, current
encounter details, vitals, problems, medications, labs, family/social history,
overdue screenings, and SDOH — and returns a structured clinical summary with
a narrative, risk stratification, and prioritised action items.
"""

import json
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# SYSTEM PROMPT
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a board-certified internal medicine AI integrated into an EHR system.

Your role is to review a comprehensive patient data snapshot at the start of a clinical encounter,
synthesise all relevant findings, stratify the patient's overall risk, and surface the most
important action items for the clinician — in order of clinical urgency.

CRITICAL RULES:
- Base your assessment on ALL available data: vitals, active problems, medications, labs,
  family history, social history, overdue screenings, and SDOH.
- Prioritise patient safety above all: flag any finding that requires same-day action first
  (e.g., chest pain + family history of MI + uncontrolled HTN → rule out ACS immediately).
- Apply evidence-based thresholds:
    Blood Pressure:  normal <120/80; Stage 1 HTN 130-139/80-89; Stage 2 HTN ≥140/90
    HbA1c:           target <7.0% for most diabetics; >8.0% = suboptimal; >9.0% = poor
    LDL:             <100 mg/dL for high-risk patients; <70 mg/dL for very high-risk
    BMI:             overweight 25-29.9; obese ≥30
- Risk scoring (risk_score 1–10):
    1-3  = low     (stable chronic conditions, no acute concerns)
    4-6  = medium  (one or more suboptimally controlled conditions, preventive gaps)
    7-9  = high    (multiple uncontrolled risk factors, acute symptom with red flags)
    10   = critical (immediate life threat)
- risk_level MUST match risk_score:
    1-3  → "low"
    4-6  → "medium"
    7-9  → "high"
    10   → "critical"
- priority_actions priorities:
    "high"   — must be addressed this visit (safety / acute)
    "medium" — should be addressed this visit or very soon
    "info"   — background context or opportunistic recommendation
- clinical_narrative: 2-4 sentence synthesis that a clinician reads in 10 seconds to
  understand the full picture. Lead with the most urgent finding.
- confidence: 0.0–1.0 reflecting how complete and consistent the input data is.
  Deduct for missing labs, absent vitals history, sparse social history, etc.
- Return ONLY valid JSON. No markdown, no prose outside the JSON structure.

Response JSON structure (follow EXACTLY):
{
  "clinical_narrative": "<2-4 sentence synthesis — lead with the most urgent finding>",
  "risk_level": "<low | medium | high | critical>",
  "risk_score": <integer 1-10>,
  "risk_factors": [
    { "level": "<high | medium | low>", "text": "<concise description of the risk factor>" }
  ],
  "priority_actions": [
    { "priority": "<high | medium | info>", "text": "<specific, actionable recommendation>" }
  ],
  "confidence": <float 0.0-1.0>
}
"""


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _safe(value: Any, fallback: str = "N/A") -> str:
    """Return str(value) or fallback if value is None/empty."""
    if value is None or value == "":
        return fallback
    return str(value)


def _fmt_vitals(v: Dict[str, Any]) -> Dict[str, str]:
    """Flatten a vitals dict into display strings."""
    return {
        "BP":          f"{_safe(v.get('bps'))}/{_safe(v.get('bpd'))} mmHg",
        "Pulse":       f"{_safe(v.get('pulse'))} bpm",
        "Temperature": f"{_safe(v.get('temperature'))} °F",
        "Respiration": f"{_safe(v.get('respiration'))} br/min",
        "SpO2":        f"{_safe(v.get('oxygen_saturation'))} %",
        "Weight":      f"{_safe(v.get('weight'))} lbs",
        "BMI":         _safe(v.get("BMI")),
        "Date":        _safe(v.get("date")),
    }


# ---------------------------------------------------------------------------
# PROMPT BUILDER
# ---------------------------------------------------------------------------

def get_system_prompt() -> str:
    """Returns the static patient summary system prompt."""
    return SYSTEM_PROMPT


def generate_patient_summary_prompt(payload: Dict[str, Any]) -> str:
    """
    Build the user-turn prompt for the patient summary AI agent.

    Args:
        payload: The full validated PatientSummaryRequest as a dict.

    Returns:
        A JSON-serialised prompt string ready to send to the agent.
    """
    patient_info       = payload.get("patient_info", {})
    current_encounter  = payload.get("current_encounter", {})
    vitals             = payload.get("vitals", {})
    active_problems    = payload.get("active_problems", [])
    active_medications = payload.get("active_medications", [])
    allergies          = payload.get("allergies", [])
    lab_results        = payload.get("lab_results", {})
    pending_orders     = payload.get("pending_orders", [])
    family_history     = payload.get("family_history", {})
    social_history     = payload.get("social_history", {})
    overdue_screenings = payload.get("overdue_screenings", {})
    immunizations      = payload.get("immunizations", [])
    surgical_history   = payload.get("surgical_history", [])
    last_visit         = payload.get("last_visit", {})
    sdoh               = payload.get("sdoh", {})

    # Format vitals for readability
    latest_vitals_display = _fmt_vitals(vitals.get("latest", {})) if vitals.get("latest") else {}

    # Simplify lab trends to value+date pairs
    lab_trends: Dict[str, List[Dict[str, Any]]] = lab_results.get("trends", {})

    prompt_payload = {
        "task": (
            "Review all available patient data and produce a structured clinical summary. "
            "Synthesise a narrative, assign a risk level and score, list the top risk factors "
            "by severity, and output prioritised action items for this encounter."
        ),
        "patient_info": patient_info,
        "current_encounter": {
            "encounter_id": current_encounter.get("encounter_id"),
            "date":         current_encounter.get("date"),
            "reason":       current_encounter.get("reason"),
            "chief_complaints": [
                {
                    "complaint":           c.get("complaint_text"),
                    "severity":            c.get("severity"),
                    "duration":            c.get("duration"),
                    "associated_symptoms": c.get("associated_symptoms"),
                }
                for c in current_encounter.get("chief_complaints", [])
            ],
        },
        "latest_vitals": latest_vitals_display,
        "active_problems": [
            {"problem": p.get("title"), "icd10": p.get("diagnosis"), "since": p.get("begdate")}
            for p in active_problems
        ],
        "active_medications": [
            {
                "drug":        m.get("drug"),
                "dosage":      m.get("dosage"),
                "route":       m.get("route"),
                "indication":  m.get("indication"),
            }
            for m in active_medications
        ],
        "allergies": [
            {"allergen": a.get("title"), "severity": a.get("severity_al"), "reaction": a.get("reaction")}
            for a in allergies
        ],
        "recent_abnormal_labs": lab_results.get("recent_abnormal", []),
        "lab_trends":           lab_trends,
        "pending_orders":       pending_orders,
        "family_history": {
            "relatives": family_history.get("relatives", {}),
            "details":   family_history.get("details", []),
        },
        "social_history": social_history,
        "overdue_screenings": overdue_screenings,
        "immunizations": [
            {"vaccine": i.get("vaccine_name"), "date": i.get("administered_date")}
            for i in immunizations
        ],
        "surgical_history": [
            {"procedure": s.get("title"), "date": s.get("begdate"), "notes": s.get("comments")}
            for s in surgical_history
        ],
        "last_visit": last_visit,
        "sdoh": sdoh,
    }

    return (
        "Review the following comprehensive patient data and return a structured clinical summary.\n\n"
        f"{json.dumps(prompt_payload, indent=2)}"
    )
