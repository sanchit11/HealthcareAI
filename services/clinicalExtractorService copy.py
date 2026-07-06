"""
clinicalExtractorService.py
============================
Clinical Data Extractor — mirrors the pattern of presidio_service.py / rxnav_service.py.

Extracts structured clinical summaries from the SoapGenerationPayload and
the completed SoapNoteOutput, providing four dedicated panels for the frontend:

  1. current_vitals        — Formatted vitals from patient_context.vitals_current
                             with clinical flags (hypertension, tachycardia, fever, etc.)

  2. past_problem_list     — Active / resolved problems from patient_context.problem_list
                             with ICD-10 codes and status labels

  3. negative_lab_results  — Labs from patient_context.labs_recent that are within
                             normal range, PLUS any negatives mentioned in the SOAP
                             objective section (e.g. "CXR: no infiltrate").

  4. complaints            — One-line chief complaint derived from intake_form,
                             with a fallback to the first sentence of the raw_transcript.

Called from main.py inside generate_soap AFTER the SOAP result is ready.
Never raises — always returns a safe structured dict.
"""

import re
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. CURRENT VITALS
# ─────────────────────────────────────────────────────────────────────────────

# Clinical thresholds for auto-flagging abnormal vitals
_VITAL_THRESHOLDS = {
    "pulse":             {"low": 60,   "high": 100,  "unit": "bpm"},
    "respiration":       {"low": 12,   "high": 20,   "unit": "breaths/min"},
    "temperature":       {"low": 97.0, "high": 99.5, "unit": "°F"},
    "oxygen_saturation": {"low": 95,   "high": 100,  "unit": "%"},
    "bmi":               {"low": 18.5, "high": 24.9, "unit": "kg/m²"},
}

_BP_SYSTOLIC_HIGH  = 140
_BP_DIASTOLIC_HIGH = 90
_BP_SYSTOLIC_LOW   = 90
_BP_DIASTOLIC_LOW  = 60


def _flag_vital(key: str, value: float) -> Optional[str]:
    """Return a short clinical flag string if the value is outside normal range."""
    thresholds = _VITAL_THRESHOLDS.get(key)
    if thresholds is None:
        return None
    if value < thresholds["low"]:
        return "LOW"
    if value > thresholds["high"]:
        return "HIGH"
    return "NORMAL"


def _flag_bp(systolic: Optional[int], diastolic: Optional[int]) -> str:
    """Classify blood pressure."""
    if systolic is None or diastolic is None:
        return "UNKNOWN"
    if systolic >= 180 or diastolic >= 120:
        return "HYPERTENSIVE CRISIS"
    if systolic >= 140 or diastolic >= 90:
        return "HYPERTENSION"
    if systolic >= 130 or diastolic >= 80:
        return "ELEVATED"
    if systolic < _BP_SYSTOLIC_LOW or diastolic < _BP_DIASTOLIC_LOW:
        return "HYPOTENSION"
    return "NORMAL"


def extract_current_vitals(vitals_raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Parses patient_context.vitals_current into a structured, flagged vitals object.

    Args:
        vitals_raw: The vitals_current dict from the incoming payload.

    Returns:
        Dict with:
            blood_pressure   — { display, systolic, diastolic, flag }
            pulse            — { value, unit, flag }
            respiration      — { value, unit, flag }
            temperature      — { value, unit, flag }
            oxygen_saturation— { value, unit, delivery_method, flag }
            height           — { value, unit }
            weight           — { value, unit }
            bmi              — { value, status, flag }
            captured_at      — timestamp string
            has_abnormals    — True if any vital is flagged non-NORMAL
    """
    empty = {
        "blood_pressure":    None,
        "pulse":             None,
        "respiration":       None,
        "temperature":       None,
        "oxygen_saturation": None,
        "height":            None,
        "weight":            None,
        "bmi":               None,
        "captured_at":       None,
        "has_abnormals":     False,
    }

    if not vitals_raw:
        return empty

    try:
        # ── Blood Pressure ────────────────────────────────────────────────────
        # Note: in the example payload bps=systolic, bpd=diastolic
        # but the display string "100/130" suggests bps may be systolic here.
        # We follow the numeric values provided.
        systolic  = _safe_int(vitals_raw.get("bps"))
        diastolic = _safe_int(vitals_raw.get("bpd"))
        bp_display = vitals_raw.get("bp_display") or _build_bp_display(systolic, diastolic)
        bp_flag = _flag_bp(systolic, diastolic)

        bp = {
            "display":   bp_display,
            "systolic":  systolic,
            "diastolic": diastolic,
            "unit":      "mmHg",
            "flag":      bp_flag,
        }

        # ── Pulse ─────────────────────────────────────────────────────────────
        pulse_val = _safe_float(vitals_raw.get("pulse"))
        pulse = {
            "value": pulse_val,
            "unit":  "bpm",
            "flag":  _flag_vital("pulse", pulse_val) if pulse_val is not None else "UNKNOWN",
        }

        # ── Respiration ───────────────────────────────────────────────────────
        resp_val = _safe_float(vitals_raw.get("respiration"))
        respiration = {
            "value": resp_val,
            "unit":  "breaths/min",
            "flag":  _flag_vital("respiration", resp_val) if resp_val is not None else "UNKNOWN",
        }

        # ── Temperature ───────────────────────────────────────────────────────
        temp_val = _safe_float(vitals_raw.get("temperature"))
        temp_method = vitals_raw.get("temp_method") or "unspecified"
        temperature = {
            "value":  temp_val,
            "unit":   "°F",
            "method": temp_method,
            "flag":   _flag_vital("temperature", temp_val) if temp_val is not None else "UNKNOWN",
        }

        # ── Oxygen Saturation ─────────────────────────────────────────────────
        spo2_val = _safe_float(vitals_raw.get("oxygen_saturation"))
        o2_delivery = vitals_raw.get("oxygen_delivery") or "room air"
        # Zero typically means "not recorded" in many EMRs
        spo2_flag = (
            "NOT RECORDED" if (spo2_val is None or spo2_val == 0)
            else _flag_vital("oxygen_saturation", spo2_val)
        )
        oxygen_saturation = {
            "value":           spo2_val if spo2_val != 0 else None,
            "unit":            "%",
            "delivery_method": o2_delivery,
            "flag":            spo2_flag,
        }

        # ── Height & Weight ───────────────────────────────────────────────────
        height = {
            "value": _safe_float(vitals_raw.get("height")),
            "unit":  vitals_raw.get("height_unit") or "in",
        }
        weight = {
            "value": _safe_float(vitals_raw.get("weight")),
            "unit":  vitals_raw.get("weight_unit") or "lbs",
        }

        # ── BMI ───────────────────────────────────────────────────────────────
        bmi_val    = _safe_float(vitals_raw.get("BMI"))
        bmi_status = vitals_raw.get("BMI_status") or _classify_bmi(bmi_val)
        bmi = {
            "value":  round(bmi_val, 2) if bmi_val is not None else None,
            "status": bmi_status,
            "flag":   _flag_vital("bmi", bmi_val) if bmi_val is not None else "UNKNOWN",
        }

        # ── Abnormal check ────────────────────────────────────────────────────
        all_flags = [
            bp_flag,
            pulse.get("flag"),
            respiration.get("flag"),
            temperature.get("flag"),
            bmi.get("flag"),
        ]
        # SpO2 zero = not recorded, not abnormal
        if spo2_flag not in ("NOT RECORDED", "NORMAL", "UNKNOWN"):
            all_flags.append(spo2_flag)

        has_abnormals = any(
            f not in ("NORMAL", "UNKNOWN", "NOT RECORDED", None)
            for f in all_flags
        )

        return {
            "blood_pressure":    bp,
            "pulse":             pulse,
            "respiration":       respiration,
            "temperature":       temperature,
            "oxygen_saturation": oxygen_saturation,
            "height":            height,
            "weight":            weight,
            "bmi":               bmi,
            "captured_at":       vitals_raw.get("captured"),
            "has_abnormals":     has_abnormals,
        }

    except Exception as e:
        logger.error(f"extract_current_vitals error: {e}")
        return empty


# ─────────────────────────────────────────────────────────────────────────────
# 2. PAST PROBLEM LIST
# ─────────────────────────────────────────────────────────────────────────────

def extract_past_problem_list(
    problem_list: Optional[List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """
    Parses patient_context.problem_list into active and historical problem buckets.

    Args:
        problem_list: Raw problem_list array from the payload.

    Returns:
        Dict with:
            active   — list of active problems with icd10 + display + severity
            resolved — list of resolved/inactive problems
            total    — total count
    """
    empty = {"active": [], "resolved": [], "total": 0}

    if not problem_list:
        return empty

    try:
        active   = []
        resolved = []

        for problem in problem_list:
            if not isinstance(problem, dict):
                continue

            status   = (problem.get("status") or "unknown").lower().strip()
            icd10    = (problem.get("icd10") or "").strip()
            display  = (problem.get("display") or "").strip()
            severity = (problem.get("severity") or "").strip()
            onset    = problem.get("onset_year")

            # Normalise ICD-10: strip "ICD10:" prefix if present
            if icd10.upper().startswith("ICD10:"):
                icd10 = icd10[6:].strip()

            entry = {
                "display":    display or "Unspecified",
                "icd10":      icd10 or None,
                "severity":   severity or None,
                "onset_year": onset,
                "status":     status,
            }

            if status in ("active", "current", "ongoing"):
                active.append(entry)
            else:
                resolved.append(entry)

        return {
            "active":   active,
            "resolved": resolved,
            "total":    len(active) + len(resolved),
        }

    except Exception as e:
        logger.error(f"extract_past_problem_list error: {e}")
        return empty


# ─────────────────────────────────────────────────────────────────────────────
# 3. NEGATIVE LAB RESULTS
# ─────────────────────────────────────────────────────────────────────────────

# Patterns that indicate a result is negative / normal / within range
_NEGATIVE_PATTERNS = re.compile(
    r"\b(negative|normal|wnl|within normal limits?|not detected|undetected|"
    r"no growth|no organisms|unremarkable|absent|clear|clean)\b",
    re.IGNORECASE,
)

# Patterns in SOAP objective text that mention negative / clear findings
_SOAP_NEGATIVE_PATTERNS = re.compile(
    r"(?:"
    r"(?:no\s+(?:infiltrate|effusion|consolidation|mass|opacity|fracture|acute|lesion|"
    r"lymphadenopathy|edema|cardiomegaly))"
    r"|(?:(?:chest\s+x-?ray|cxr|ct|mri|echo|ekg|ecg)[\s\w,]+(?:normal|unremarkable|clear|negative))"
    r"|(?:(?:urine|blood|culture|swab|pcr|rapid\s+test|antigen)[\s\w,]+(?:negative|no\s+growth|not\s+detected))"
    r"|(?:ketones?[\s:]+(?:negative|none|absent))"
    r"|(?:(?:wbc|rbc|platelets?|hemoglobin|hematocrit)[\s:]+(?:normal|wnl|within\s+normal\s+limits?))"
    r")",
    re.IGNORECASE,
)


def extract_negative_lab_results(
    labs_recent:     Optional[List[Dict[str, Any]]],
    soap_objective:  Optional[str] = None,
) -> Dict[str, Any]:
    """
    Identifies negative / normal lab and diagnostic results from two sources:

    Source A — patient_context.labs_recent
        Any lab entry whose result text contains "negative", "normal", "WNL", etc.

    Source B — SoapNoteOutput.objective (AI-generated)
        Scans the SOAP objective section for phrases like
        "CXR: no infiltrate", "urine culture: negative", "WBC: WNL", etc.

    Args:
        labs_recent:    Raw labs_recent array from patient_context.
        soap_objective: The `objective` string from the generated SoapNoteOutput.

    Returns:
        Dict with:
            from_records   — negative labs found in labs_recent records
            from_soap_text — negative findings extracted from the SOAP objective
            total          — combined count
    """
    empty = {"from_records": [], "from_soap_text": [], "total": 0}

    try:
        from_records   = []
        from_soap_text = []

        # ── Source A: labs_recent ─────────────────────────────────────────────
        if labs_recent:
            for lab in labs_recent:
                if not isinstance(lab, dict):
                    continue

                result_text = str(lab.get("result") or lab.get("value") or "")
                lab_name    = str(lab.get("name") or lab.get("test") or "Unknown lab")
                lab_date    = lab.get("date") or lab.get("collected_date") or lab.get("reported_date")

                if _NEGATIVE_PATTERNS.search(result_text):
                    from_records.append({
                        "test":   lab_name,
                        "result": result_text.strip(),
                        "date":   lab_date,
                        "source": "labs_recent",
                    })

        # ── Source B: SOAP objective text ─────────────────────────────────────
        if soap_objective and isinstance(soap_objective, str):
            # Split into sentences / clauses for granular extraction
            sentences = re.split(r"[.\n;]", soap_objective)
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and _SOAP_NEGATIVE_PATTERNS.search(sentence):
                    # Clean and de-duplicate
                    cleaned = re.sub(r"\s{2,}", " ", sentence).strip(" -•:")
                    if cleaned and not any(
                        r["finding"] == cleaned for r in from_soap_text
                    ):
                        from_soap_text.append({
                            "finding": cleaned,
                            "source":  "soap_objective",
                        })

        return {
            "from_records":   from_records,
            "from_soap_text": from_soap_text,
            "total":          len(from_records) + len(from_soap_text),
        }

    except Exception as e:
        logger.error(f"extract_negative_lab_results error: {e}")
        return empty


# ─────────────────────────────────────────────────────────────────────────────
# 4. CHIEF COMPLAINT / INTAKE COMPLAINTS
# ─────────────────────────────────────────────────────────────────────────────

def extract_complaints(
    intake_form:    Optional[Dict[str, Any]],
    raw_transcript: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Extracts the patient's chief complaint and reported symptoms.

    Priority order:
        1. intake_form.chief_complaint  (explicit structured field)
        2. First meaningful sentence of raw_transcript  (fallback)

    Args:
        intake_form:    intake_form dict from patient_context.
        raw_transcript: raw_transcript string from ambient_details.

    Returns:
        Dict with:
            chief_complaint        — Single concise string (≤ ~150 chars)
            patient_symptoms       — List of self-reported symptom strings
            pain_scale             — Numeric pain score if provided
            source                 — "intake_form" | "transcript_fallback" | "none"
    """
    empty = {
        "chief_complaint":  "Not documented",
        "patient_symptoms": [],
        "pain_scale":       None,
        "source":           "none",
    }

    try:
        chief_complaint  = None
        patient_symptoms = []
        pain_scale       = None
        source           = "none"

        # ── Source 1: intake_form ─────────────────────────────────────────────
        if intake_form and isinstance(intake_form, dict):
            cc = (intake_form.get("chief_complaint") or "").strip()
            if cc:
                chief_complaint = _truncate(cc, 200)
                source = "intake_form"

            # Patient-reported symptoms
            raw_symptoms = intake_form.get("patient_reported_symptoms") or []
            if isinstance(raw_symptoms, list):
                patient_symptoms = [
                    str(s).strip() for s in raw_symptoms if s and str(s).strip()
                ]
            elif isinstance(raw_symptoms, str) and raw_symptoms.strip():
                patient_symptoms = [raw_symptoms.strip()]

            # Pain scale
            ps = intake_form.get("pain_scale")
            if ps is not None:
                pain_scale = _safe_int(ps)

        # ── Source 2: transcript fallback ─────────────────────────────────────
        if not chief_complaint and raw_transcript and isinstance(raw_transcript, str):
            first_sentence = _extract_first_sentence(raw_transcript)
            if first_sentence:
                chief_complaint = first_sentence
                source = "transcript_fallback"

        return {
            "chief_complaint":  chief_complaint or "Not documented",
            "patient_symptoms": patient_symptoms,
            "pain_scale":       pain_scale,
            "source":           source,
        }

    except Exception as e:
        logger.error(f"extract_complaints error: {e}")
        return empty


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT — Full extraction pass
# ─────────────────────────────────────────────────────────────────────────────

def extract_clinical_summary(
    validated_request,   # SoapGenerationPayload
    soap_result,         # SoapNoteOutput  (may be None on error paths)
) -> Dict[str, Any]:
    """
    Run all four extractors in one call and return a combined clinical summary dict.

    This is the function called from main.py generate_soap:

        clinical_summary = extract_clinical_summary(validated_request, result)

    Args:
        validated_request: The SoapGenerationPayload Pydantic object.
        soap_result:       The completed SoapNoteOutput (or None).

    Returns:
        {
            "current_vitals":       { ... },
            "new_vitals":           { ... },  # If we implement vitals extraction from SOAP (objective) text in the future
            "past_problem_list":    { ... },
            "negative_lab_results": { ... },
            "complaints":           { ... },
            "medicine":              { ... },  # If we implement medication extraction from SOAP (Plan) text in the future
        }

    This function NEVER raises — all errors are caught per-extractor.
    """
    patient_ctx  = validated_request.patient_context
    ambient      = validated_request.ambient_details
    soap_obj_text = soap_result.objective if soap_result else None

    current_vitals = extract_current_vitals(
        _model_to_dict(patient_ctx.vitals_current) if patient_ctx.vitals_current else None
    )

    past_problem_list = extract_past_problem_list(
        [_model_to_dict(p) for p in patient_ctx.problem_list]
        if patient_ctx.problem_list else []
    )

    negative_lab_results = extract_negative_lab_results(
        labs_recent    = [_model_to_dict(l) for l in patient_ctx.labs_recent]
                         if patient_ctx.labs_recent else [],
        soap_objective = soap_obj_text,
    )

    complaints = extract_complaints(
        intake_form    = _model_to_dict(patient_ctx.intake_form)
                         if patient_ctx.intake_form else None,
        raw_transcript = ambient.raw_transcript,
    )

    return {
        "current_vitals":       current_vitals,
        "past_problem_list":    past_problem_list,
        "negative_lab_results": negative_lab_results,
        "complaints":           complaints,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(val: Any) -> Optional[float]:
    """Convert a value to float safely, returning None on failure."""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _safe_int(val: Any) -> Optional[int]:
    """Convert a value to int safely, returning None on failure."""
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def _build_bp_display(systolic: Optional[int], diastolic: Optional[int]) -> str:
    """Build a BP display string from components."""
    if systolic is not None and diastolic is not None:
        return f"{systolic}/{diastolic}"
    return "N/A"


def _classify_bmi(bmi: Optional[float]) -> str:
    """Return a standard BMI classification string."""
    if bmi is None:
        return "Unknown"
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25.0:
        return "Normal"
    if bmi < 30.0:
        return "Overweight"
    return "Obese"


def _extract_first_sentence(text: str, max_chars: int = 200) -> str:
    """
    Returns the first meaningful sentence from a transcript or free-text field.
    Strips disfluencies and filler words common in dictated/ambient recordings.
    """
    # Remove common disfluencies
    cleaned = re.sub(r"\b(um|uh|er|ah|like|you know|okay|ok)\b", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()

    # Split at sentence boundaries
    match = re.split(r"(?<=[.!?])\s+", cleaned)
    if match:
        first = match[0].strip()
        return _truncate(first, max_chars)
    return _truncate(cleaned, max_chars)


def _truncate(text: str, max_chars: int) -> str:
    """Truncate text to max_chars, appending '…' if needed."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars - 1].rstrip() + "…"


def _model_to_dict(obj: Any) -> Any:
    """Convert a Pydantic model to dict, or return as-is if already a dict."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return obj