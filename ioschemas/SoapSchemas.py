# schemas.py
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

# =====================================================================
# INPUT SCHEMAS
#
# Every previously-required field now carries a default value so that
# a partially-populated PHP payload never triggers a 422.
# Fields that are genuinely numeric use Optional[type] = None rather
# than 0 so the LLM can distinguish "not recorded" from "zero".
# =====================================================================

class TranscriptTurn(BaseModel):
    speaker:   str = ""
    text:      str = ""
    start_ms:  int = 0
    end_ms:    int = 0

class AmbientDetails(BaseModel):
    session_id:           str                    = ""
    encounter_id:         int                    = 0
    encounter_date:       str                    = ""
    encounter_time:       str                    = ""
    encounter_type:       str                    = ""
    duration_seconds:     Optional[int]          = None
    status:               Optional[str]          = None
    provider_token:       Optional[str]          = None
    provider_credentials: str                    = ""
    clinic_name:          str                    = ""
    location:             str                    = ""
    raw_transcript:       str                    = ""
    diarized_transcript:  List[TranscriptTurn]   = Field(default_factory=list)

class Demographics(BaseModel):
    age:                Optional[int]  = None
    sex:                str            = ""
    preferred_language: Optional[str]  = None
    occupation:         Optional[str]  = None

class ProblemItem(BaseModel):
    icd10:        str           = ""
    display:      str           = ""
    onset_year:   Optional[int] = None
    severity:     str           = ""
    status:       str           = "active"
    _source_table: Optional[str] = None
    _source_type:  Optional[str] = None

class ActiveMedication(BaseModel):
    rxnorm:       Optional[str]  = None
    drug_name:    str            = ""
    dosage:       str            = ""
    frequency:    str            = ""
    route:        str            = ""
    indication:   str            = ""
    start_date:   Optional[str]  = None
    active:       Optional[bool] = None
    _source_table: Optional[str] = None

class Allergy(BaseModel):
    allergen:       str           = ""
    allergen_type:  str           = ""
    reaction:       str           = ""
    severity:       str           = ""
    onset_minutes:  int           = 0
    documented_date: str          = ""
    _source_table:  Optional[str] = None
    _source_type:   Optional[str] = None

class VitalsCurrent(BaseModel):
    bps:               str            = ""
    bpd:               str            = ""
    bp_display:        str            = ""
    pulse:             Optional[int]   = None
    respiration:       Optional[int]   = None
    temperature:       Optional[float] = None
    temp_method:       Optional[str]   = None
    oxygen_saturation: Optional[int]   = None
    oxygen_delivery:   str            = ""
    height:            Optional[int]   = None
    height_unit:       str            = ""
    weight:            Optional[int]   = None
    weight_unit:       str            = ""
    BMI:               Optional[float] = None
    BMI_status:        str            = ""
    captured:          str            = ""
    _source_table:     Optional[str]   = None

class LabPrior(BaseModel):
    value: float = 0.0
    date:  str   = ""

class LabResult(BaseModel):
    loinc:           str                    = ""
    test:            str                    = ""
    value:           float                  = 0.0
    unit:            str                    = ""
    reference_range: str                    = ""
    abnormal:        str                    = ""
    date:            str                    = ""
    prior:           Optional[List[LabPrior]] = None
    _source_table:   Optional[str]           = None

class DiagnosticTest(BaseModel):
    test_name:      str           = ""
    loinc:          Optional[str] = None
    test_date:      str           = ""
    result:         str           = ""
    interpretation: str           = ""
    _source_table:  Optional[str] = None

class SocialHistory(BaseModel):
    smoking_status:        str           = "unknown"
    alcohol_use:           str           = "unknown"
    recreational_drugs:    str           = "unknown"
    occupation:            Optional[str] = None
    stress_level:          str           = "unknown"
    sleep_quality:         str           = "unknown"
    sleep_hours_per_night: Optional[int] = None
    _source_table:         Optional[str] = None

class FamilyHistoryItem(BaseModel):
    relation:   str       = ""
    conditions: List[str] = Field(default_factory=list)

class IntakeForm(BaseModel):
    chief_complaint:           str            = ""
    patient_reported_symptoms: List[str]      = Field(default_factory=list)
    pain_scale:                Optional[int]  = None
    screening:                 Optional[Dict[str, Any]] = None

class LastVisit(BaseModel):
    date:     str           = ""
    reason:   str           = ""
    provider: Optional[str] = None

class PriorLikedNote(BaseModel):
    """
    A previously generated SOAP draft that the physician explicitly liked.
    Per-section ratings are 1–10; higher = stronger preference.
    weight  — normalised importance score derived from average section rating / 10,
              range 0.1–1.0.  The LLM should mirror style / structure of high-weight
              notes more strongly than low-weight ones.
    Disliked drafts are never included here (filtered out in PHP).
    """
    date:        str            = ""
    weight:      float          = Field(default=0.6, ge=0.0, le=1.0)
    subjective:  str            = ""
    objective:   str            = ""
    assessment:  str            = ""
    plan:        str            = ""

    # Per-section physician feedback and ratings (1–10)
    ai_draft_subjective_feedback:   Optional[str] = None
    ai_draft_subjective_rating:     Optional[int] = Field(default=None, ge=1, le=10)
    ai_draft_objective_feedback:    Optional[str] = None
    ai_draft_objective_rating:      Optional[int] = Field(default=None, ge=1, le=10)
    ai_draft_assessment_feedback:   Optional[str] = None
    ai_draft_assessment_rating:     Optional[int] = Field(default=None, ge=1, le=10)
    ai_draft_plan_feedback:         Optional[str] = None
    ai_draft_plan_rating:           Optional[int] = Field(default=None, ge=1, le=10)

class PatientContext(BaseModel):
    patient_token:    Optional[str]              = None
    _source_table:    Optional[str]              = None
    demographics:     Demographics               = Field(default_factory=Demographics)
    problem_list:     List[ProblemItem]          = Field(default_factory=list)
    active_medications: List[ActiveMedication]   = Field(default_factory=list)
    allergies:        List[Allergy]              = Field(default_factory=list)
    vitals_current:   VitalsCurrent              = Field(default_factory=VitalsCurrent)
    labs_recent:      List[LabResult]            = Field(default_factory=list)
    diagnostic_tests: List[DiagnosticTest]       = Field(default_factory=list)
    social_history:   SocialHistory              = Field(default_factory=SocialHistory)
    family_history:   List[FamilyHistoryItem]    = Field(default_factory=list)
    intake_form:      IntakeForm                 = Field(default_factory=IntakeForm)
    cds_alerts:       Optional[List[Dict[str, Any]]] = None
    last_visit:       LastVisit                  = Field(default_factory=LastVisit)
    prior_liked_notes: List[PriorLikedNote]      = Field(
        default_factory=list,
        description=(
            "Physician-approved SOAP drafts from prior visits for this patient. "
            "Each entry carries per-section ratings (1–10) and a normalised weight (0.1–1.0). "
            "Use these as style/structure examples; give more weight to higher-rated notes."
        )
    )

class GenerationOptions(BaseModel):
    include_icd10_codes:       bool           = Field(default=True)
    include_snomed_codes:      bool           = Field(default=False)
    include_confidence_scores: bool           = Field(default=True)
    include_safety_checks:     bool           = Field(default=True)
    temperature:               Optional[float] = None
    max_tokens:                Optional[int]   = None
    output_format:             Optional[str]   = None

class SoapGenerationPayload(BaseModel):
    """The complete wrapper matching the actual incoming JSON layout."""
    action:          str               = Field(description="Must be 'generate_soap'")
    ambient_details: AmbientDetails    = Field(default_factory=AmbientDetails)
    patient_context: PatientContext    = Field(default_factory=PatientContext)
    options:         GenerationOptions = Field(default_factory=GenerationOptions)


# =====================================================================
# OUTPUT SCHEMAS
# =====================================================================

class SoapNoteOutput(BaseModel):
    """The strict structured format for SOAP note generation."""
    subjective:       str           = Field(description="History of present illness, symptoms, and patient narrative.")
    objective:        str           = Field(description="Physical exam, vital signs, and diagnostic/lab data.")
    assessment:       str           = Field(description="Differential diagnoses and clinical impressions.")
    plan:             str           = Field(description="Therapeutic interventions, medications, referrals, and patient education.")
    confidence_score: Optional[float] = Field(None, description="Model's internal score if requested in options.")