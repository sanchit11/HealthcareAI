import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Hides info and warning logs

from fastapi import FastAPI, HTTPException, status, WebSocket, WebSocketDisconnect, UploadFile, File

from pydantic import ValidationError
from typing import Dict, Any
from services.soapAudio import transcribe_audio_stream
from dotenv import load_dotenv

# Import both AI handlers
from soapAI import generate_soap_note as generate_soap_openai
from soapOllama import generate_soap_note_ollama
from ioschemas.SoapSchemas import (
    SoapGenerationPayload,
    AmbientDetails,
    PatientContext,
    SoapNoteOutput,
)

# Drug interaction checker — runs only after SOAP generation is complete
from aiprompts.druginteraction import (
    get_system_prompt as get_drug_interaction_system_prompt,
    generate_drug_interaction_prompt,
    extract_medications_from_plan,
)
from agents import Agent, Runner
from pathlib import Path

# ── Presidio PHI masking ──────────────────────────────────────────────────────
# Masks patient PHI before any text is sent to OpenAI or Ollama.
# If presidio_service is not installed, masking is skipped with a warning.
try:
    from services.presidio_service import mask_phi, mask_phi_in_dict, mask_phi_in_text_fields
    PRESIDIO_AVAILABLE = True
except ImportError:
    import logging
    logging.warning(
        "⚠️  presidio_service not available — PHI masking is DISABLED.\n"
        "Run:  pip install presidio-analyzer presidio-anonymizer spacy\n"
        "      python -m spacy download en_core_web_lg"
    )
    PRESIDIO_AVAILABLE = False

    # No-op stubs so the rest of the code doesn't need if/else everywhere
    def mask_phi(text: str, **_) -> str:
        return text

    def mask_phi_in_dict(data: dict, fields: list, **_) -> dict:
        return data

    def mask_phi_in_text_fields(data: dict, **_) -> dict:
        return data
# ─────────────────────────────────────────────────────────────────────────────


load_dotenv(override=True)

# Detect router engine from environment
USE_OPENAI = os.getenv("USE_OPENAI", "true").lower() in ("true", "1", "yes")

app = FastAPI(title="AI Medical API", version="1.0.0")



@app.post("/transcribe_chunk")
async def transcribe_chunk(file: UploadFile = File(...)):
    # Read the incoming chunk bytes
    audio_bytes = await file.read()
    # Extract file extension from filename (e.g., 'wav')
    ext = file.filename.split(".")[-1] if "." in file.filename else "wav"

    # Route to your soapAudio processing engine
    transcript_text = await transcribe_audio_stream(audio_bytes, file_extension=ext)

    # ── Presidio: mask PHI in transcript before returning to PHP ─────────────
    # The raw transcript may contain patient names, DOB, phone numbers, etc.
    # We mask here so the PHP layer never receives unmasked PHI over the wire.
    if PRESIDIO_AVAILABLE:
        print("🔒 Masking PHI in transcript chunk...")
        transcript_text = mask_phi(transcript_text)

    return {"text": transcript_text}

@app.websocket("/stream_audio")
async def stream_audio_endpoint(websocket: WebSocket, format: str = "wav"):
    """
    WebSocket endpoint accepting a live sequential binary stream of audio chunks.
    When the client finishes sending data and gives a stop command, it returns
    the text transcription.

    Query Parameter:
        format: The incoming format codec (e.g. wav, mp3, m4a, webm)
    """
    await websocket.accept()
    print(f"📡 Audio connection opened. Streaming format target: {format}")

    # Store continuous binary chunks in an active memory buffer array
    audio_stream_buffer = bytearray()

    try:
        while True:
            # Constantly await payloads from the client connection
            message = await websocket.receive()

            # If data received is raw binary chunk
            if "bytes" in message:
                audio_stream_buffer.extend(message["bytes"])

            # If data received is text command structure
            elif "text" in message:
                if message["text"] == "STOP_STREAMING":
                    print("Received STOP instruction. Finalizing stream and transcribing...")
                    break

    except WebSocketDisconnect:
        print("🔌 WebSocket closed by client prematurely.")

    except Exception as e:
        print(f"❌ Error reading streaming frame: {str(e)}")
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        return

    # Once stream completes safely, execute processing block
    if len(audio_stream_buffer) > 0:
        try:
            await websocket.send_json({"status": "processing", "message": "Transcribing payload..."})

            # Fire transcription logic handler
            transcribed_text = await transcribe_audio_stream(
                bytes(audio_stream_buffer),
                file_extension=format
            )

            # ── Presidio: mask PHI before returning transcript over WebSocket ─
            # The full stream transcript may contain names, DOB, contact info, etc.
            if PRESIDIO_AVAILABLE:
                print("🔒 Masking PHI in full audio stream transcript...")
                transcribed_text = mask_phi(transcribed_text)

            # Return result back to user interface over the socket
            await websocket.send_json({
                "status": "success",
                "transcript": transcribed_text
            })

        except Exception as e:
            await websocket.send_json({
                "status": "error",
                "message": f"Transcription engine error: {str(e)}"
            })
    else:
        await websocket.send_json({
            "status": "empty",
            "message": "Stream finalized but zero binary data chunks were registered."
        })

    await websocket.close()
    print("🏁 Audio processing stream session ended.")



def validate_request_structure(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Explicitly validate request against SoapGenerationPayload schema.

    Args:
        request_data: Dictionary representation of the incoming request

    Returns:
        Validated SoapGenerationPayload object as dict

    Raises:
        ValueError: If request structure doesn't match schema
    """
    validation_errors = []

    # Check top-level fields
    required_fields = ["action", "ambient_details", "patient_context", "options"]
    for field in required_fields:
        if field not in request_data:
            validation_errors.append(f"Missing required field: '{field}'")

    if validation_errors:
        raise ValueError(f"Request validation failed: {'; '.join(validation_errors)}")

    # Validate action field
    if request_data.get("action") != "generate_soap":
        raise ValueError(f"Invalid action: '{request_data.get('action')}'. Must be 'generate_soap'")

    # Attempt full Pydantic validation
    try:
        validated_payload = SoapGenerationPayload(**request_data)
        return validated_payload
    except ValidationError as e:
        # Parse Pydantic errors for better readability
        error_details = []
        for error in e.errors():
            field_path = " -> ".join(str(x) for x in error['loc'])
            error_msg = error['msg']
            error_type = error['type']
            error_details.append(f"Field '{field_path}': {error_msg} (type: {error_type})")

        detailed_error = "\n".join(error_details)
        raise ValueError(f"Request structure does not match SoapSchemas:\n{detailed_error}")


def validate_ambient_details(ambient_data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate ambient_details sub-structure."""
    required_fields = [
        "session_id", "encounter_id", "encounter_date", "encounter_time",
        "encounter_type", "provider_credentials", "clinic_name", "location",
        "raw_transcript", "diarized_transcript"
    ]

    missing_fields = [f for f in required_fields if f not in ambient_data]
    if missing_fields:
        raise ValueError(f"ambient_details missing required fields: {missing_fields}")

    return ambient_data


def validate_patient_context(patient_data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate patient_context sub-structure."""
    required_sections = [
        "demographics", "problem_list", "active_medications", "allergies",
        "vitals_current", "labs_recent", "diagnostic_tests", "social_history",
        "family_history", "intake_form", "last_visit"
    ]

    missing_sections = [s for s in required_sections if s not in patient_data]
    if missing_sections:
        raise ValueError(f"patient_context missing required sections: {missing_sections}")

    return patient_data


def validate_generation_options(options_data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate options sub-structure."""
    required_fields = [
        "include_icd10_codes", "include_snomed_codes",
        "include_confidence_scores", "include_safety_checks"
    ]

    missing_fields = [f for f in required_fields if f not in options_data]
    if missing_fields:
        raise ValueError(f"options missing required fields: {missing_fields}")

    # Validate boolean fields
    for field in required_fields:
        if field in options_data and not isinstance(options_data[field], bool):
            raise ValueError(f"options.{field} must be boolean, got {type(options_data[field]).__name__}")

    return options_data

@app.get("/")
def read_root():
    engine = "Azure OpenAI Agents" if USE_OPENAI else f"Ollama ({os.getenv('OLLAMA_MODEL', 'unknown')})"
    return {
        "message": "AI Medical SOAP Generator API is running",
        "active_engine": engine,
        "phi_masking": "presidio" if PRESIDIO_AVAILABLE else "disabled"
    }

@app.get("/schema")
def get_schema():
    """
    Return the expected schema structure for debugging.
    """
    return {
        "message": "Expected SoapGenerationPayload structure",
        "required_fields": ["action", "ambient_details", "patient_context", "options"],
        "action": "Must be 'generate_soap'",
        "ambient_details": {
            "required": ["session_id", "encounter_id", "encounter_date", "encounter_time",
                        "encounter_type", "provider_credentials", "clinic_name", "location",
                        "raw_transcript", "diarized_transcript"],
            "optional": ["duration_seconds", "status", "provider_token"]
        },
        "patient_context": {
            "required": ["demographics", "problem_list", "active_medications", "allergies",
                        "vitals_current", "labs_recent", "diagnostic_tests", "social_history",
                        "family_history", "intake_form", "last_visit"]
        },
        "options": {
            "required": ["include_icd10_codes", "include_snomed_codes",
                        "include_confidence_scores", "include_safety_checks"],
            "optional": ["temperature", "max_tokens", "output_format"]
        }
    }


def _mask_soap_payload(request: SoapGenerationPayload) -> SoapGenerationPayload:
    """
    Deep-mask PHI across the full SOAP generation payload.

    Strategy:
      1. Mask raw_transcript and diarized_transcript in ambient_details —
         these are the highest-risk fields (direct audio-to-text, unfiltered).
      2. Mask patient_context entirely via recursive walk —
         demographics, social_history, intake_form, etc. can all contain PHI.

    The masked payload is reconstructed as a new SoapGenerationPayload so
    the rest of the pipeline (soapAI / soapOllama) receives a clean object.
    """
    payload_dict = request.model_dump()

    # Step 1: Mask high-risk transcript fields
    TRANSCRIPT_FIELDS = [
        "ambient_details.raw_transcript",
        "ambient_details.diarized_transcript",
    ]
    payload_dict = mask_phi_in_dict(payload_dict, TRANSCRIPT_FIELDS)

    # Step 2: Recursively mask all string values inside patient_context
    # This covers names, addresses, DOB, MRN, phone, email, SSN, etc.
    if "patient_context" in payload_dict:
        payload_dict["patient_context"] = mask_phi_in_text_fields(
            payload_dict["patient_context"]
        )

    # Reconstruct as a validated Pydantic model
    return SoapGenerationPayload(**payload_dict)


@app.post("/generate_soap")
async def generate_soap(request: SoapGenerationPayload):
    session_id = None

    try:
        validated_request = request
        session_id = request.ambient_details.session_id
        print(f"✓ Request validation passed for session: {session_id}")

        # ── Presidio: mask PHI before sending to LLM ─────────────────────────
        # This is the critical gate — PHI must never reach OpenAI or Ollama.
        # Even if the PHP layer (PhiDeidentifyService.php) already masked PHI,
        # we apply a second pass here as a safety net.
        if PRESIDIO_AVAILABLE:
            print(f"🔒 Masking PHI in payload for session {session_id}...")
            validated_request = _mask_soap_payload(validated_request)
            print(f"✅ PHI masking complete for session {session_id}.")
        # ─────────────────────────────────────────────────────────────────────

        # Dynamic routing based on .env toggle - pass the validated Pydantic object
        if USE_OPENAI:
            print(f"Routing session {session_id} to OpenAI Engine...")
            result = await generate_soap_openai(validated_request)
        else:
            print(f"Routing session {session_id} to Ollama Engine...")
            result = await generate_soap_note_ollama(validated_request)

        return {
            "status": "success",
            "session_id": session_id,
            "engine_used": "openai" if USE_OPENAI else "ollama",
            "phi_masking_applied": PRESIDIO_AVAILABLE,
            "data": result
        }

    except ValueError as ve:
        print(f"✗ Validation error for session {session_id}: {str(ve)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "status": "validation_error",
                "message": str(ve),
                "session_id": session_id
            }
        )
    except Exception as e:
        import traceback
        print(f"✗ Processing error for session {session_id}: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "error",
                "message": str(e),
                "session_id": session_id or "unknown"
            }
        )


@app.post("/validate_payload")
async def validate_payload(request: Dict[str, Any]):
    """
    Endpoint to validate a payload without generating SOAP note.
    Useful for debugging request structure.
    """
    try:
        validated = validate_request_structure(request)
        return {
            "status": "valid",
            "message": "Payload structure matches SoapSchemas",
            "session_id": validated.ambient_details.session_id
        }
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "status": "invalid",
                "message": str(ve)
            }
        )


@app.get("/phi_status")
def phi_status():
    """
    Health check endpoint to confirm whether Presidio PHI masking is active.
    """
    return {
        "presidio_available": PRESIDIO_AVAILABLE,
        "masking_active": PRESIDIO_AVAILABLE,
        "message": (
            "PHI masking is ACTIVE via Presidio."
            if PRESIDIO_AVAILABLE
            else "PHI masking is DISABLED. Install presidio-analyzer and restart."
        )
    }


# =============================================================================
# DRUG INTERACTION ENDPOINT
#
# Called AFTER /generate_soap returns successfully.
# Accepts the SOAP result plus the original patient_context so the checker
# can cross-reference against existing active medications and allergies.
#
# CDS Hooks (future path):
#   This endpoint is designed to be called as a CDS Hooks `medication-prescribe`
#   service. The incoming `context.draftOrders` FHIR bundle maps to `soap_plan`,
#   and `prefetch.patientMedications` / `prefetch.patientAllergies` map to
#   `active_medications` / `allergies` below.
# =============================================================================

from pydantic import BaseModel as _BaseModel

class DrugInteractionRequest(_BaseModel):
    """
    Payload for /check_drug_interactions.

    Fields:
        soap_plan         — The `plan` string from a completed SoapNoteOutput.
        active_medications — Patient's current med list (from patient_context).
        allergies          — Patient's documented allergy list.
        patient_context    — Optional full patient_context dict for special-
                             population checks (renal, hepatic, elderly, etc.).
        session_id         — Passed through for traceability.
    """
    soap_plan:          str
    active_medications: list = []
    allergies:          list = []
    patient_context:    Dict[str, Any] = {}
    session_id:         str = ""


@app.post("/check_drug_interactions")
async def check_drug_interactions(request: DrugInteractionRequest):
    """
    Perform drug interaction analysis on medications found in a completed SOAP plan.

    Workflow:
        1. Extract Rx entries from soap_plan.
        2. Merge with existing active_medications and allergies.
        3. Send to pharmacology AI agent.
        4. Return structured DrugInteractionReport JSON.

    This endpoint must be called only AFTER /generate_soap has returned
    a successful response.
    """
    session_id = request.session_id or "unknown"

    try:
        # Step 1 — Quick pre-check: are there any Rx lines at all?
        new_meds = extract_medications_from_plan(request.soap_plan)
        if not new_meds:
            return {
                "status": "skipped",
                "session_id": session_id,
                "message": "No medications found in SOAP plan — drug interaction check not required.",
                "data": {
                    "medications_analyzed": [],
                    "drug_drug_interactions": [],
                    "side_effects": [],
                    "dose_warnings": [],
                    "allergy_alerts": [],
                    "special_population_flags": [],
                    "overall_safety_level": "SAFE",
                    "summary": "No pharmacologic therapy was prescribed.",
                    "report_confidence": 1.0
                }
            }

        # Step 2 — Build the user prompt
        user_prompt = generate_drug_interaction_prompt(
            soap_plan=request.soap_plan,
            active_medications=request.active_medications,
            allergies=request.allergies,
            patient_context=request.patient_context or {}
        )

        # Step 3 — Instantiate the pharmacology agent
        drug_agent = Agent(
            name="Clinical Pharmacology Safety Agent",
            instructions=get_drug_interaction_system_prompt(),
            model="gpt-4o-mini",
        )

        # Step 4 — Run the agent
        run_result = await Runner.run(drug_agent, user_prompt)
        raw_output = run_result.final_output

        # Step 5 — Parse JSON output (agent returns raw JSON string)
        if isinstance(raw_output, str):
            # Strip accidental markdown fences if present
            cleaned = raw_output.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            interaction_report = json.loads(cleaned)
        else:
            # Already a dict (shouldn't happen without output_type, but guard anyway)
            interaction_report = raw_output

        return {
            "status":  "success",
            "session_id": session_id,
            "medications_found": len(new_meds),
            "data": interaction_report
        }

    except json.JSONDecodeError as je:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "parse_error",
                "message": f"Agent returned non-JSON output: {str(je)}",
                "session_id": session_id
            }
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "error",
                "message": str(e),
                "session_id": session_id
            }
        )