import os
import json
import httpx
from dotenv import load_dotenv
import aiprompts.soapprompts as soapprompts
from ioschemas.SoapSchemas import SoapGenerationPayload, SoapNoteOutput

load_dotenv(override=True)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

async def generate_soap_note_ollama(payload: SoapGenerationPayload) -> SoapNoteOutput:
    """
    Generate SOAP note using Ollama with structured JSON outputs.
    """
    if hasattr(payload, "model_dump"):
        input_dict = payload.model_dump()
    else:
        input_dict = payload.dict()
    
    # 1. Prepare system and user prompts
    system_prompt = soapprompts.get_system_prompt()
    user_prompt = soapprompts.generate_user_prompt(input_dict)
    
    # FIX: Replace the complex model_json_schema dump with explicit, clear structure instructions.
    # This prevents the LLM from echoing the schema description strings.
    full_user_prompt = (
        f"{user_prompt}\n\n"
        f"CRITICAL: You must return ONLY a flat valid JSON object containing exactly these four keys:\n"
        f"-\"subjective\": (string text)\n"
        f"-\"objective\": (string text)\n"
        f"-\"assessment\": (string text)\n"
        f"-\"plan\": (string text)\n\n"
        f"-\"current_vitals\": (string text)\n\n"
        f"-\"vitals\": (string text)\n\n"
        f"-\"confidence_score\": (decimal number between 0.0 and 1.0)\n\n"
        f"Do not write markdown backticks like ```json. Fill out the values with real information generated from the user context."
    )

    # 2. Prepare payload for Ollama API
    ollama_payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_user_prompt}
        ],
        "format": "json",
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_ctx": 8192,      # extend context window for large clinical prompts
            "num_predict": 2048,  # ensure enough output tokens for a full SOAP note
        }
    }

    # 3. Call Ollama via HTTP request
    try:
        timeout_config = httpx.Timeout(120.0, read=None)
        
        async with httpx.AsyncClient(timeout=timeout_config) as client:
            print(f"🔄 Sending request to Ollama at {OLLAMA_BASE_URL} with model '{OLLAMA_MODEL}'…")
            print("⏳ Waiting for Ollama generation (this may take a few minutes on CPU)...")
            
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/chat", 
                json=ollama_payload
            )
            
            if response.status_code != 200:
                print(f"❌ Ollama returned bad status code: {response.status_code}, Text: {response.text}")
                raise Exception(f"Ollama API error ({response.status_code}): {response.text}")
            
            result_json = response.json()
            content_string = result_json["message"]["content"]
            print(f"📝 Raw Ollama content ({len(content_string)} chars): {content_string[:200]}…")

            # 4. Clean up common model quirks before parsing
            cleaned = content_string.strip()
            # Strip markdown fences if present
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
                cleaned = cleaned.strip()

            # Attempt to repair truncated JSON by closing open braces/strings
            try:
                parsed_json = json.loads(cleaned)
            except json.JSONDecodeError as je:
                print(f"⚠️  JSON parse failed ({je}), attempting truncation repair…")
                # Close any open string then close the object
                cleaned = cleaned.rstrip()
                if not cleaned.endswith("}"):
                    if not cleaned.endswith('"'):
                        cleaned += '"'
                    cleaned += "}"
                parsed_json = json.loads(cleaned)

            # Clean up the object if the model still accidentally wraps keys in an object layer
            if "SoapNoteOutput" in parsed_json:
                parsed_json = parsed_json["SoapNoteOutput"]

            return SoapNoteOutput(**parsed_json)
            
    except httpx.ConnectError as ce:
        print(f"❌ Could not connect to Ollama at {OLLAMA_BASE_URL}. Is Ollama running?")
        raise Exception(f"Failed to connect to local Ollama instance: {str(ce)}")
    except Exception as e:
        import traceback
        print(f"❌ Error encountered in soapOllama processing: {str(e)}")
        traceback.print_exc()
        raise e