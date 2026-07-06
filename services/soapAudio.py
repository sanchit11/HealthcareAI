"""
soapAudio.py  —  Audio transcription engine for the Ambient Docs FastAPI backend

USE_OPENAI=true   → Cloud OpenAI Whisper API  (requires OPENAI_API_KEY)
USE_OPENAI=false  → faster-whisper running 100% locally  (no API key needed)
                    Handles webm/opus, wav, mp3, m4a, etc. via bundled FFmpeg.
                    Model is loaded once and cached for the lifetime of the server.
"""

import io
import os
import tempfile
from dotenv import load_dotenv
from pathlib import Path
# 1. Get the directory of the current file (synapta_AI/services)
current_file_dir = Path(__file__).resolve().parent

# 2. Go one level up to get to the root directory (synapta_AI) and point to .env
root_env_path = current_file_dir.parent / '.env'

# 3. Load the specific path explicitly
load_dotenv(dotenv_path=root_env_path, override=True)

# ── Local model cache ────────────────────────────────────────────────────────
# The WhisperModel is expensive to load (~1-3 s + model download on first run).
# We cache it at module level so every subsequent request is fast.
_whisper_model = None


def _get_local_model():
    """Load faster-whisper model once and return it on every subsequent call."""
    global _whisper_model
    if _whisper_model is None:
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise RuntimeError(
                "faster-whisper is not installed.\n"
                "Run:  pip install faster-whisper\n"
                "Then restart the FastAPI server."
            )
        model_size = os.getenv("LOCAL_WHISPER_MODEL", "base")
        print(f"🔄 Loading faster-whisper model '{model_size}' (first request only)…")
        # device="cpu"  works on any machine; compute_type="int8" keeps RAM usage low.
        # Change to device="cuda" + compute_type="float16" if you have a GPU.
        _whisper_model = WhisperModel(model_size, device="cpu", compute_type="int8")
        print(f"✅ faster-whisper model '{model_size}' ready.")
    return _whisper_model


# ── Public API ───────────────────────────────────────────────────────────────

async def transcribe_audio_stream(audio_bytes: bytes, file_extension: str = "wav") -> str:
    """
    Transcribe raw audio bytes.

    USE_OPENAI=true   → OpenAI cloud Whisper (whisper-1)
    USE_OPENAI=false  → faster-whisper locally (model set by LOCAL_WHISPER_MODEL)

    Supported formats: webm, opus, wav, mp3, m4a, ogg, flac, mp4, …
    (faster-whisper handles all formats that FFmpeg understands, and PyAV
    bundles its own FFmpeg binaries so no system-level FFmpeg is required.)
    """
    if not audio_bytes:
        return ""

    USE_OPENAI = os.getenv("USE_OPENAI", "true").lower() in ("true", "1", "yes")

    try:
        if USE_OPENAI:
            return await _transcribe_openai(audio_bytes, file_extension)
        else:
            return await _transcribe_local(audio_bytes, file_extension)

    except Exception as e:
        print(f"❌ Audio transcription error: {e}")
        raise


# ── Private helpers ──────────────────────────────────────────────────────────

async def _transcribe_openai(audio_bytes: bytes, file_extension: str) -> str:
    """Send audio to OpenAI Cloud Whisper API."""
    # Import lazily so the server starts even if openai isn't installed
    # (useful when USE_OPENAI=false).
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("openai package is not installed. Run: pip install openai")

    print("🎙️  Routing audio to Cloud OpenAI Whisper (whisper-1)…")
    client = OpenAI()  # reads OPENAI_API_KEY from environment

    audio_buffer = io.BytesIO(audio_bytes)
    audio_buffer.name = f"audio.{file_extension}"

    response = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_buffer,
        response_format="text",
    )
    text = response.strip() if isinstance(response, str) else str(response).strip()
    print(f"✅ OpenAI Whisper returned {len(text)} chars.")
    return text


def _convert_to_wav(audio_bytes: bytes, file_extension: str) -> bytes:
    """
    Convert any audio format (including streaming WebM/Opus chunks) to
    16 kHz mono WAV using PyAV + soundfile.  This step is what makes
    partial WebM fragments decodable — we re-mux the raw bytes rather
    than relying on the container header being present.
    """
    import av
    import numpy as np

    in_buf = io.BytesIO(audio_bytes)
    out_buf = io.BytesIO()

    # Open input from the in-memory buffer; ignore broken container metadata
    pcm_frames = []
    try:
        with av.open(in_buf, mode="r", format=file_extension if file_extension != "webm" else None) as container:
            for frame in container.decode(audio=0):
                arr = frame.to_ndarray()   # shape: (channels, samples) or (samples,)
                if arr.ndim > 1:
                    arr = arr.mean(axis=0)  # mix down to mono
                pcm_frames.append(arr.astype(np.float32))
    except Exception:
        # If container-level decode fails (e.g. headerless chunk), PyAV can't
        # help — re-raise so the caller can decide what to do.
        raise

    if not pcm_frames:
        return audio_bytes  # nothing decoded, return as-is and let Whisper try

    import soundfile as sf
    pcm = np.concatenate(pcm_frames)
    sf.write(out_buf, pcm, samplerate=16000, format="WAV", subtype="PCM_16")
    out_buf.seek(0)
    return out_buf.read()


async def _transcribe_local(audio_bytes: bytes, file_extension: str) -> str:
    """Transcribe using faster-whisper running locally on CPU."""
    print(f"🦙 Routing audio to local faster-whisper (format: {file_extension})…")

    # WebM chunks from MediaRecorder are streaming fragments — each chunk after
    # the first lacks a container header and PyAV will raise InvalidDataError.
    # Convert to WAV first so Whisper always gets a complete, valid file.
    if file_extension in ("webm", "ogg", "opus"):
        try:
            audio_bytes = _convert_to_wav(audio_bytes, file_extension)
            file_extension = "wav"
            print("🔄 Converted streaming chunk to WAV for Whisper.")
        except Exception as conv_err:
            print(f"⚠️  WAV conversion failed ({conv_err}); passing raw bytes to Whisper.")

    # faster-whisper needs a file path, not raw bytes — write to a temp file.
    suffix = f".{file_extension}" if not file_extension.startswith(".") else file_extension
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        model = _get_local_model()
        segments, info = model.transcribe(tmp_path, beam_size=5)

        # Consume the generator and join all segment texts
        transcript = " ".join(seg.text.strip() for seg in segments).strip()

        print(
            f"✅ faster-whisper transcribed {len(transcript)} chars "
            f"(language: {info.language}, prob: {info.language_probability:.2f})"
        )
        return transcript

    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass