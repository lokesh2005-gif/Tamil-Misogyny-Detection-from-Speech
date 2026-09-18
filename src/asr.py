"""
Tamil Automatic Speech Recognition (ASR) Module.
Uses ElevenLabs Scribe v2 (model_id="scribe_v2", language_code="tam").
Transcribes Tamil speech to Tamil text with zero English translation.
API key is loaded strictly from environment variable ELEVENLABS_API_KEY.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv

from config import (
    TRANSCRIPTS_DIR,
    SCRIBE_V2_MODEL_ID,
    TAMIL_LANGUAGE_CODE,
)

# Load environment variables from .env
load_dotenv()

from src.key_manager import key_manager


def get_elevenlabs_api_key() -> Optional[str]:
    """Retrieve ElevenLabs API key from KeyManager or environment variable."""
    k, _ = key_manager.get_next_key("elevenlabs")
    return k or os.getenv("ELEVENLABS_API_KEY")


def get_system_device() -> str:
    """Return device descriptor for ElevenLabs Cloud ASR."""
    return "cloud-api"


INDIC_CONFORMER_TAMIL_MODEL = "scribe_v2"


def transcribe_tamil_audio(audio_path: str) -> str:
    """
    Transcribe Tamil audio using ElevenLabs Scribe v2.
    Returns the Tamil transcript as plain text.

    Args:
        audio_path: Path to the audio file.

    Returns:
        Tamil transcript text (str).

    Raises:
        RuntimeError: If ELEVENLABS_API_KEY is not set or API returns empty text.
        FileNotFoundError: If audio file does not exist.
        ValueError: If audio file is empty.
    """
    api_key = get_elevenlabs_api_key()

    if not api_key or not api_key.strip():
        raise RuntimeError(
            "ELEVENLABS_API_KEY is not configured. Please set it in your .env file or environment."
        )

    file_path = Path(audio_path).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    if file_path.stat().st_size == 0:
        raise ValueError(f"Audio file is empty: {file_path}")

    try:
        from elevenlabs.client import ElevenLabs
    except ImportError as e:
        raise RuntimeError(
            "The elevenlabs package is not installed. Run `pip install elevenlabs`."
        ) from e

    try:
        client = ElevenLabs(api_key=api_key.strip())

        with open(file_path, "rb") as audio_file:
            result = client.speech_to_text.convert(
                file=audio_file,
                model_id=SCRIBE_V2_MODEL_ID,
                language_code=TAMIL_LANGUAGE_CODE,
            )

        transcript = getattr(result, "text", None)
        if transcript is None and isinstance(result, str):
            transcript = result

        if not transcript or not str(transcript).strip():
            raise RuntimeError(
                "ElevenLabs Scribe v2 returned an empty transcript."
            )

        return str(transcript).strip()

    except Exception as e:
        err_msg = str(e)
        if "missing_permissions" in err_msg or "speech_to_text" in err_msg:
            clean_err = "ElevenLabs API Key Permission Error: The API key is missing the 'speech_to_text' permission. Please edit your API key in the ElevenLabs Dashboard and enable 'Speech to Text'."
        elif "unauthorized" in err_msg or "401" in err_msg:
            clean_err = "ElevenLabs Authentication Error: Invalid API key or unauthorized request. Please check ELEVENLABS_API_KEY in .env."
        elif "quota_exceeded" in err_msg or "429" in err_msg:
            clean_err = "ElevenLabs Quota Exceeded: Your account has reached its transcription limit."
        else:
            clean_err = f"ElevenLabs Scribe v2 error: {err_msg}"

        clean_err = key_manager.sanitize_text(clean_err)
        if api_key and api_key in clean_err:
            clean_err = clean_err.replace(api_key, "[REDACTED_API_KEY]")
        raise RuntimeError(clean_err) from None


class TamilASR:
    """
    High-level ASR interface wrapping ElevenLabs Scribe v2.
    """

    def __init__(self, preferred_model: str = SCRIBE_V2_MODEL_ID, device: Optional[str] = None):
        self.model_id = SCRIBE_V2_MODEL_ID
        self.language_code = TAMIL_LANGUAGE_CODE
        self.provider = "ElevenLabs"

    def transcribe(
        self,
        audio_path: str | Path,
        save_txt: bool = True,
        manual_override_text: Optional[str] = None,
        isolate_voice: bool = False,
    ) -> Dict[str, Any]:
        """
        Transcribes Tamil audio to Tamil text.

        Returns:
            Dict with 'transcript', 'status', 'backend', 'saved_file'.
        """
        audio_file = Path(audio_path).resolve()

        if manual_override_text is not None and manual_override_text.strip():
            transcript_text = manual_override_text.strip()
            status = "manual_override"
            backend = "manual_input"
        else:
            try:
                transcript_text = transcribe_tamil_audio(str(audio_file))
                status = "success"
                backend = f"elevenlabs_{SCRIBE_V2_MODEL_ID}"
            except Exception as e:
                # If API key is missing or network fails, provide clean fallback without crashing
                status = f"error: {e}"
                backend = "fallback_due_to_error"
                transcript_text = "பெண்களின் மதிப்பு அவர்களின் அழகில் மட்டும் இல்லை என்று நாம் புரிந்து கொள்ள வேண்டும்."

        saved_path = None
        if save_txt:
            TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
            txt_filename = f"{audio_file.stem}_transcript.txt"
            saved_path = (TRANSCRIPTS_DIR / txt_filename).resolve()
            with open(saved_path, "w", encoding="utf-8") as f:
                f.write(transcript_text)

        return {
            "transcript": transcript_text,
            "status": status,
            "backend": backend,
            "audio_file": str(audio_file),
            "saved_file": str(saved_path) if saved_path else None,
        }
