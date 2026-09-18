"""
Configuration module for Tamil Misogyny Detection from Speech.
Student Research Project (Speech / Audio-only pipeline).
"""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
VIDEOS_DIR = DATA_DIR / "videos"
AUDIO_DIR = DATA_DIR / "audio"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

DEMO_DATASET_PATH = DATA_DIR / "demo_dataset.csv"
DATASET_PATH = DATA_DIR / "dataset.csv"
EXCEL_DATASET_PATH = DATA_DIR / "tamil_misogyny_dataset.xlsx"
EXCEL_DATASET_COLUMNS = [
    "video_id",
    "video_filename",
    "audio_filename",
    "asr_transcript",
    "manual_transcript",
    "label",
    "category",
    "annotation_notes",
    "created_at",
]

# Audio Extraction Settings (Strictly Audio-only)
AUDIO_SAMPLE_RATE = 16000  # 16 kHz mono standard for ASR models
AUDIO_CHANNELS = 1         # Mono
AUDIO_FORMAT = "wav"

# Supported Input Extensions
SUPPORTED_VIDEO_EXTENSIONS = [".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv"]
SUPPORTED_AUDIO_EXTENSIONS = [".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"]

# ASR Model Settings (ElevenLabs Scribe v2)
ASR_PROVIDER = "ElevenLabs"
SCRIBE_V2_MODEL_ID = "scribe_v2"
TAMIL_LANGUAGE_CODE = "tam"

# Misogyny Labels & Categories
LABEL_NON_MISOGYNISTIC = 0
LABEL_MISOGYNISTIC = 1

LABEL_MAP = {
    0: "NON-MISOGYNISTIC",
    1: "MISOGYNISTIC"
}

CATEGORIES = [
    "OBJECTIFICATION",
    "STEREOTYPING",
    "SHAMING",
    "VIOLENCE",
    "GENERAL_ABUSE",
    "NONE"
]

CATEGORY_DESCRIPTIONS = {
    "OBJECTIFICATION": "Reduction of women to sexual objects, physical appearance, or commodities.",
    "STEREOTYPING": "Imposing rigid, restrictive traditional roles or undermining women's autonomy.",
    "SHAMING": "Belittling or judging women based on clothing, choices, modesty, or moral policing.",
    "VIOLENCE": "Threats, incitement, or justification of physical, sexual, or psychological harm against women.",
    "GENERAL_ABUSE": "Derogatory remarks, slurs, insults, or demeaning statements targeting women as inferior.",
    "NONE": "Neutral, supportive, equality-oriented, or unrelated content."
}
