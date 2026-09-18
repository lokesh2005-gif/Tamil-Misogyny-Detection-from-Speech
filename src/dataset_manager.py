"""
Dataset Management Module for Tamil Misogyny Detection System (Training Studio).
Handles continuous data collection, Excel storage (data/tamil_misogyny_dataset.xlsx),
unique ID generation (VID_0001, VID_0002...), and dataset queries.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

from config import (
    EXCEL_DATASET_PATH,
    EXCEL_DATASET_COLUMNS,
    DATA_DIR,
)


def initialize_excel_dataset() -> pd.DataFrame:
    """
    Ensure data/tamil_misogyny_dataset.xlsx exists with standard schema.
    Returns the loaded or freshly initialized DataFrame.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if EXCEL_DATASET_PATH.exists():
        try:
            df = pd.read_excel(EXCEL_DATASET_PATH)
            # Verify columns
            for col in EXCEL_DATASET_COLUMNS:
                if col not in df.columns:
                    df[col] = ""
            return df[EXCEL_DATASET_COLUMNS]
        except Exception:
            pass

    # Create empty dataset with standard columns
    df = pd.DataFrame(columns=EXCEL_DATASET_COLUMNS)
    df.to_excel(EXCEL_DATASET_PATH, index=False, engine="openpyxl")
    return df


def load_excel_dataset() -> pd.DataFrame:
    """Load the current training dataset from Excel."""
    if not EXCEL_DATASET_PATH.exists():
        return initialize_excel_dataset()

    try:
        df = pd.read_excel(EXCEL_DATASET_PATH)
        for col in EXCEL_DATASET_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        
        # Ensure labels are normalized strings for UI consistency
        def _to_str_label(val):
            s = str(val).strip().upper()
            if s in ("1", "1.0"):
                return "MISOGYNISTIC"
            if s in ("0", "0.0"):
                return "NON-MISOGYNISTIC"
            if "MIS" in s and "NON" not in s:
                return "MISOGYNISTIC"
            return "NON-MISOGYNISTIC"

        if "label" in df.columns:
            df["label"] = df["label"].apply(_to_str_label)
        return df
    except Exception:
        return initialize_excel_dataset()



def generate_next_video_id(df: Optional[pd.DataFrame] = None) -> str:
    """
    Generate the next unique sequential video ID.
    Format: VID_0001, VID_0002, VID_0003...
    """
    if df is None:
        df = load_excel_dataset()

    if df.empty or "video_id" not in df.columns:
        return "VID_0001"

    max_num = 0
    for vid in df["video_id"].dropna().astype(str):
        clean_id = vid.strip()
        if clean_id.startswith("VID_"):
            try:
                num = int(clean_id[4:])
                if num > max_num:
                    max_num = num
            except ValueError:
                continue

    return f"VID_{max_num + 1:04d}"


def append_training_sample(
    video_filename: str,
    audio_filename: str,
    asr_transcript: str,
    manual_transcript: str,
    label: str,
    category: str,
    annotation_notes: str = "",
) -> Dict[str, Any]:
    """
    Append a verified Tamil speech sample to the Excel training dataset.
    Never overwrites previous records.

    Args:
        video_filename: Filename of uploaded video (e.g., speech_clip.mp4)
        audio_filename: Filename of extracted WAV audio
        asr_transcript: Scribe v2 generated Tamil transcript
        manual_transcript: Human verified / corrected Tamil transcript
        label: "MISOGYNISTIC" or "NON-MISOGYNISTIC"
        category: "SHAMING" | "STEREOTYPING" | "OBJECTIFICATION" | "VIOLENCE" | "GENERAL_ABUSE" | "NONE"
        annotation_notes: Optional user annotation notes

    Returns:
        Dict of the added sample record including generated video_id.
    """
    df = load_excel_dataset()
    new_video_id = generate_next_video_id(df)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Clean text values
    clean_manual = str(manual_transcript).strip() if manual_transcript else ""
    clean_asr = str(asr_transcript).strip() if asr_transcript else clean_manual
    clean_label = str(label).strip().upper()
    clean_cat = str(category).strip().upper()

    # Safety: If non-misogynistic, category is always NONE
    if clean_label == "NON-MISOGYNISTIC":
        clean_cat = "NONE"

    new_record = {
        "video_id": new_video_id,
        "video_filename": Path(video_filename).name if video_filename else "video.mp4",
        "audio_filename": Path(audio_filename).name if audio_filename else "audio.wav",
        "asr_transcript": clean_asr,
        "manual_transcript": clean_manual,
        "label": clean_label,
        "category": clean_cat,
        "annotation_notes": str(annotation_notes).strip() if annotation_notes else "",
        "created_at": now_str,
    }

    new_df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)
    new_df.to_excel(EXCEL_DATASET_PATH, index=False, engine="openpyxl")

    return new_record


def get_dataset_summary() -> Dict[str, Any]:
    """
    Compute real dataset summary metrics for dashboard cards.
    Handles both integer labels (0/1) and string labels (MISOGYNISTIC/NON-MISOGYNISTIC).
    """
    if not EXCEL_DATASET_PATH.exists():
        return {
            "total_samples": 0,
            "misogynistic_count": 0,
            "non_misogynistic_count": 0,
            "category_distribution": {},
            "is_empty": True,
        }

    try:
        df = pd.read_excel(EXCEL_DATASET_PATH)
    except Exception:
        return {
            "total_samples": 0,
            "misogynistic_count": 0,
            "non_misogynistic_count": 0,
            "category_distribution": {},
            "is_empty": True,
        }

    total = len(df)

    if total == 0 or "label" not in df.columns:
        return {
            "total_samples": 0,
            "misogynistic_count": 0,
            "non_misogynistic_count": 0,
            "category_distribution": {},
            "is_empty": True,
        }

    # Normalize labels — handles integer (0/1) and string formats
    def _is_misogynistic(val) -> bool:
        s = str(val).strip().upper()
        if s in ("1", "1.0"):
            return True
        if s in ("0", "0.0"):
            return False
        return "MIS" in s and "NON" not in s

    misogynistic_count   = int(df["label"].apply(_is_misogynistic).sum())
    non_misogynistic_count = total - misogynistic_count

    category_counts: Dict[str, int] = {}
    if "category" in df.columns:
        raw_cats = df["category"].dropna().astype(str).str.strip().str.upper()
        raw_cats = raw_cats.replace({"NAN": "NONE", "": "NONE"})
        category_counts = raw_cats.value_counts().to_dict()

    return {
        "total_samples": total,
        "misogynistic_count": misogynistic_count,
        "non_misogynistic_count": non_misogynistic_count,
        "category_distribution": category_counts,
        "is_empty": False,
    }



def seed_dataset_if_empty() -> int:
    """
    If the Excel dataset has fewer than 5 samples, populate with verified
    initial demonstration samples from demo_dataset.csv so training can be run.
    """
    from config import DEMO_DATASET_PATH

    df = load_excel_dataset()
    if len(df) >= 5:
        return len(df)

    if not DEMO_DATASET_PATH.exists():
        return len(df)

    demo_df = pd.read_csv(DEMO_DATASET_PATH)
    added = 0
    for _, row in demo_df.iterrows():
        t = str(row["transcript"]).strip()
        l = "MISOGYNISTIC" if row["label"] == 1 else "NON-MISOGYNISTIC"
        c = str(row["category"]).strip()
        # Avoid exact duplicate text
        if not df.empty and t in df["manual_transcript"].values:
            continue
        append_training_sample(
            video_filename=f"demo_video_{row['id']}.mp4",
            audio_filename=f"demo_audio_{row['id']}.wav",
            asr_transcript=t,
            manual_transcript=t,
            label=l,
            category=c,
            annotation_notes=f"Demonstration seed: {row.get('reason', '')}",
        )
        added += 1

    return len(load_excel_dataset())
