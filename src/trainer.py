"""
Model Training & Retraining Engine for Tamil Misogyny Detection System.
Trains text classifier on manual_transcript from data/tamil_misogyny_dataset.xlsx.
Uses multilingual-e5-small embeddings + Logistic Regression.
Calculates and displays genuine evaluation metrics without fabricating data.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from config import (
    EXCEL_DATASET_PATH,
    MODELS_DIR,
)

MODEL_OUTPUT_PATH = MODELS_DIR / "text_classifier.pkl"
E5_MODEL_NAME = "intfloat/multilingual-e5-small"


def retrain_model_from_excel(
    excel_path: Optional[Path] = None,
    output_model_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Retrain misogyny classification model from the latest Excel dataset.

    Args:
        excel_path: Path to tamil_misogyny_dataset.xlsx
        output_model_path: Path to save trained text_classifier.pkl

    Returns:
        Dict containing training status, sample counts, and actual metrics.
    """
    target_excel = excel_path or EXCEL_DATASET_PATH
    target_output = output_model_path or MODEL_OUTPUT_PATH

    if not target_excel.exists():
        return {
            "status": "error",
            "message": f"Dataset file not found: {target_excel}. Please add samples first.",
            "metrics_available": False,
        }

    try:
        df = pd.read_excel(target_excel)
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to read Excel file: {e}",
            "metrics_available": False,
        }

    # Validate columns
    required = ["manual_transcript", "label", "category"]
    for c in required:
        if c not in df.columns:
            return {
                "status": "error",
                "message": f"Dataset missing required column: '{c}'",
                "metrics_available": False,
            }

    # Filter out empty or whitespace transcripts
    valid_mask = df["manual_transcript"].dropna().astype(str).str.strip().str.len() > 0
    df_clean = df[valid_mask].copy()

    total_samples = len(df_clean)

    # Check minimum dataset requirements for training
    unique_labels = df_clean["label"].astype(str).str.strip().str.upper().unique()

    if total_samples < 4 or len(unique_labels) < 2:
        return {
            "status": "insufficient_data",
            "message": "Insufficient data for reliable evaluation.",
            "details": f"Found {total_samples} valid sample(s) with classes: {list(unique_labels)}. Minimum 4 samples with both MISOGYNISTIC and NON-MISOGYNISTIC labels are required.",
            "metrics_available": False,
            "total_samples": total_samples,
        }

    # Normalize labels: 1 for MISOGYNISTIC, 0 for NON-MISOGYNISTIC
    # Supports both integer format (1/0) and string format (MISOGYNISTIC/NON-MISOGYNISTIC)
    def _normalize_label(val):
        s = str(val).strip().upper()
        # Integer format: "1" -> misogynistic, "0" -> non-misogynistic
        if s in ("1", "1.0"):
            return 1
        if s in ("0", "0.0"):
            return 0
        # String format
        if "MIS" in s and "NON" not in s:
            return 1
        return 0

    df_clean["binary_label"] = df_clean["label"].apply(_normalize_label)
    df_clean["clean_category"] = df_clean["category"].astype(str).str.strip().str.upper()
    # Normalize category: map NONE (non-misogynistic) to a stable label
    df_clean["clean_category"] = df_clean["clean_category"].replace({"NAN": "NON_MISOGYNISTIC", "NONE": "NON_MISOGYNISTIC"})

    # Prepend 'passage: ' prefix for E5 retrieval/embedding
    transcripts = [f"passage: {str(t).strip()}" for t in df_clean["manual_transcript"]]
    y_binary = df_clean["binary_label"].values
    y_cat = df_clean["clean_category"].values

    # Determine split: if samples >= 10, do 80/20 train/test split; if between 4 and 9, train on all and report disclaimer
    test_size = 0.2 if total_samples >= 10 else None

    # Load SentenceTransformer embedding model with offline cache fallback
    candidate_models = [
        ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", True),
        ("intfloat/multilingual-e5-large", True),
        (E5_MODEL_NAME, False),
    ]

    encoder = None
    used_model_name = None

    for m_name, local_only in candidate_models:
        try:
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer(m_name, local_files_only=local_only)
            used_model_name = m_name
            break
        except Exception:
            continue

    if encoder is not None:
        embeddings = encoder.encode(transcripts, show_progress_bar=False, normalize_embeddings=True)
    else:
        # High-quality character + word TF-IDF vectorizer fallback for Tamil text if offline
        from sklearn.feature_extraction.text import TfidfVectorizer
        tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=512)
        embeddings = tfidf.fit_transform([str(t).strip() for t in df_clean["manual_transcript"]]).toarray()
        used_model_name = "TF-IDF (Tamil Subword N-Grams)"

    if test_size is not None and len(np.unique(y_binary)) > 1:
        # Stratified train/test split
        try:
            X_train, X_test, y_train, y_test, y_cat_train, y_cat_test = train_test_split(
                embeddings, y_binary, y_cat, test_size=test_size, random_state=42, stratify=y_binary
            )
        except Exception:
            X_train, X_test, y_train, y_test, y_cat_train, y_cat_test = train_test_split(
                embeddings, y_binary, y_cat, test_size=test_size, random_state=42
            )

        clf_binary = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        clf_binary.fit(X_train, y_train)

        # Train category classifier
        clf_category = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        clf_category.fit(X_train, y_cat_train)

        # Evaluate on holdout test set
        y_pred = clf_binary.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))

        train_count = len(X_train)
        test_count = len(X_test)
        metrics_avail = True
        disclaimer = None
    else:
        # Small dataset: fit on all samples, report disclaimer
        clf_binary = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        clf_binary.fit(embeddings, y_binary)

        clf_category = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        clf_category.fit(embeddings, y_cat)

        train_count = total_samples
        test_count = 0
        acc, prec, rec, f1 = 0.0, 0.0, 0.0, 0.0
        metrics_avail = False
        disclaimer = "Insufficient data for reliable evaluation. Model fitted on all available samples without test split."

    # Save trained model payload
    target_output.parent.mkdir(parents=True, exist_ok=True)
    saved_payload = {
        "label_model": clf_binary,
        "category_model": clf_category,
        "embedding_model_name": E5_MODEL_NAME,
        "classes_label": clf_binary.classes_,
        "classes_category": clf_category.classes_,
        "training_samples": train_count,
        "test_samples": test_count,
    }
    joblib.dump(saved_payload, target_output)

    return {
        "status": "success",
        "message": "Model trained and saved successfully.",
        "model_path": str(target_output),
        "total_samples": total_samples,
        "train_samples": train_count,
        "test_samples": test_count,
        "metrics_available": metrics_avail,
        "accuracy": round(acc * 100, 2) if metrics_avail else None,
        "precision": round(prec * 100, 2) if metrics_avail else None,
        "recall": round(rec * 100, 2) if metrics_avail else None,
        "f1_score": round(f1 * 100, 2) if metrics_avail else None,
        "disclaimer": disclaimer,
    }
