"""
Training Script for Text-Based Tamil Misogyny Classifier.
Trains Logistic Regression models on multilingual-e5-small semantic embeddings
for future real annotated research datasets.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression

# Force UTF-8 on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    DATA_DIR,
    MODELS_DIR,
    DATASET_PATH,
    DEMO_DATASET_PATH,
)

E5_MODEL_NAME = "intfloat/multilingual-e5-small"
OUTPUT_MODEL_PATH = MODELS_DIR / "text_classifier.pkl"


def train_model(data_path: Path, output_path: Path):
    """
    Train text classifier on Tamil transcript dataset using multilingual-e5-small + Logistic Regression.
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {data_path}")

    # Important research disclaimer check
    if data_path.resolve() == DEMO_DATASET_PATH.resolve():
        print("=" * 70)
        print("[WARNING] TRAINING ON DEMO DATASET!")
        print("This is DEMO DATA only. Do not call it the actual research dataset.")
        print("Do not report demo accuracy as research accuracy.")
        print("=" * 70)

    print(f"[*] Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)

    # Support both column naming schemes
    if "transcript" not in df.columns and "manual_transcript" in df.columns:
        df["transcript"] = df["manual_transcript"]

    # Validate columns
    required_cols = ["transcript", "label", "category"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Dataset missing required column: '{col}'")

    print(f"[*] Total samples: {len(df)}")
    print(f"[*] Class distribution:\n{df['label'].value_counts()}")
    print(f"[*] Category distribution:\n{df['category'].value_counts()}")

    # Load sentence-transformers multilingual-e5-small
    print(f"[*] Loading embedding model: {E5_MODEL_NAME}...")
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(E5_MODEL_NAME)

    # In E5 models, inputs are prefixed with 'passage: ' for retrieval/classification
    transcripts = [f"passage: {t.strip()}" for t in df["transcript"]]
    print("[*] Generating multilingual embeddings...")
    embeddings = encoder.encode(transcripts, show_progress_bar=True, normalize_embeddings=True)

    y_label = df["label"].values
    y_category = df["category"].values

    # Train Logistic Regression for binary misogyny classification
    print("[*] Training Logistic Regression (Binary Misogyny Label)...")
    clf_label = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    clf_label.fit(embeddings, y_label)

    # Train Logistic Regression for multi-class category classification
    print("[*] Training Logistic Regression (Category Label)...")
    clf_category = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    clf_category.fit(embeddings, y_category)

    # Package and save
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    saved_payload = {
        "label_model": clf_label,
        "category_model": clf_category,
        "embedding_model_name": E5_MODEL_NAME,
        "classes_label": clf_label.classes_,
        "classes_category": clf_category.classes_,
        "training_samples": len(df),
    }

    joblib.dump(saved_payload, output_path)
    print(f"[+] Successfully trained and saved text classifier to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Tamil Text Misogyny Classifier.")
    parser.add_argument(
        "--data",
        type=str,
        default=str(DEMO_DATASET_PATH),
        help="Path to training dataset CSV (default: demo_dataset.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_MODEL_PATH),
        help="Path to save trained text_classifier.pkl",
    )
    args = parser.parse_args()

    train_model(Path(args.data), Path(args.output))


if __name__ == "__main__":
    main()
