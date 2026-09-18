"""
Evaluation Script for Text-Based Tamil Misogyny Classifier.
Computes Accuracy, Precision, Recall, F1-score, Macro-F1, and Confusion Matrix.

IMPORTANT RESEARCH DISCLAIMER:
Do NOT report demo accuracy as research accuracy.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Force UTF-8 on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DEMO_DATASET_PATH, DATASET_PATH
from src.classifier import TamilMisogynyClassifier
from src.metrics import compute_classification_metrics


def evaluate_text_model(data_path: Path):
    """
    Evaluate Tamil text misogyny classifier on a labeled CSV dataset.
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    # Check for demo dataset usage
    is_demo = (data_path.resolve() == DEMO_DATASET_PATH.resolve())
    if is_demo:
        print("\n" + "=" * 75)
        print("[WARNING] IMPORTANT RESEARCH DISCLAIMER:")
        print("This evaluation was conducted on DEMO DATA ONLY (10 safe demonstration examples).")
        print("Do NOT call this the actual research dataset.")
        print("Do NOT report demo accuracy as research accuracy.")
        print("=" * 75 + "\n")

    print(f"[*] Loading evaluation dataset: {data_path}")
    df = pd.read_csv(data_path)

    # Support both column naming schemes
    if "id" not in df.columns and "video_id" in df.columns:
        df["id"] = df["video_id"]
    if "transcript" not in df.columns and "manual_transcript" in df.columns:
        df["transcript"] = df["manual_transcript"]

    # Ensure required columns
    for col in ["id", "transcript", "label", "category"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in dataset")

    classifier = TamilMisogynyClassifier()

    y_true_binary = []
    y_pred_binary = []
    y_true_category = []
    y_pred_category = []

    results_table = []

    print("[*] Running inference on transcripts...")
    for _, row in df.iterrows():
        sample_id = row["id"]
        transcript = row["transcript"]
        ground_truth_label = int(row["label"])
        ground_truth_cat = str(row["category"])

        res = classifier.predict(transcript)
        pred_label_int = 1 if res["label"] == "MISOGYNISTIC" else 0
        pred_cat = res["category"]

        y_true_binary.append(ground_truth_label)
        y_pred_binary.append(pred_label_int)
        y_true_category.append(ground_truth_cat)
        y_pred_category.append(pred_cat)

        results_table.append({
            "ID": sample_id,
            "Transcript": transcript[:45] + "..." if len(transcript) > 45 else transcript,
            "True Label": "MISOGYNISTIC" if ground_truth_label == 1 else "NON-MISOGYNISTIC",
            "Pred Label": res["label"],
            "True Cat": ground_truth_cat,
            "Pred Cat": pred_cat,
            "Reason": res["reason"],
            "Evidence": res["evidence"],
        })

    # Compute binary metrics
    metrics = compute_classification_metrics(y_true_binary, y_pred_binary)

    print("\n" + "=" * 50)
    print("CLASSIFICATION EVALUATION METRICS")
    print("=" * 50)
    print(f"Sample Count:    {metrics['sample_count']}")
    print(f"Accuracy:        {metrics['accuracy']:.2f}%")
    print(f"Precision:       {metrics['precision']:.2f}%")
    print(f"Recall:          {metrics['recall']:.2f}%")
    print(f"F1-Score:        {metrics['f1_score']:.2f}%")
    print(f"Macro-F1:        {metrics['f1_macro']:.2f}%")

    print("\nConfusion Matrix (Rows: True [0, 1], Cols: Pred [0, 1]):")
    cm = np.array(metrics["confusion_matrix"])
    print(f"                Pred Non-Misogynistic    Pred Misogynistic")
    print(f"True Non-Mis:          {cm[0][0]:<20}    {cm[0][1]:<20}")
    print(f"True Mis:              {cm[1][0]:<20}    {cm[1][1]:<20}")

    print("\nPer-Class Classification Report:")
    for cls_name, vals in metrics["classification_report"].items():
        if isinstance(vals, dict):
            print(f"  {cls_name:<20}: Precision={vals['precision']:.3f}, Recall={vals['recall']:.3f}, F1={vals['f1-score']:.3f}")

    print("\nDetailed Sample Predictions:")
    pred_df = pd.DataFrame(results_table)
    print(pred_df[["ID", "True Label", "Pred Label", "True Cat", "Pred Cat"]].to_string(index=False))

    if is_demo:
        print("\n" + "=" * 75)
        print("[REMINDER] Do NOT report the above demo metrics as official research results.")
        print("=" * 75)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Tamil Text Misogyny Classifier.")
    parser.add_argument(
        "--data",
        type=str,
        default=str(DEMO_DATASET_PATH),
        help="Path to evaluation dataset CSV (default: demo_dataset.csv)",
    )
    args = parser.parse_args()

    evaluate_text_model(Path(args.data))


if __name__ == "__main__":
    main()
