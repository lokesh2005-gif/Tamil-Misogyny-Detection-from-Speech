"""
ASR and Speech Evaluation Metrics Module.
Computes Word Error Rate (WER) and Character Error Rate (CER) using jiwer.
Evaluates Substitutions (S), Insertions (I), and Deletions (D) with exact research formulation:
WER = (S + I + D) / N

IMPORTANT: Reference transcript is NOT modified before calculating metrics.
"""

from typing import Dict, Any, List, Union
import jiwer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def calculate_wer(reference: str, hypothesis: str) -> float:
    """
    Calculate Word Error Rate (WER) using jiwer.
    Does NOT modify or preprocess reference text.

    Args:
        reference: Ground truth manual Tamil transcription.
        hypothesis: ASR generated Tamil transcription.

    Returns:
        WER as a decimal float (e.g., 0.125 for 12.5%).
    """
    return float(jiwer.wer(reference, hypothesis))


def calculate_cer(reference: str, hypothesis: str) -> float:
    """
    Calculate Character Error Rate (CER) using jiwer.
    Measures character-level transcription errors.

    Args:
        reference: Ground truth manual Tamil transcription.
        hypothesis: ASR generated Tamil transcription.

    Returns:
        CER as a decimal float.
    """
    return float(jiwer.cer(reference, hypothesis))


def calculate_error_details(reference: str, hypothesis: str) -> Dict[str, Any]:
    """
    Calculate detailed ASR error breakdown including Substitutions,
    Insertions, and Deletions using jiwer.

    Formula:
        WER = (S + I + D) / N
        where:
        S = substitutions
        I = insertions
        D = deletions
        N = reference words (S + D + Hits)

    IMPORTANT:
        The reference transcript is evaluated as-is without modification.

    Returns:
        Dict with keys: wer, wer_percent, cer, cer_percent,
        substitutions, insertions, deletions, hits, reference_words, formula.
    """
    word_output = jiwer.process_words(reference, hypothesis)
    char_output = jiwer.process_characters(reference, hypothesis)

    s = int(word_output.substitutions)
    i = int(word_output.insertions)
    d = int(word_output.deletions)
    hits = int(word_output.hits)
    n = s + d + hits

    # WER formula: (S + I + D) / N
    wer_calculated = (s + i + d) / n if n > 0 else 0.0
    cer_calculated = float(char_output.cer)

    return {
        "wer": float(wer_calculated),
        "wer_percent": round(wer_calculated * 100.0, 2),
        "cer": cer_calculated,
        "cer_percent": round(cer_calculated * 100.0, 2),
        "substitutions": s,
        "insertions": i,
        "deletions": d,
        "hits": hits,
        "reference_words": n,
        "reference_chars": len(reference),
        "formula": "WER = (S + I + D) / N",
    }


def compute_asr_metrics(reference_texts: Union[str, List[str]], hypothesis_texts: Union[str, List[str]]) -> Dict[str, float]:
    """High-level batch wrapper for ASR metrics."""
    if isinstance(reference_texts, str):
        reference_texts = [reference_texts]
    if isinstance(hypothesis_texts, str):
        hypothesis_texts = [hypothesis_texts]

    wer = jiwer.wer(reference_texts, hypothesis_texts)
    cer = jiwer.cer(reference_texts, hypothesis_texts)

    return {
        "wer": round(float(wer) * 100.0, 2),
        "cer": round(float(cer) * 100.0, 2),
        "wer_decimal": round(float(wer), 4),
        "cer_decimal": round(float(cer), 4),
        "sample_count": len(reference_texts),
    }


def compute_classification_metrics(
    y_true: List[int],
    y_pred: List[int],
    target_names: List[str] = ["NON-MISOGYNISTIC", "MISOGYNISTIC"],
) -> Dict[str, Any]:
    """Classification metrics helper (maintained for pipeline compatibility)."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": round(float(acc) * 100.0, 2),
        "precision": round(float(prec) * 100.0, 2),
        "recall": round(float(rec) * 100.0, 2),
        "f1_score": round(float(f1) * 100.0, 2),
        "f1_macro": round(float(f1_macro) * 100.0, 2),
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
        "sample_count": len(y_true),
    }
