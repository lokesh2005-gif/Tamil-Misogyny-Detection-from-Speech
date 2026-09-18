"""
Text-Based Tamil Misogyny Classifier.
Strictly audio transcript text classification. NO computer vision or image models.

Uses multilingual-e5-small for multilingual semantic embeddings, paired with
Logistic Regression when trained, and a prototype semantic/contextual classifier
for initial demonstration.

Categories:
1. SHAMING
2. STEREOTYPING
3. OBJECTIFICATION
4. VIOLENCE
5. GENERAL_ABUSE
6. NONE
"""

import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
import warnings
import numpy as np
import joblib

from config import (
    MODELS_DIR,
    LABEL_MAP,
)

E5_MODEL_NAME = "intfloat/multilingual-e5-small"
MODEL_SAVE_PATH = MODELS_DIR / "text_classifier.pkl"

# Categories specified for Phase 4
CATEGORIES = [
    "SHAMING",
    "STEREOTYPING",
    "OBJECTIFICATION",
    "VIOLENCE",
    "GENERAL_ABUSE",
    "NONE",
]

# Gender words that alone DO NOT constitute misogyny
NEUTRAL_GENDER_WORDS = [
    "பெண்", "பெண்கள்", "பெண்களும்", "சிறுமி", "மனைவி", "தாய்", "அம்மா",
    "சகோதரி", "மாணவி", "மகள்", "woman", "girl", "female", "wife", "mother"
]

# Category definitions used for semantic contextual prototype
CATEGORY_CONTEXTS = {
    "OBJECTIFICATION": {
        "description": "Reduction of women to sexual objects, bodily measurements, physical appearance, or decorative value.",
        "cues": ["அழகில் மட்டும்", "அழகு", "உடல்", "தோற்றம்", "பொருளாக", "கவர்ச்சி"],
        "reason": "The statement reduces women to physical appearance or objectifying attributes.",
    },
    "STEREOTYPING": {
        "description": "Imposing restrictive traditional roles, kitchen confinement, or domestic limitations on women.",
        "cues": [
            "பொம்பளன்னா அப்படித்தான்",
            "வேலையும் செஞ்சுதான் ஆகணும்",
            "செஞ்சுதான் ஆகணும்",
            "அடக்க ஒடுக்கமா",
            "பொம்பளன்னா",
            "வாய் நீண்டுட்டு",
            "வீட்டை மட்டும்",
            "வீட்டு வேலை",
            "அடுப்பூதும்",
            "சமையல் மட்டும்",
            "பாலின பாகுபாடு",
            "பெண்கள் அடங்கி",
        ],
        "reason": "The statement describes a restrictive gender stereotype limiting women's autonomy.",
    },
    "SHAMING": {
        "description": "Judging, belittling, or moral policing women based on dress, attire, modesty, or lifestyle choices.",
        "cues": ["உடையை வைத்து", "அவமதிப்பது", "ஆடை ஒழுக்கம்", "கற்பு", "அசிங்கம்", "கேலி", "ஒழுக்கம் கெட்டவ", "மரியாதை கெட்டு"],
        "reason": "The statement concerns appearance or modesty-based shaming.",
    },
    "VIOLENCE": {
        "description": "Advocating, encouraging, threatening, or normalizing violence or physical abuse against women.",
        "cues": ["வன்முறையை ஆதரிப்பது", "வன்முறை", "அடிப்பது", "தாக்குவது", "கொலை", "மிரட்டல்", "துன்புறுத்துவது", "சாவடிக்குறேன்"],
        "reason": "The statement discusses or supports violence against women.",
    },
    "GENERAL_ABUSE": {
        "description": "Demeaning statements claiming women are inferior, incapable, or expressing contempt toward women.",
        "cues": ["குறைவானவர்கள்", "ஆண்களை விட குறைவு", "மூளை இல்லை", "தகுதியற்றவர்கள்", "இழிவான", "மூளை கிடையாது", "எத்தயத்த பேசிட்டு"],
        "reason": "The statement expresses an inferior or derogatory view of women.",
    },
}

# Positive equality & neutral activity indicators
EQUALITY_NEUTRAL_INDICATORS = [
    "சமமாக கல்வி", "சம உரிமை", "மரியாதை", "பொறியாளர்", "மருத்துவமனை",
    "விளையாட்டு", "சிகிச்சை", "சம வாய்ப்பு", "முன்னேற்றம்", "வேலை செய்கிறார்",
    "படிக்கிறார்", "சாதிக்க", "சமத்துவம்"
]


class TamilMisogynyClassifier:
    """
    Text-Based Tamil Misogyny Classifier.
    Evaluates Tamil speech transcripts strictly from text without visual data.
    """

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or MODEL_SAVE_PATH
        self.trained_pipeline = None
        self.encoder = None
        self._load_trained_model()

    def _load_trained_model(self):
        """Loads trained Logistic Regression model if exists on disk."""
        if self.model_path.exists():
            try:
                self.trained_pipeline = joblib.load(self.model_path)
            except Exception as e:
                warnings.warn(f"Could not load saved model from {self.model_path}: {e}")
                self.trained_pipeline = None

    def _get_encoder(self):
        """Lazy loads sentence-transformers encoder with offline fallback."""
        if self.encoder is None:
            model_name = None
            if self.trained_pipeline and "embedding_model_name" in self.trained_pipeline:
                model_name = self.trained_pipeline["embedding_model_name"]

            candidate_models = [
                (model_name, True) if model_name else None,
                ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", True),
                ("intfloat/multilingual-e5-large", True),
                (E5_MODEL_NAME, False),
            ]

            for entry in candidate_models:
                if not entry:
                    continue
                m_name, local_only = entry
                try:
                    from sentence_transformers import SentenceTransformer
                    self.encoder = SentenceTransformer(m_name, local_files_only=local_only)
                    break
                except Exception:
                    continue

        return self.encoder

    def _extract_evidence(self, transcript: str, category: str) -> Optional[str]:
        """Finds actual evidence substring in transcript. Never invents evidence."""
        if category not in CATEGORY_CONTEXTS:
            return None
        for cue in CATEGORY_CONTEXTS[category]["cues"]:
            if cue in transcript:
                return cue
        return None

    def predict(self, transcript: str) -> Dict[str, str]:
        """
        Classify Tamil transcript into MISOGYNISTIC or NON-MISOGYNISTIC,
        assign category and short explanation.

        Follows critical rules:
        1. Words like 'woman' / 'girl' alone do not constitute misogyny.
        2. Neutral statements about women are NOT misogyny.
        3. Never invent evidence.
        4. If evidence is insufficient:
           label = NON-MISOGYNISTIC, category = NONE, reason = "Insufficient evidence."

        Returns:
            {
                "label": "MISOGYNISTIC" or "NON-MISOGYNISTIC",
                "category": "OBJECTIFICATION" | "STEREOTYPING" | "SHAMING" | "VIOLENCE" | "GENERAL_ABUSE" | "NONE",
                "reason": "...",
                "evidence": "..."
            }
        """
        if not transcript or not transcript.strip():
            return {
                "label": "NON-MISOGYNISTIC",
                "category": "NONE",
                "reason": "Insufficient evidence.",
                "evidence": "",
            }

        text = transcript.strip()

        # If a trained model is present, use it
        if self.trained_pipeline is not None:
            try:
                enc = self._get_encoder()
                if enc is not None:
                    emb = enc.encode([f"query: {text}"])
                    pred_label = self.trained_pipeline["label_model"].predict(emb)[0]
                    pred_cat = self.trained_pipeline["category_model"].predict(emb)[0]

                    label_str = "MISOGYNISTIC" if pred_label == 1 else "NON-MISOGYNISTIC"

                    # Check contextual cues for exact category matching and evidence extraction
                    cue_cat = None
                    cue_ev = None
                    for cat, data in CATEGORY_CONTEXTS.items():
                        for cue in data["cues"]:
                            if cue in text:
                                cue_cat = cat
                                cue_ev = cue
                                break
                        if cue_cat:
                            break

                    # If an explicit misogynistic cue is present, ensure it is classified as MISOGYNISTIC
                    if cue_cat:
                        label_str = "MISOGYNISTIC"
                        pred_cat = cue_cat
                        evidence = cue_ev
                        reason = CATEGORY_CONTEXTS[pred_cat]["reason"]
                    elif label_str == "NON-MISOGYNISTIC":
                        pred_cat = "NONE"
                        # Standardize reason for neutral / non-misogynistic statements
                        is_explicitly_neutral = any(ind in text for ind in EQUALITY_NEUTRAL_INDICATORS)
                        reason = "The statement is neutral or supports equality." if is_explicitly_neutral else "Insufficient evidence."
                        evidence = ""
                    else:
                        evidence = self._extract_evidence(text, pred_cat) or ""
                        reason = CATEGORY_CONTEXTS.get(pred_cat, {}).get("reason", "Contains misogynistic rhetoric.")


                        reason = CATEGORY_CONTEXTS.get(pred_cat, {}).get("reason", "Contains misogynistic rhetoric.")

                    return {
                        "label": label_str,
                        "category": pred_cat,
                        "reason": reason,
                        "evidence": evidence,
                    }
            except Exception as e:
                warnings.warn(f"Trained model inference failed: {e}. Falling back to prototype semantic matcher.")

        # PROTOTYPE SEMANTIC / CONTEXTUAL CLASSIFIER
        # Check for equality or neutral activity indicators
        is_explicitly_neutral = any(ind in text for ind in EQUALITY_NEUTRAL_INDICATORS)

        # Search for actual contextual cues for the 5 misogyny categories
        detected_category = None
        detected_evidence = None

        for cat, data in CATEGORY_CONTEXTS.items():
            for cue in data["cues"]:
                if cue in text:
                    detected_category = cat
                    detected_evidence = cue
                    break
            if detected_category:
                break

        # Rule: A neutral statement about a woman is NOT misogyny
        if detected_category is None:
            # Contains only neutral gender words or general text without misogynistic rhetoric
            return {
                "label": "NON-MISOGYNISTIC",
                "category": "NONE",
                "reason": "The statement is neutral or supports equality." if is_explicitly_neutral else "Insufficient evidence.",
                "evidence": "",
            }

        # If detected cues are present and not overridden
        reason_text = CATEGORY_CONTEXTS[detected_category]["reason"]

        return {
            "label": "MISOGYNISTIC",
            "category": detected_category,
            "reason": reason_text,
            "evidence": detected_evidence or "",
        }

    def explain(self, transcript: str, pred: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generates an in-depth LLM explanation using Google Gemini for why
        the statement is classified as MISOGYNISTIC or NON-MISOGYNISTIC.
        """
        from src.explainer import explain_misogyny_with_gemini

        if pred is None:
            pred = self.predict(transcript)

        return explain_misogyny_with_gemini(
            transcript=transcript,
            label=pred.get("label", "NON-MISOGYNISTIC"),
            category=pred.get("category", "NONE"),
            evidence=pred.get("evidence", ""),
        )
