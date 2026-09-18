"""
Explanation Module.
Generates research explanations and category rationales for Tamil misogyny classifications.
Runs entirely locally without any external LLM APIs.
"""

from typing import Dict, Any, List
from config import CATEGORY_DESCRIPTIONS, LABEL_MAP


class MisogynyExplainer:
    """
    Produces deterministic, research-oriented rationales explaining why
    a Tamil transcript was categorized as Misogynistic or Non-Misogynistic.
    """

    REASON_TEMPLATES = {
        "OBJECTIFICATION": "The statement discusses the reduction of women to physical appearance or objectifying attributes.",
        "STEREOTYPING": "The statement describes a restrictive gender stereotype or traditional confinement of women.",
        "SHAMING": "The statement concerns modesty or appearance-based shaming and moral policing of women.",
        "VIOLENCE": "The statement discusses or references violence, abuse, or physical aggression against women.",
        "GENERAL_ABUSE": "The statement expresses an inferior or derogatory view targeting women.",
        "NONE": "The statement is neutral, descriptive, or supports gender equality without misogynistic rhetoric."
    }

    # Contextual nuances for specific topics
    CONTEXTUAL_RULES = [
        ("கல்வி", "The statement supports equal education for women."),
        ("சம உரிமை", "The statement advocates for equal rights and mutual respect."),
        ("விளையாட்டு", "The statement neutrally describes participation in sports or social activities."),
        ("பொறியாளர்", "The statement is a neutral professional description."),
        ("மருத்துவமனை", "The statement neutrally refers to healthcare and patient care."),
        ("அழகில் மட்டும் இல்லை", "The statement discusses the reduction of women to physical appearance."),
        ("வீட்டை மட்டும் கவனிக்க", "The statement describes a restrictive gender stereotype."),
        ("உடையை வைத்து அவளை அவமதிப்பது", "The statement concerns appearance-based shaming."),
        ("வன்முறையை ஆதரிப்பது", "The statement discusses support for violence against women."),
        ("ஆண்களை விட குறைவானவர்கள்", "The statement expresses an inferior view of women.")
    ]

    def explain(self, prediction_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive explanation including reason, category meaning,
        and linguistic rationale.

        Args:
            prediction_result: Output from TamilMisogynyClassifier.predict().

        Returns:
            Dict containing formatted reason, category explanation, and evidence summary.
        """
        category = prediction_result.get("category", "NONE")
        label = prediction_result.get("label", 0)
        label_str = prediction_result.get("label_str", LABEL_MAP.get(label, "UNKNOWN"))
        transcript = prediction_result.get("transcript", "")
        matched_cues = prediction_result.get("matched_cues", [])

        # Check for specific contextual rule match
        reason = None
        for pattern, specific_reason in self.CONTEXTUAL_RULES:
            if pattern in transcript:
                reason = specific_reason
                break

        if not reason:
            reason = self.REASON_TEMPLATES.get(category, "No specific misogynistic markers identified.")

        category_desc = CATEGORY_DESCRIPTIONS.get(category, "No category description available.")

        return {
            "label": label,
            "label_str": label_str,
            "category": category,
            "category_description": category_desc,
            "reason": reason,
            "matched_cues": matched_cues,
            "confidence": prediction_result.get("confidence", 0.0),
            "summary": f"Classified as **{label_str}** [{category}] - {reason}"
        }
