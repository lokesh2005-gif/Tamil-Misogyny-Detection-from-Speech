"""
Gemini LLM Explainer Module for Tamil Misogyny Detection.
Explains WHY a Tamil speech transcript is classified as MISOGYNISTIC or NON-MISOGYNISTIC,
pinpointing linguistic evidence, cultural context, and providing clear explanations
in both English and Tamil.
"""

import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

from src.key_manager import key_manager

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


def get_gemini_api_key() -> Optional[str]:
    """Retrieve Gemini API key from KeyManager or environment variable."""
    k, _ = key_manager.get_next_key("gemini")
    return k or os.getenv("GEMINI_API_KEY")


def is_gemini_available() -> bool:
    """Check if any Gemini API key is configured and accessible."""
    return key_manager.has_keys("gemini") or bool(os.getenv("GEMINI_API_KEY"))


def explain_misogyny_with_gemini(
    transcript: str,
    label: str,
    category: str,
    evidence: str = "",
    model_name: str = DEFAULT_GEMINI_MODEL,
) -> Dict[str, Any]:
    """
    Generate an in-depth LLM explanation of why a Tamil statement is classified as
    MISOGYNISTIC or NON-MISOGYNISTIC using Google Gemini with multi-key pool failover.

    Args:
        transcript: The Tamil transcript text.
        label: "MISOGYNISTIC" or "NON-MISOGYNISTIC".
        category: "OBJECTIFICATION" | "STEREOTYPING" | "SHAMING" | "VIOLENCE" | "GENERAL_ABUSE" | "NONE".
        evidence: Extracted verbatim evidence phrase, if any.
        model_name: Gemini model name (default: gemini-2.5-flash).

    Returns:
        Dict with keys:
            - available (bool)
            - why_explanation (str)
            - cultural_context (str)
            - evidence_analysis (str)
            - tamil_explanation (str)
            - full_response (str)
            - model_used (str)
    """
    api_key = get_gemini_api_key()

    if not api_key or not api_key.strip():
        return _fallback_explanation(
            transcript=transcript,
            label=label,
            category=category,
            evidence=evidence,
            reason="GEMINI_API_KEY is not configured in .env or environment.",
        )

    clean_text = transcript.strip() if transcript else ""
    if not clean_text:
        return _fallback_explanation(
            transcript="",
            label="NON-MISOGYNISTIC",
            category="NONE",
            evidence="",
            reason="Empty transcript provided.",
        )

    # Carefully constructed sociolinguistic prompt for Tamil gender discourse
    prompt = f"""You are a senior sociolinguist and expert AI researcher specializing in Tamil natural language processing, gender discourse analysis, and content safety.

Analyze the following Tamil statement:
Statement: "{clean_text}"

Current System Classification:
- Classification: {label}
- Category: {category}
- Extracted Evidence: "{evidence if evidence else 'None identified'}"

TASK:
Provide a clear, authoritative, and compassionate analysis explaining WHY this statement is classified as {label} in the context of the category '{category}'.

Follow these guidelines strictly:
1. Explain WHY: Break down the semantic meaning and intent. If Misogynistic, explain how it demeans, restricts, objectifies, shames, or threatens women. If Non-Misogynistic, explain why it is neutral, positive, or supportive of equality, and why gender words (like 'பெண்', 'தாய்', 'மனைவி') are used here in a completely benign context.
2. Cultural & Linguistic Context: Discuss the cultural implications in contemporary Tamil society, media, or everyday discourse.
3. Evidence Breakdown: Highlight the exact words/phrases that justify or refute misogyny.
4. Tamil Explanation: Provide a 2-3 sentence summary in pure, respectful Tamil (தமிழ் விளக்கம்) explaining the finding for Tamil speakers.

Please structure your response with these exact markdown sections:
### 1. Core Reasoning (Why this Classification)
### 2. Cultural & Societal Context
### 3. Linguistic Evidence Analysis
### 4. தமிழ் விளக்கம் (Tamil Summary)
"""

    def _call_gemini_api(key_to_use: str) -> str:
        from google import genai
        client = genai.Client(api_key=key_to_use.strip())
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )
        full_text = getattr(response, "text", None)
        if not full_text:
            raise RuntimeError("Gemini returned an empty response.")
        return full_text

    try:
        if key_manager.has_keys("gemini"):
            full_text = key_manager.execute_with_failover("gemini", _call_gemini_api)
        else:
            full_text = _call_gemini_api(api_key)

        # Parse sections for structured UI presentation
        parsed = _parse_sections(full_text)

        return {
            "available": True,
            "why_explanation": parsed.get("core_reasoning", full_text),
            "cultural_context": parsed.get("cultural_context", ""),
            "evidence_analysis": parsed.get("evidence_analysis", ""),
            "tamil_explanation": parsed.get("tamil_summary", ""),
            "full_response": full_text,
            "model_used": model_name,
            "error": None,
        }

    except Exception as e:
        clean_err = key_manager.sanitize_text(str(e))
        return _fallback_explanation(
            transcript=clean_text,
            label=label,
            category=category,
            evidence=evidence,
            reason=f"Gemini API call failed: {clean_err}",
        )



def _parse_sections(markdown_text: str) -> Dict[str, str]:
    """Extract structured sections from Gemini markdown response."""
    sections = {
        "core_reasoning": "",
        "cultural_context": "",
        "evidence_analysis": "",
        "tamil_summary": "",
    }

    current_sec = None
    lines = markdown_text.splitlines()
    buffer = []

    for line in lines:
        lower = line.lower()
        if "core reasoning" in lower or "1. core" in lower:
            if current_sec and buffer:
                sections[current_sec] = "\n".join(buffer).strip()
            current_sec = "core_reasoning"
            buffer = []
        elif "cultural" in lower or "2. cultural" in lower:
            if current_sec and buffer:
                sections[current_sec] = "\n".join(buffer).strip()
            current_sec = "cultural_context"
            buffer = []
        elif "linguistic evidence" in lower or "3. linguistic" in lower or "evidence analysis" in lower:
            if current_sec and buffer:
                sections[current_sec] = "\n".join(buffer).strip()
            current_sec = "evidence_analysis"
            buffer = []
        elif "தமிழ் விளக்கம்" in line or "tamil summary" in lower or "4. தமிழ்" in line:
            if current_sec and buffer:
                sections[current_sec] = "\n".join(buffer).strip()
            current_sec = "tamil_summary"
            buffer = []
        else:
            buffer.append(line)

    if current_sec and buffer:
        sections[current_sec] = "\n".join(buffer).strip()

    return sections


def _fallback_explanation(
    transcript: str,
    label: str,
    category: str,
    evidence: str,
    reason: str,
) -> Dict[str, Any]:
    """Provide rule-based explanation when Gemini API is unavailable or unconfigured."""
    if label == "MISOGYNISTIC":
        cat_desc = {
            "OBJECTIFICATION": "The statement reduces a woman's value exclusively to bodily appearance, cosmetics, or physical attraction rather than individual dignity or skills.",
            "STEREOTYPING": "The statement enforces restrictive domestic roles, kitchen confinement, or undermines women's autonomy and career potential.",
            "SHAMING": "The statement judges, belittles, or moral-polices women based on clothing, modesty, lifestyle, or personal choices.",
            "VIOLENCE": "The statement justifies, threatens, or promotes physical, psychological, or domestic violence against women.",
            "GENERAL_ABUSE": "The statement asserts that women are mentally or functionally inferior to men or employs derogatory rhetoric.",
        }.get(category, "The statement exhibits gender bias against women.")

        why = f"Classified as **MISOGYNISTIC** under **{category}**. {cat_desc}"
        tam = "இந்த வாக்கியம் பெண்களுக்கு எதிரான பாலின பாகுபாடு அல்லது இழிவான கருத்துக்களை கொண்டுள்ளதாக கண்டறியப்பட்டுள்ளது."
    else:
        why = "Classified as **NON-MISOGYNISTIC**. The statement contains no derogatory rhetoric, harmful stereotypes, shaming, or violence against women. Gender-related words here are used in a safe, neutral, or equality-promoting manner."
        tam = "இந்த வாக்கியம் பெண்களுக்கு எதிரான எந்தவித இழிவான அல்லது பாகுபாடான கருத்துக்களையும் கொண்டிருக்கவில்லை. இது பொதுவான அல்லது சமத்துவத்தை ஆதரிக்கும் உரையாடலாகும்."

    return {
        "available": False,
        "why_explanation": why,
        "cultural_context": "Rule-based baseline contextual analysis (Gemini API unavailable or offline).",
        "evidence_analysis": f"Evaluated phrase: '{evidence}'" if evidence else "No misogynistic cues detected.",
        "tamil_explanation": tam,
        "full_response": why,
        "model_used": "Rule-Based Baseline (Offline)",
        "error": reason,
    }
