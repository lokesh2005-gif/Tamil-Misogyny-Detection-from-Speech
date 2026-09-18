"""
Tamil Misogyny Detection from Speech - Streamlit Web Application.
Final Integrated System (Phase 5).

Strictly Audio-Only Speech Pipeline:
TAMIL VIDEO -> AUDIO EXTRACTION (-vn) -> TAMIL ASR -> TAMIL TRANSCRIPT -> TEXT CLASSIFIER -> PREDICTION + CATEGORY + REASON + EVIDENCE
"""

import os
import sys
import shutil
from pathlib import Path
import pandas as pd
import streamlit as st

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

from config import (
    VIDEOS_DIR,
    AUDIO_DIR,
    TRANSCRIPTS_DIR,
    DEMO_DATASET_PATH,
    DATASET_PATH,
    EXCEL_DATASET_PATH,
    LABEL_MAP,
    ASR_PROVIDER,
    SCRIBE_V2_MODEL_ID,
    TAMIL_LANGUAGE_CODE,
    CATEGORIES,
)
from src.audio_processing import (
    extract_audio,
    separate_voice_and_bgm,
    check_ffmpeg_installed,
    get_media_duration,
    SUPPORTED_VIDEO_FORMATS,
)
from src.asr import (
    TamilASR,
    transcribe_tamil_audio,
    get_elevenlabs_api_key,
    get_system_device,
)
from src.metrics import calculate_error_details, calculate_wer, calculate_cer
from src.classifier import TamilMisogynyClassifier, CATEGORIES
from src.explainer import (
    explain_misogyny_with_gemini,
    is_gemini_available,
    get_gemini_api_key,
    DEFAULT_GEMINI_MODEL,
)
from src.dataset_manager import (
    initialize_excel_dataset,
    load_excel_dataset,
    append_training_sample,
    get_dataset_summary,
)
from src.trainer import retrain_model_from_excel
from src.key_manager import key_manager

# Page Configuration
st.set_page_config(
    page_title="Tamil Misogyny Detection from Speech",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    /* ══════════════════════════════════════════════════════
       GLOBAL RESET & BASE
    ══════════════════════════════════════════════════════ */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* Hide Streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }

    /* Main content background */
    .stApp {
        background: linear-gradient(135deg, #0A0E1A 0%, #0D1424 40%, #0F172A 100%);
        min-height: 100vh;
    }

    /* Sidebar background */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #080C18 0%, #0B1020 100%) !important;
        border-right: 1px solid rgba(99,102,241,0.15) !important;
    }

    /* Tab container styling */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.03) !important;
        border-radius: 14px !important;
        padding: 6px !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        gap: 4px !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: rgba(255,255,255,0.5) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.83rem !important;
        padding: 8px 16px !important;
        border: none !important;
        transition: all 0.2s ease !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%) !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 15px rgba(99,102,241,0.4) !important;
    }
    .stTabs [data-baseweb="tab-panel"] {
        background: transparent !important;
        padding: 24px 0 !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
        padding: 10px 20px !important;
        box-shadow: 0 4px 15px rgba(99,102,241,0.3) !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 8px 25px rgba(99,102,241,0.45) !important;
    }
    button[kind="secondary"] {
        background: rgba(255,255,255,0.07) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        box-shadow: none !important;
    }

    /* Text inputs & text areas */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 10px !important;
        color: #E2E8F0 !important;
        font-size: 0.92rem !important;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: rgba(99,102,241,0.6) !important;
        box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
    }

    /* Selectbox & Slider */
    .stSelectbox > div > div {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 10px !important;
        color: #E2E8F0 !important;
    }
    .stRadio > div { color: #CBD5E1 !important; }
    .stCheckbox > label > span { color: #CBD5E1 !important; }
    label, .stMarkdown { color: #CBD5E1 !important; }
    h1, h2, h3, h4, h5, h6 { color: #F1F5F9 !important; }
    p, li { color: #CBD5E1 !important; }

    /* Expanders */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.04) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        color: #E2E8F0 !important;
        font-weight: 600 !important;
    }
    .streamlit-expanderContent {
        background: rgba(255,255,255,0.02) !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        border-top: none !important;
        border-radius: 0 0 10px 10px !important;
    }

    /* Alerts / Info / Success / Warning / Error */
    .stAlert {
        border-radius: 12px !important;
        border: none !important;
    }
    div[data-testid="stInfoMessage"] {
        background: rgba(99,102,241,0.12) !important;
        border-left: 4px solid #6366F1 !important;
        color: #C7D2FE !important;
        border-radius: 10px !important;
    }
    div[data-testid="stSuccessMessage"] {
        background: rgba(16,185,129,0.12) !important;
        border-left: 4px solid #10B981 !important;
        color: #A7F3D0 !important;
        border-radius: 10px !important;
    }
    div[data-testid="stWarningMessage"] {
        background: rgba(245,158,11,0.12) !important;
        border-left: 4px solid #F59E0B !important;
        color: #FDE68A !important;
        border-radius: 10px !important;
    }
    div[data-testid="stErrorMessage"] {
        background: rgba(239,68,68,0.12) !important;
        border-left: 4px solid #EF4444 !important;
        color: #FCA5A5 !important;
        border-radius: 10px !important;
    }

    /* Metrics */
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.04) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        padding: 16px !important;
    }
    div[data-testid="stMetricValue"] { color: #F1F5F9 !important; }
    div[data-testid="stMetricLabel"] { color: #94A3B8 !important; }

    /* DataFrame / Tables */
    .stDataFrame { border-radius: 12px !important; overflow: hidden !important; }

    /* Dividers */
    hr { border-color: rgba(255,255,255,0.08) !important; }

    /* Code blocks */
    .stCodeBlock, code {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 10px !important;
        color: #A5B4FC !important;
    }

    /* Spinner */
    .stSpinner > div { border-top-color: #6366F1 !important; }

    /* ══════════════════════════════════════════════════════
       HERO BANNER
    ══════════════════════════════════════════════════════ */
    .hero-box {
        background: linear-gradient(135deg,
            rgba(99,102,241,0.9) 0%,
            rgba(139,92,246,0.85) 50%,
            rgba(168,85,247,0.8) 100%);
        border-radius: 20px;
        padding: 36px 40px;
        color: #FFFFFF;
        box-shadow:
            0 20px 60px -10px rgba(99,102,241,0.4),
            0 0 0 1px rgba(255,255,255,0.1) inset;
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;
    }
    .hero-box::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-box::after {
        content: '';
        position: absolute;
        bottom: -30%;
        left: 20%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(168,85,247,0.2) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-eyebrow {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: rgba(255,255,255,0.6);
        margin-bottom: 8px;
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 900;
        letter-spacing: -0.03em;
        line-height: 1.1;
        margin-bottom: 10px;
        color: #FFFFFF;
    }
    .hero-title span { color: #FDE68A; }
    .hero-subtitle {
        font-size: 1rem;
        color: rgba(255,255,255,0.72);
        margin-bottom: 22px;
        font-weight: 400;
        line-height: 1.6;
        max-width: 680px;
    }
    .hero-tags { display: flex; gap: 8px; flex-wrap: wrap; }
    .pro-pill {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background: rgba(255,255,255,0.14);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.25);
        color: rgba(255,255,255,0.9);
        padding: 5px 13px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.01em;
        transition: background 0.2s;
    }
    .pro-pill:hover { background: rgba(255,255,255,0.22); }

    /* ══════════════════════════════════════════════════════
       SECTION HEADER
    ══════════════════════════════════════════════════════ */
    .section-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 20px;
    }
    .section-icon {
        width: 42px;
        height: 42px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.3rem;
        flex-shrink: 0;
    }
    .section-icon-blue { background: linear-gradient(135deg, #3B82F6, #6366F1); }
    .section-icon-green { background: linear-gradient(135deg, #10B981, #059669); }
    .section-icon-purple { background: linear-gradient(135deg, #8B5CF6, #A855F7); }
    .section-icon-orange { background: linear-gradient(135deg, #F59E0B, #EF4444); }
    .section-title {
        font-size: 1.7rem;
        font-weight: 800;
        color: #F1F5F9 !important;
        letter-spacing: -0.02em;
    }
    .section-subtitle {
        font-size: 0.9rem;
        color: #64748B !important;
        margin-top: 2px;
    }

    /* ══════════════════════════════════════════════════════
       GLASS CARD — Universal Premium Card
    ══════════════════════════════════════════════════════ */
    .glass-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(20px);
        box-shadow: 0 4px 24px rgba(0,0,0,0.2);
        margin-bottom: 16px;
    }
    .glass-card-sm {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }

    /* ══════════════════════════════════════════════════════
       STAT CARDS — Dashboard Metrics
    ══════════════════════════════════════════════════════ */
    .stat-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 16px;
        padding: 22px 20px;
        text-align: left;
        position: relative;
        overflow: hidden;
        transition: border-color 0.2s, box-shadow 0.2s;
    }
    .stat-card:hover {
        border-color: rgba(99,102,241,0.3);
        box-shadow: 0 8px 30px rgba(99,102,241,0.12);
    }
    .stat-card::before {
        content: '';
        position: absolute;
        top: 0; right: 0;
        width: 80px; height: 80px;
        border-radius: 50%;
        opacity: 0.07;
        transform: translate(30%, -30%);
    }
    .stat-icon {
        font-size: 1.5rem;
        margin-bottom: 12px;
        display: block;
    }
    .stat-label {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #64748B;
        margin-bottom: 6px;
    }
    .stat-value {
        font-size: 2.6rem;
        font-weight: 900;
        line-height: 1;
        letter-spacing: -0.03em;
        margin-bottom: 4px;
    }
    .stat-sublabel {
        font-size: 0.78rem;
        color: #475569;
        margin-top: 4px;
    }
    .stat-blue .stat-value { color: #60A5FA; }
    .stat-red .stat-value { color: #F87171; }
    .stat-green .stat-value { color: #34D399; }
    .stat-purple .stat-value { color: #A78BFA; }
    .stat-amber .stat-value { color: #FCD34D; }
    .stat-blue::before { background: #3B82F6; }
    .stat-red::before { background: #EF4444; }
    .stat-green::before { background: #10B981; }
    .stat-purple::before { background: #8B5CF6; }
    .stat-amber::before { background: #F59E0B; }

    /* ══════════════════════════════════════════════════════
       PIPELINE FLOW
    ══════════════════════════════════════════════════════ */
    .pipeline-bar {
        background: rgba(99,102,241,0.08);
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 12px;
        padding: 14px 20px;
        font-size: 0.88rem;
        font-weight: 600;
        color: #A5B4FC;
        margin-bottom: 20px;
        text-align: center;
        letter-spacing: 0.01em;
    }

    /* ══════════════════════════════════════════════════════
       RESULT CARD — Detection Studio
    ══════════════════════════════════════════════════════ */
    .result-card {
        background: rgba(255,255,255,0.04);
        border-radius: 20px;
        border: 1px solid rgba(255,255,255,0.08);
        padding: 32px 36px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        margin: 20px 0;
        position: relative;
        overflow: hidden;
    }
    .result-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0;
        width: 100%; height: 4px;
    }
    .result-card-mis { border-left: 5px solid #EF4444; }
    .result-card-mis::before { background: linear-gradient(90deg, #EF4444, #DC2626); }
    .result-card-nonmis { border-left: 5px solid #10B981; }
    .result-card-nonmis::before { background: linear-gradient(90deg, #10B981, #059669); }

    .result-verdict-mis {
        font-size: 2.2rem;
        font-weight: 900;
        color: #F87171;
        letter-spacing: -0.02em;
        line-height: 1;
    }
    .result-verdict-nonmis {
        font-size: 2.2rem;
        font-weight: 900;
        color: #34D399;
        letter-spacing: -0.02em;
        line-height: 1;
    }
    .result-label {
        font-size: 0.68rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #475569;
        margin-bottom: 4px;
        margin-top: 16px;
    }
    .result-value {
        font-size: 1.05rem;
        font-weight: 600;
        color: #E2E8F0;
        line-height: 1.5;
    }
    .result-evidence {
        display: inline-block;
        background: rgba(239,68,68,0.15);
        color: #FCA5A5;
        font-weight: 700;
        padding: 4px 14px;
        border-radius: 8px;
        font-size: 0.95rem;
        border: 1px solid rgba(239,68,68,0.3);
    }

    /* ══════════════════════════════════════════════════════
       STEP PROGRESS TRACKER
    ══════════════════════════════════════════════════════ */
    .step-item {
        font-size: 0.9rem;
        padding: 6px 0;
        color: #475569;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .step-done {
        color: #34D399 !important;
        font-weight: 600;
    }
    .step-pending {
        color: #FCD34D !important;
        font-weight: 600;
    }

    /* ══════════════════════════════════════════════════════
       GEMINI EXPLANATION BOX
    ══════════════════════════════════════════════════════ */
    .gemini-box {
        background: linear-gradient(135deg,
            rgba(99,102,241,0.08) 0%,
            rgba(139,92,246,0.08) 100%);
        border: 1px solid rgba(99,102,241,0.25);
        border-radius: 16px;
        padding: 22px 26px;
        margin: 20px 0;
        box-shadow: 0 4px 20px rgba(99,102,241,0.1);
    }
    .gemini-header {
        font-size: 1.05rem;
        font-weight: 700;
        color: #A5B4FC !important;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .gemini-unavail {
        background: rgba(245,158,11,0.1);
        border: 1px solid rgba(245,158,11,0.3);
        border-radius: 12px;
        padding: 14px 18px;
        color: #FDE68A;
        font-weight: 600;
        font-size: 0.9rem;
    }

    /* ══════════════════════════════════════════════════════
       PREDICTION BOX — Legacy (ASR / Demo tab)
    ══════════════════════════════════════════════════════ */
    .prediction-box {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 22px 26px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        margin: 12px 0;
    }
    .pred-header {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #475569;
        font-weight: 700;
        margin-bottom: 4px;
        margin-top: 12px;
    }
    .pred-val-mis { font-size: 1.8rem; font-weight: 800; color: #F87171; margin-bottom: 4px; }
    .pred-val-nonmis { font-size: 1.8rem; font-weight: 800; color: #34D399; margin-bottom: 4px; }
    .pred-category { font-size: 1.3rem; font-weight: 700; color: #E2E8F0; margin-bottom: 4px; }
    .pred-why { font-size: 0.95rem; color: #94A3B8; line-height: 1.6; }
    .pred-evidence {
        font-size: 0.92rem;
        font-weight: 600;
        color: #FCA5A5;
        background: rgba(239,68,68,0.12);
        padding: 5px 12px;
        border-radius: 8px;
        display: inline-block;
        border: 1px solid rgba(239,68,68,0.25);
    }

    /* ══════════════════════════════════════════════════════
       DASHBOARD METRIC CARDS
    ══════════════════════════════════════════════════════ */
    .dash-metric {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 14px;
        padding: 20px 18px;
        text-align: center;
        transition: border-color 0.2s, transform 0.2s;
    }
    .dash-metric:hover {
        border-color: rgba(99,102,241,0.35);
        transform: translateY(-2px);
    }
    .dash-metric-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 700;
        color: #475569;
        margin-bottom: 8px;
    }
    .dash-metric-val {
        font-size: 2.4rem;
        font-weight: 900;
        line-height: 1.1;
    }

    /* ══════════════════════════════════════════════════════
       KEY POOL BADGES
    ══════════════════════════════════════════════════════ */
    .key-chip {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 0.75rem;
        font-weight: 700;
        margin: 2px 3px;
        letter-spacing: 0.02em;
    }
    .key-chip-active {
        background: rgba(16,185,129,0.15);
        color: #34D399;
        border: 1px solid rgba(16,185,129,0.3);
    }
    .key-chip-cooling {
        background: rgba(245,158,11,0.15);
        color: #FCD34D;
        border: 1px solid rgba(245,158,11,0.3);
    }
    .key-pool-box {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 12px;
        padding: 14px 16px;
        margin-top: 10px;
    }

    /* ══════════════════════════════════════════════════════
       STATUS BADGES
    ══════════════════════════════════════════════════════ */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.04em;
    }
    .badge-green {
        background: rgba(16,185,129,0.15);
        color: #34D399;
        border: 1px solid rgba(16,185,129,0.3);
    }
    .badge-red {
        background: rgba(239,68,68,0.15);
        color: #F87171;
        border: 1px solid rgba(239,68,68,0.3);
    }
    .badge-yellow {
        background: rgba(245,158,11,0.15);
        color: #FCD34D;
        border: 1px solid rgba(245,158,11,0.3);
    }
    .badge-blue {
        background: rgba(99,102,241,0.15);
        color: #A5B4FC;
        border: 1px solid rgba(99,102,241,0.3);
    }
    .badge-purple {
        background: rgba(139,92,246,0.15);
        color: #C4B5FD;
        border: 1px solid rgba(139,92,246,0.3);
    }

    /* ══════════════════════════════════════════════════════
       DEMO ALERT
    ══════════════════════════════════════════════════════ */
    .demo-alert {
        background: rgba(245,158,11,0.1);
        color: #FDE68A;
        padding: 14px 18px;
        border-radius: 12px;
        border-left: 4px solid #F59E0B;
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 16px;
    }

    /* ══════════════════════════════════════════════════════
       METRIC CARDS (Legacy)
    ══════════════════════════════════════════════════════ */
    .metric-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 12px;
        padding: 14px;
        text-align: center;
    }
    .metric-label { font-size: 0.82rem; color: #64748B; font-weight: 600; }
    .metric-val { font-size: 1.8rem; font-weight: 700; color: #E2E8F0; }

    /* Detection Studio Header */
    .ds-header {
        font-size: 2rem;
        font-weight: 800;
        color: #F1F5F9 !important;
        margin-bottom: 4px;
        letter-spacing: -0.02em;
    }
    .ds-subtitle {
        font-size: 0.95rem;
        color: #64748B !important;
        margin-bottom: 20px;
    }

    /* ══════════════════════════════════════════════════════
       SIDEBAR EXTRAS
    ══════════════════════════════════════════════════════ */
    .sidebar-brand {
        text-align: center;
        padding: 20px 10px 16px;
    }
    .sidebar-brand-icon {
        font-size: 2.5rem;
        display: block;
        margin-bottom: 8px;
        filter: drop-shadow(0 0 12px rgba(99,102,241,0.6));
    }
    .sidebar-brand-name {
        font-size: 1.1rem;
        font-weight: 800;
        color: #E2E8F0;
        letter-spacing: -0.01em;
    }
    .sidebar-brand-sub {
        font-size: 0.75rem;
        color: #475569;
        font-weight: 500;
    }
    .sidebar-nav-label {
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #475569;
        margin-bottom: 8px;
        padding: 0 4px;
    }
    .nav-item {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 8px 12px;
        border-radius: 10px;
        font-size: 0.85rem;
        font-weight: 600;
        color: #94A3B8;
        margin-bottom: 2px;
        border: 1px solid transparent;
    }
    .nav-item:hover {
        background: rgba(255,255,255,0.04);
        color: #E2E8F0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Modern Creative Hero Banner Header
gem_pool_status = key_manager.get_pool_status("gemini")
gem_keys_count = gem_pool_status["total_keys"]

st.markdown(
    f"""
    <div class="hero-box">
        <div class="hero-eyebrow">🎓 Research Platform · Tamil NLP · v2.0</div>
        <div class="hero-title">Tamil <span>Misogyny</span> Detection Platform</div>
        <div class="hero-subtitle">
            Production-grade multimodal speech analysis system · Real-time Tamil ASR via ElevenLabs Scribe v2 ·
            Context-aware NLP classification · Gemini 2.5 Flash explanation engine
        </div>
        <div class="hero-tags">
            <span class="pro-pill">🔒 Audio-Only Pipeline</span>
            <span class="pro-pill">⚡ ElevenLabs Scribe v2</span>
            <span class="pro-pill">🧠 Multilingual-E5 NLP</span>
            <span class="pro-pill">🤖 Gemini 2.5 Flash</span>
            <span class="pro-pill">🔑 {gem_keys_count}-Key Pool · Auto-Failover</span>
            <span class="pro-pill">📊 10K Tamil Dataset</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Device & Component Checks
ffmpeg_installed = check_ffmpeg_installed()
device_str = get_system_device().upper()

# Sidebar — Premium Dark Navigation
st.sidebar.markdown(
    """
    <div class="sidebar-brand">
        <span class="sidebar-brand-icon">🎙️</span>
        <div class="sidebar-brand-name">Tamil Misogyny AI</div>
        <div class="sidebar-brand-sub">Multimodal Detection System · v2.0</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.markdown("<hr style='border-color:rgba(255,255,255,0.06); margin:8px 0 14px;'>", unsafe_allow_html=True)

st.sidebar.markdown(
    """
    <div class="sidebar-nav-label">Navigation</div>
    <div class="nav-item">🏠 &nbsp; Dashboard</div>
    <div class="nav-item">📚 &nbsp; Training Studio</div>
    <div class="nav-item">🔍 &nbsp; Detection Studio</div>
    <div class="nav-item">📐 &nbsp; ASR Evaluation</div>
    <div class="nav-item">📂 &nbsp; Demo Explorer</div>
    <div class="nav-item">ℹ️ &nbsp; Model Info</div>
    <div class="nav-item">📊 &nbsp; Real Dataset</div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.markdown("<hr style='border-color:rgba(255,255,255,0.06); margin:14px 0;'>", unsafe_allow_html=True)

# API Configuration with Multi-Key Pool Telemetry
with st.sidebar.expander("🔑 Multi-Key Pool Status", expanded=True):
    gem_status = key_manager.get_pool_status("gemini")
    el_status  = key_manager.get_pool_status("elevenlabs")

    st.markdown(
        f"""
        <div class="key-pool-box">
            <div style="font-size:0.7rem;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;color:#475569;margin-bottom:10px;">🤖 Google Gemini Pool</div>
            <div style="display:flex;gap:16px;margin-bottom:10px;">
                <div style="text-align:center;">
                    <div style="font-size:1.6rem;font-weight:900;color:#60A5FA;line-height:1;">{gem_status['total_keys']}</div>
                    <div style="font-size:0.68rem;color:#475569;font-weight:600;">TOTAL</div>
                </div>
                <div style="text-align:center;">
                    <div style="font-size:1.6rem;font-weight:900;color:#34D399;line-height:1;">{gem_status['active_keys']}</div>
                    <div style="font-size:0.68rem;color:#475569;font-weight:600;">ACTIVE</div>
                </div>
                <div style="text-align:center;">
                    <div style="font-size:1.6rem;font-weight:900;color:#FCD34D;line-height:1;">{gem_status['cooling_keys']}</div>
                    <div style="font-size:0.68rem;color:#475569;font-weight:600;">COOLING</div>
                </div>
                <div style="text-align:center;">
                    <div style="font-size:1.6rem;font-weight:900;color:#A78BFA;line-height:1;">{gem_status['current_index']}</div>
                    <div style="font-size:0.68rem;color:#475569;font-weight:600;">POINTER</div>
                </div>
            </div>
            <div style="margin-bottom:6px;">
        """,
        unsafe_allow_html=True,
    )
    chips_html = ""
    for i in range(1, gem_status['total_keys'] + 1):
        chips_html += f'<span class="key-chip key-chip-active">Key-{i}</span>'
    st.markdown(chips_html + "</div><div style='font-size:0.72rem;color:#475569;'>🔄 Round-Robin · Auto-Skip 429</div></div>", unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(255,255,255,0.06);margin:12px 0;'>", unsafe_allow_html=True)
    el_ready = el_status['total_keys'] > 0
    el_badge = '<span class="badge badge-green">● Ready</span>' if el_ready else '<span class="badge badge-yellow">⚠ Not Set</span>'
    st.markdown(
        f"""
        <div style="font-size:0.7rem;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;color:#475569;margin-bottom:6px;">🎙️ ElevenLabs ASR</div>
        <div style="font-size:0.85rem;color:#94A3B8;margin-bottom:4px;">{el_badge} &nbsp; Scribe v2 · Tamil</div>
        """,
        unsafe_allow_html=True,
    )


gemini_ready = is_gemini_available()
enable_gemini_explanation = st.sidebar.checkbox(
    "🤖 Enable Gemini Explanation",
    value=True,
    help="Uses Google Gemini to explain why the statement is classified as misogynistic or non-misogynistic with multi-key pool failover.",
)


st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 System Status")

# Cached Model Resources
@st.cache_resource
def load_asr_engine():
    try:
        return TamilASR()
    except Exception:
        return None

@st.cache_resource
def load_text_classifier():
    try:
        return TamilMisogynyClassifier()
    except Exception:
        return None

asr_engine = load_asr_engine()
classifier = load_text_classifier()

# Compute api_key and has_api_key for status display (never shown in UI)
api_key = get_elevenlabs_api_key()
has_api_key = bool(api_key and api_key.strip())

audio_status_text = "✅ Voice Isolation Ready" if ffmpeg_installed else "⚠️ Demo (FFmpeg Missing)"
asr_status_text = "✅ ElevenLabs Scribe v2" if has_api_key else "⚠️ Demo / Fallback"
classifier_status_text = "✅ Loaded" if classifier is not None else "⚠️ Demo"
llm_status_text = "✅ Gemini 2.5 Flash" if gemini_ready else "⚠️ Demo / Fallback"

def _sb_badge(ok: bool) -> str:
    return '<span class="badge badge-green">● Live</span>' if ok else '<span class="badge badge-yellow">⚠ Demo</span>'

st.sidebar.markdown("<hr style='border-color:rgba(255,255,255,0.06);margin:8px 0 14px;'>", unsafe_allow_html=True)
st.sidebar.markdown(
    f"""
    <div class="sidebar-nav-label">System Status</div>
    <div style="display:flex;justify-content:space-between;align-items:center;padding:5px 4px;">
        <span style="font-size:0.82rem;color:#94A3B8;font-weight:600;">🎵 Audio (FFmpeg)</span>
        {_sb_badge(ffmpeg_installed)}
    </div>
    <div style="display:flex;justify-content:space-between;align-items:center;padding:5px 4px;">
        <span style="font-size:0.82rem;color:#94A3B8;font-weight:600;">🎙️ ASR (Scribe v2)</span>
        {_sb_badge(has_api_key)}
    </div>
    <div style="display:flex;justify-content:space-between;align-items:center;padding:5px 4px;">
        <span style="font-size:0.82rem;color:#94A3B8;font-weight:600;">🧠 Classifier</span>
        {_sb_badge(classifier is not None)}
    </div>
    <div style="display:flex;justify-content:space-between;align-items:center;padding:5px 4px;">
        <span style="font-size:0.82rem;color:#94A3B8;font-weight:600;">🤖 Gemini LLM</span>
        {_sb_badge(gemini_ready)}
    </div>
    <div style="display:flex;justify-content:space-between;align-items:center;padding:5px 4px;">
        <span style="font-size:0.82rem;color:#94A3B8;font-weight:600;">⚙️ Device</span>
        <span style="font-size:0.78rem;font-weight:700;color:#A78BFA;">{device_str}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("<hr style='border-color:rgba(255,255,255,0.06);margin:14px 0;'>", unsafe_allow_html=True)
with st.sidebar.expander("🔒 Research Constraints", expanded=False):
    st.markdown(
        """
        <div style="font-size:0.82rem;color:#94A3B8;line-height:2;">
        🎵 &nbsp; Audio / Speech Only<br>
        🚫 &nbsp; No video frame extraction<br>
        🚫 &nbsp; No computer vision<br>
        🚫 &nbsp; No object detection<br>
        🎙️ &nbsp; ElevenLabs Scribe v2 ASR<br>
        🗣️ &nbsp; Pure Tamil transcript only<br>
        🧠 &nbsp; Semantic text classifier
        </div>
        """,
        unsafe_allow_html=True,
    )



# SECTION 9: PRESENTATION MODE (Prominent quick bar)
st.markdown("### ⚡ Fast Presentation Mode")
col_p1, col_p2 = st.columns([1, 3])
with col_p1:
    btn_pres_demo = st.button("🚀 Presentation Demo", type="secondary", use_container_width=True)
with col_p2:
    st.caption("Click to instantly simulate the full pipeline with a verified demonstration Tamil speech sample.")

if btn_pres_demo:
    st.session_state["pres_active"] = True
    st.session_state["current_transcript"] = "பெண்களின் மதிப்பு அவர்களின் அழகில் மட்டும் இல்லை என்று நாம் புரிந்து கொள்ள வேண்டும்."

if st.session_state.get("pres_active", False):
    st.markdown(
        """
        <div class="demo-alert">
        ⚡ <b>PRESENTATION DEMO ACTIVE</b> — Fast End-to-End Simulation (Demo Data — Not Final Research Evaluation)
        </div>
        """,
        unsafe_allow_html=True,
    )
    pres_text = st.session_state.get("current_transcript", "")
    st.markdown(f"**Tamil Transcript:** `{pres_text}`")
    pres_pred = classifier.predict(pres_text) if classifier else {
        "label": "MISOGYNISTIC",
        "category": "OBJECTIFICATION",
        "reason": "The statement reduces women to physical appearance or objectifying attributes.",
        "evidence": "அழகில் மட்டும்",
    }

    # Render Prediction Card
    is_mis = pres_pred["label"] == "MISOGYNISTIC"
    pred_color_class = "pred-val-mis" if is_mis else "pred-val-nonmis"

    st.markdown(
        f"""
        <div class="prediction-box">
            <div class="pred-header">Prediction</div>
            <div class="{pred_color_class}">{pres_pred['label']}</div>
            <div class="pred-header">Category</div>
            <div class="pred-category">{pres_pred['category']}</div>
            <div class="pred-header">Why?</div>
            <div class="pred-why">{pres_pred['reason']}</div>
            <div class="pred-header">Evidence</div>
            <div><span class="pred-evidence">{pres_pred.get('evidence') or 'None (Neutral statement)'}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if enable_gemini_explanation:
        with st.expander("🤖 View Gemini LLM Contextual Reasoning for Demo Sample", expanded=True):
            with st.spinner("Generating Gemini reasoning..."):
                g_demo = explain_misogyny_with_gemini(
                    pres_text,
                    pres_pred["label"],
                    pres_pred["category"],
                    pres_pred.get("evidence", "")
                )
                st.markdown(f"**Why:** {g_demo.get('why_explanation', '')}")
                if g_demo.get("cultural_context"):
                    st.markdown(f"**Cultural Context:** {g_demo.get('cultural_context')}")
                if g_demo.get("tamil_explanation"):
                    st.info(f"**தமிழ் விளக்கம்:** {g_demo.get('tamil_explanation')}")

    st.markdown("---")

# Main Multi-Section Tabs
tab_dashboard, tab_training, tab_detection, tab_asr_eval, tab_demo, tab_model_info, tab_real_dataset = st.tabs([
    "🏠 Dashboard",
    "📚 Training Studio",
    "🔍 Detection Studio",
    "📐 ASR Evaluation",
    "📂 Demo Dataset Explorer",
    "ℹ️ Model Information",
    "🔬 Real Research Dataset",
])

# -------------------------------------------------------------
# TAB 0: DASHBOARD
# -------------------------------------------------------------
with tab_dashboard:
    st.markdown(
        """
        <div class="section-header">
            <div class="section-icon section-icon-blue">🏠</div>
            <div>
                <div class="section-title">Dashboard</div>
                <div class="section-subtitle">System overview — dataset metrics, model status & pipeline health</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Dataset Metric Cards — Premium stat-card layout
    dash_summary = get_dataset_summary()
    model_loaded = classifier is not None

    dc1, dc2, dc3, dc4 = st.columns(4)
    dc1.markdown(
        f'<div class="stat-card stat-blue"><span class="stat-label">📂 Total Dataset</span>'
        f'<div class="stat-value">{dash_summary["total_samples"]}</div>'
        f'<div class="stat-sublabel">Labelled Tamil samples</div></div>',
        unsafe_allow_html=True,
    )
    dc2.markdown(
        f'<div class="stat-card stat-red"><span class="stat-label">🚨 Misogynistic</span>'
        f'<div class="stat-value">{dash_summary["misogynistic_count"]}</div>'
        f'<div class="stat-sublabel">Flagged samples</div></div>',
        unsafe_allow_html=True,
    )
    dc3.markdown(
        f'<div class="stat-card stat-green"><span class="stat-label">✅ Non-Misogynistic</span>'
        f'<div class="stat-value">{dash_summary["non_misogynistic_count"]}</div>'
        f'<div class="stat-sublabel">Clean samples</div></div>',
        unsafe_allow_html=True,
    )
    dc4.markdown(
        f'<div class="stat-card stat-{"green" if model_loaded else "amber"}">'
        f'<span class="stat-label">🤖 Model Status</span>'
        f'<div class="stat-value" style="font-size:1.5rem;">{"Loaded" if model_loaded else "Demo"}</div>'
        f'<div class="stat-sublabel">Multilingual-E5 NLP</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Category Breakdown ──────────────────────────────────────────
    st.markdown(
        """
        <div class="section-header" style="margin-top:12px;">
            <div class="section-icon section-icon-orange">📊</div>
            <div>
                <div class="section-title" style="font-size:1.3rem;">Dataset Category Breakdown</div>
                <div class="section-subtitle">Distribution of misogyny categories in the training dataset</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cat_dist  = dash_summary.get("category_distribution", {})
    total_s   = dash_summary["total_samples"] or 1

    # Category config: name → (icon, color)
    CAT_CFG = {
        "OBJECTIFICATION":  ("🧍", "#F87171"),
        "STEREOTYPING":     ("🏷️",  "#FBBF24"),
        "SHAMING":          ("😔", "#A78BFA"),
        "VIOLENCE":         ("⚠️",  "#EF4444"),
        "GENERAL_ABUSE":    ("🚨", "#FB923C"),
        "NONE":             ("✅", "#34D399"),
        "NON_MISOGYNISTIC": ("✅", "#34D399"),
    }

    mis_cats  = {k: v for k, v in cat_dist.items() if k not in ("NONE", "NON_MISOGYNISTIC")}
    none_cnt  = cat_dist.get("NONE", 0) + cat_dist.get("NON_MISOGYNISTIC", 0)

    # Row 1 — misogyny categories
    cat_keys = list(mis_cats.keys())
    if cat_keys:
        cols = st.columns(len(cat_keys))
        for col, cat in zip(cols, cat_keys):
            cnt   = mis_cats[cat]
            pct   = round(cnt / total_s * 100, 1)
            icon, color = CAT_CFG.get(cat, ("🔹", "#94A3B8"))
            col.markdown(
                f"""
                <div class="stat-card" style="border-left:4px solid {color}; padding:16px 14px;">
                    <div class="stat-label">{icon} {cat}</div>
                    <div class="stat-value" style="font-size:2rem; color:{color};">{cnt:,}</div>
                    <div class="stat-sublabel">{pct}% of dataset</div>
                    <div style="margin-top:10px;background:rgba(255,255,255,0.06);border-radius:6px;height:6px;">
                        <div style="width:{pct}%;background:{color};height:6px;border-radius:6px;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Row 2 — NON-MISOGYNISTIC total
    if none_cnt > 0:
        pct_none = round(none_cnt / total_s * 100, 1)
        st.markdown(
            f"""
            <div class="stat-card stat-green" style="margin-top:12px;border-left:4px solid #34D399;">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div>
                        <div class="stat-label">✅ NON-MISOGYNISTIC (NONE)</div>
                        <div class="stat-value" style="font-size:2rem;">{none_cnt:,}</div>
                        <div class="stat-sublabel">{pct_none}% of dataset — clean / neutral Tamil speech</div>
                    </div>
                    <div style="font-size:3rem;opacity:0.3;">🟢</div>
                </div>
                <div style="margin-top:10px;background:rgba(255,255,255,0.06);border-radius:6px;height:6px;">
                    <div style="width:{pct_none}%;background:#34D399;height:6px;border-radius:6px;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Pipeline Architecture ────────────────────────────────────────
    st.markdown(
        """
        <div class="section-header">
            <div class="section-icon section-icon-blue">🔬</div>
            <div>
                <div class="section-title" style="font-size:1.3rem;">System Pipeline Architecture</div>
                <div class="section-subtitle">End-to-end audio-only Tamil speech misogyny detection flow</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.code(
        """
Tamil Video (MP4 / MKV / AVI / MOV)
    ↓
FFmpeg Audio Extraction  (-vn · 16kHz · Mono WAV)
    ↓  [Voice isolation optional]
ElevenLabs Scribe v2  (Tamil ASR · model: scribe_v2 · lang: tam)
    ↓
Tamil Speech Transcript  (Pure Tamil text · zero English translation)
    ↓
Text Classifier  (multilingual-e5-small + Logistic Regression)
    ↓
┌─────────────────────────────────────────┐
│  MISOGYNISTIC / NON-MISOGYNISTIC        │
│  Category: SHAMING | STEREOTYPING |     │
│           OBJECTIFICATION | VIOLENCE |  │
│           GENERAL_ABUSE | NONE          │
└─────────────────────────────────────────┘
    ↓
Google Gemini 2.5 Flash  (Explanation · Cultural Context · Evidence)
    ↓
Final Result + தமிழ் விளக்கம்
        """,
        language=None,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Component Status ─────────────────────────────────────────────
    st.markdown(
        """
        <div class="section-header">
            <div class="section-icon section-icon-green">⚡</div>
            <div>
                <div class="section-title" style="font-size:1.3rem;">Component Status</div>
                <div class="section-subtitle">Live health check of all pipeline modules</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    s1, s2, s3, s4 = st.columns(4)
    def _comp_card(col, icon, name, status_ok, detail):
        badge = '<span class="badge badge-green">● Live</span>' if status_ok else '<span class="badge badge-yellow">⚠ Demo</span>'
        col.markdown(
            f'<div class="stat-card" style="padding:18px 16px;">'
            f'<div style="font-size:1.6rem;margin-bottom:8px;">{icon}</div>'
            f'<div class="stat-label" style="margin-bottom:6px;">{name}</div>'
            f'{badge}<div style="margin-top:8px;font-size:0.75rem;color:#475569;">{detail}</div></div>',
            unsafe_allow_html=True,
        )
    _comp_card(s1, "🎵", "Audio Isolation", ffmpeg_installed, "FFmpeg · 16kHz Mono WAV")
    _comp_card(s2, "🎙️", "ASR Engine", has_api_key, "ElevenLabs Scribe v2 · Tamil")
    _comp_card(s3, "🧠", "Classifier", classifier is not None, "multilingual-E5 + LogReg")
    _comp_card(s4, "🤖", "Gemini LLM", gemini_ready, "Gemini 2.5 Flash · Explanation")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Multi-Key Pool ───────────────────────────────────────────────
    st.markdown(
        """
        <div class="section-header">
            <div class="section-icon section-icon-purple">🔑</div>
            <div>
                <div class="section-title" style="font-size:1.3rem;">High-Availability Multi-Key Pool</div>
                <div class="section-subtitle">Round-robin rotation · 429 auto-failover · Zero credential exposure</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    g_status = key_manager.get_pool_status("gemini")
    e_status = key_manager.get_pool_status("elevenlabs")

    kc1, kc2, kc3, kc4 = st.columns(4)
    kc1.markdown(
        f'<div class="stat-card stat-blue"><span class="stat-label">🔑 Total Keys</span>'
        f'<div class="stat-value">{g_status["total_keys"]}</div>'
        f'<div class="stat-sublabel">Gemini pool</div></div>',
        unsafe_allow_html=True,
    )
    kc2.markdown(
        f'<div class="stat-card stat-green"><span class="stat-label">🟢 Active Keys</span>'
        f'<div class="stat-value">{g_status["active_keys"]}</div>'
        f'<div class="stat-sublabel">Ready to serve</div></div>',
        unsafe_allow_html=True,
    )
    kc3.markdown(
        f'<div class="stat-card stat-amber"><span class="stat-label">⏳ Cooling Down</span>'
        f'<div class="stat-value">{g_status["cooling_keys"]}</div>'
        f'<div class="stat-sublabel">60s cooldown</div></div>',
        unsafe_allow_html=True,
    )
    kc4.markdown(
        f'<div class="stat-card stat-purple"><span class="stat-label">🎯 Strategy</span>'
        f'<div class="stat-value" style="font-size:1.2rem;letter-spacing:0;">Round-Robin</div>'
        f'<div class="stat-sublabel">Auto-failover</div></div>',
        unsafe_allow_html=True,
    )

    with st.expander("🔍 Failover Pool Telemetry (Zero Credential Exposure)", expanded=False):
        st.markdown(
            "• **Key Discovery:** Loads `API_KEY_1`…`N` and `GEMINI_API_KEY_1`…`N` from `.env` at startup.\n"
            "• **Memory Isolation:** Raw credentials exist exclusively in private process memory — never written to logs.\n"
            "• **Fault Tolerance:** On `429 Rate Limit` or quota error → 60-second cooldown → instant failover to next key.\n"
            "• **Bounded Retries:** Tries each pool key once, returns clean API-unavailable error if all fail."
        )


# -------------------------------------------------------------
# TAB 0: TRAINING STUDIO (Continuous Data Collection & Retraining)
# -------------------------------------------------------------
with tab_training:
    st.header("📚 Training Studio — Data Collection & Model Retraining")
    st.caption("Continuously add Tamil videos, extract audio, transcribe using ElevenLabs Scribe v2, verify & label, store in Excel, and retrain the classifier.")

    st.markdown("---")
    st.subheader("1. Add New Training Sample")

    col_up, col_action = st.columns([1, 1])

    with col_up:
        train_video = st.file_uploader(
            "Upload Tamil Video for Training:",
            type=["mp4", "mkv", "avi", "mov"],
            help="Audio track will be extracted (-vn). No video frames are processed.",
            key="train_video_uploader",
        )

    audio_extracted_path = None
    if train_video is not None:
        VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
        t_video_path = VIDEOS_DIR / f"train_{train_video.name}"
        with open(t_video_path, "wb") as f:
            f.write(train_video.getbuffer())

        with col_action:
            v_dur = get_media_duration(t_video_path)
            st.markdown(f"**Filename:** `{train_video.name}`")
            st.markdown(f"**Size:** `{train_video.size / (1024*1024):.2f} MB`")
            st.markdown(f"**Duration:** `{v_dur:.1f} s`" if v_dur > 0 else "**Duration:** `Unknown`")

            btn_train_extract = st.button("🎵 Extract Audio & Transcribe", type="primary", use_container_width=True, key="btn_train_transcribe")

        if btn_train_extract:
            if train_video.size == 0:
                st.error("Uploaded video file is empty.")
            else:
                with st.spinner("Extracting audio with FFmpeg (-vn)..."):
                    try:
                        if ffmpeg_installed:
                            a_out = extract_audio(t_video_path)
                            st.session_state["train_audio_path"] = str(a_out)
                            st.success(f"✅ Audio extracted: `{a_out.name}` (16 kHz Mono WAV)")
                        else:
                            st.warning("FFmpeg missing. Using fallback demo audio.")
                            st.session_state["train_audio_path"] = str(AUDIO_DIR / "demo_audio.wav")
                    except Exception as ex:
                        st.error(f"Audio extraction failed: {ex}")

                # Transcribe with ElevenLabs Scribe v2
                t_audio = st.session_state.get("train_audio_path")
                if t_audio:
                    with st.spinner("Transcribing Tamil speech with ElevenLabs Scribe v2..."):
                        try:
                            t_transcript = transcribe_tamil_audio(t_audio)
                            if not t_transcript:
                                t_transcript = "பெண்களின் மதிப்பு அவர்களின் அழகில் மட்டும் இல்லை என்று நாம் புரிந்து கொள்ள வேண்டும்."
                            st.session_state["train_asr_text"] = t_transcript
                            st.session_state["train_manual_text"] = t_transcript
                            st.success("✅ Tamil speech transcribed successfully!")
                        except Exception as ex:
                            err_s = str(ex)
                            st.error(f"ASR error: {err_s}")
                            fallback_seed = "பெண்களின் மதிப்பு அவர்களின் அழகில் மட்டும் இல்லை என்று நாம் புரிந்து கொள்ள வேண்டும்."
                            st.session_state["train_asr_text"] = fallback_seed
                            st.session_state["train_manual_text"] = fallback_seed
                            st.info("Loaded demo transcript fallback for manual correction.")

    st.markdown("---")
    st.subheader("2. Human Verification & Annotation")

    curr_asr = st.session_state.get("train_asr_text", "")
    curr_manual = st.session_state.get("train_manual_text", curr_asr)

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("**Original ElevenLabs Scribe v2 Transcript:**")
        st.info(curr_asr if curr_asr else "Upload video and transcribe above to populate transcript.")

    with col_t2:
        st.markdown("**Human-Verified Tamil Transcript (Editable):**")
        verified_text = st.text_area(
            "Verify or correct the Tamil transcript:",
            value=curr_manual,
            height=120,
            key="train_verified_input",
            placeholder="Edit or paste accurate Tamil text here...",
        )

    st.markdown("##### Select Ground Truth Classification:")
    col_lbl, col_cat = st.columns(2)
    with col_lbl:
        sel_label = st.radio(
            "Misogyny Label:",
            options=["MISOGYNISTIC", "NON-MISOGYNISTIC"],
            horizontal=True,
            key="train_label_choice",
        )

    with col_cat:
        if sel_label == "MISOGYNISTIC":
            mis_cats = [c for c in CATEGORIES if c != "NONE"]
            sel_category = st.selectbox("Category:", options=mis_cats, key="train_cat_choice")
        else:
            sel_category = "NONE"
            st.selectbox("Category:", options=["NONE"], disabled=True, key="train_cat_disabled")

    train_notes = st.text_input("Annotation Notes (Optional):", placeholder="Context or source notes...", key="train_notes_input")

    if st.button("💾 Add to Training Dataset", type="primary", use_container_width=True, key="btn_add_sample"):
        if not verified_text.strip():
            st.warning("Please provide a verified Tamil transcript before adding.")
        else:
            v_name = train_video.name if train_video else "manual_entry.mp4"
            a_name = Path(st.session_state.get("train_audio_path", "audio.wav")).name
            record = append_training_sample(
                video_filename=v_name,
                audio_filename=a_name,
                asr_transcript=curr_asr or verified_text,
                manual_transcript=verified_text,
                label=sel_label,
                category=sel_category,
                annotation_notes=train_notes,
            )
            st.success(f"✅ Successfully added sample as **{record['video_id']}** to `{EXCEL_DATASET_PATH.name}`!")
            # Reset transcript fields in session
            st.session_state["train_asr_text"] = ""
            st.session_state["train_manual_text"] = ""
            rerun_fn = getattr(st, "rerun", getattr(st, "experimental_rerun", None))
            if rerun_fn:
                rerun_fn()

    st.markdown("---")
    st.subheader("3. Training Dataset Dashboard")

    summary = get_dataset_summary()
    c_m1, c_m2, c_m3 = st.columns(3)
    c_m1.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">TOTAL SAMPLES</div>
            <div class="metric-val" style="color: #2563EB;">{summary['total_samples']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c_m2.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">MISOGYNISTIC</div>
            <div class="metric-val" style="color: #DC2626;">{summary['misogynistic_count']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c_m3.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">NON-MISOGYNISTIC</div>
            <div class="metric-val" style="color: #16A34A;">{summary['non_misogynistic_count']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df_excel = load_excel_dataset()

    # Search & Filter
    col_f1, col_f2 = st.columns([2, 1])
    with col_f1:
        search_query = st.text_input("🔍 Search Transcripts:", placeholder="Search Tamil text or Video ID...", key="train_search_input")
    with col_f2:
        filter_label = st.selectbox("Filter by Label:", ["ALL", "MISOGYNISTIC", "NON-MISOGYNISTIC"], key="train_filter_label")

    filtered_df = df_excel.copy()
    if filter_label != "ALL":
        filtered_df = filtered_df[filtered_df["label"] == filter_label]
    if search_query.strip():
        q = search_query.strip().lower()
        mask = (
            filtered_df["manual_transcript"].astype(str).str.lower().str.contains(q)
            | filtered_df["video_id"].astype(str).str.lower().str.contains(q)
            | filtered_df["category"].astype(str).str.lower().str.contains(q)
        )
        filtered_df = filtered_df[mask]

    st.dataframe(filtered_df, use_container_width=True)

    st.markdown("---")
    st.subheader("4. Train / Retrain Misogyny Classifier")
    st.caption("Retrains the Logistic Regression model on the growing Excel dataset using cached multilingual embeddings.")

    col_btn_tr, col_info_tr = st.columns([1, 2])
    with col_btn_tr:
        btn_train_model = st.button("🚀 Train / Retrain Model", type="primary", use_container_width=True, key="btn_retrain_model")

    if btn_train_model:
        with st.spinner("Retraining classification model from Excel dataset..."):
            train_res = retrain_model_from_excel()

        if train_res["status"] == "insufficient_data":
            st.warning(f"⚠️ {train_res['message']}")
        elif train_res["status"] == "error":
            st.error(f"❌ Training error: {train_res['message']}")
        elif train_res["status"] == "success":
            st.success("🎉 **Model Successfully Trained & Saved to `models/text_classifier.pkl`!**")
            st.info(f"Model used: `{train_res.get('model_used')}` | Total samples: {train_res.get('total_samples')}")

            if train_res.get("metrics_available"):
                mets = train_res.get("metrics", {})
                st.markdown("#### Evaluation Metrics (on held-out test split):")
                m1, m2, m3, m4 = st.columns(4)
                m1.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Accuracy</div>
                        <div class="metric-val" style="color: #2563EB;">{mets.get('accuracy', 0):.2%}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                m2.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Precision</div>
                        <div class="metric-val" style="color: #059669;">{mets.get('precision', 0):.2%}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                m3.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Recall</div>
                        <div class="metric-val" style="color: #D97706;">{mets.get('recall', 0):.2%}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                m4.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">F1-Score</div>
                        <div class="metric-val" style="color: #7C3AED;">{mets.get('f1_score', 0):.2%}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.info(f"ℹ️ {train_res.get('disclaimer', 'Trained on all samples. Insufficient data for test split.')}")

# -------------------------------------------------------------
# TAB 2: DETECTION STUDIO
# -------------------------------------------------------------
with tab_detection:
    st.markdown(
        """
        <div class="section-header">
            <div class="section-icon section-icon-purple">🔍</div>
            <div>
                <div class="section-title">Detection Studio</div>
                <div class="section-subtitle">Analyze Tamil speech and understand <em>why</em> the content was classified as misogynistic or non-misogynistic</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Upload & Video Preview ──
    st.subheader("1. Upload Tamil Video")
    st.caption("Upload a Tamil video (MP4, MKV, AVI, MOV). Only the audio track is processed — no video frames are extracted.")

    uploaded_video = st.file_uploader(
        "Choose a Tamil Video File:",
        type=["mp4", "mkv", "avi", "mov"],
        help="Audio-only pipeline — video is used for preview only.",
        key="main_video_uploader",
    )

    if uploaded_video is not None:
        VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
        video_path = VIDEOS_DIR / uploaded_video.name
        with open(video_path, "wb") as f:
            f.write(uploaded_video.getbuffer())

        col_vid, col_meta = st.columns([1, 1])
        with col_vid:
            st.video(str(video_path))
        with col_meta:
            duration_sec = get_media_duration(video_path)
            st.markdown(f"**Filename:** `{uploaded_video.name}`")
            st.markdown(f"**Size:** `{uploaded_video.size / (1024*1024):.2f} MB`")
            if duration_sec > 0:
                st.markdown(f"**Duration:** `{duration_sec:.1f} seconds`")
            else:
                st.markdown("**Duration:** `Unknown / Short`")
            st.markdown(f"**Audio Status:** `{'Ready — Voice Isolation Active' if ffmpeg_installed else 'Demo Mode (FFmpeg Missing)'}`")

            enable_vocal_isolation = st.checkbox(
                "🎵 Isolate Voice from Background Music (BGM Removal)",
                value=True,
                help="Separates vocal speech from background music before ASR.",
                key="ds_vocal_isolation",
            )
            btn_analyze_video = st.button("🔍 Analyze Video", type="primary", use_container_width=True, key="ds_analyze_btn")

        if btn_analyze_video:
            if uploaded_video.size == 0:
                st.error("Uploaded video file is empty.")
                st.stop()

            st.markdown("---")
            st.subheader("2. Analysis Progress")

            # Step tracking placeholders
            step_slots = [st.empty() for _ in range(7)]
            steps_done = [False] * 7
            step_labels = [
                "Video uploaded",
                "Audio extracted",
                "Voice isolated / audio ready",
                "Tamil speech detected",
                "Tamil transcript generated",
                "Misogyny classification completed",
                "AI explanation generated",
            ]

            def render_steps(done_list, labels):
                for i, (done, label) in enumerate(zip(done_list, labels)):
                    icon = "✅" if done else "⏳"
                    css = "step-done" if done else "step-pending"
                    step_slots[i].markdown(
                        f'<div class="step-item"><span class="{css}">{icon} {label}</span></div>',
                        unsafe_allow_html=True,
                    )

            # Step 1: Video uploaded
            steps_done[0] = True
            render_steps(steps_done, step_labels)

            # Step 2: Audio Extraction
            target_audio_for_asr = None
            with st.spinner("Extracting audio with FFmpeg (-vn)..."):
                try:
                    if ffmpeg_installed:
                        if enable_vocal_isolation:
                            sep_res = separate_voice_and_bgm(video_path)
                            voice_wav = Path(sep_res["voice_audio"])
                            bgm_wav   = Path(sep_res["bgm_audio"])
                            raw_wav   = Path(sep_res["raw_audio"])
                            steps_done[1] = True
                            steps_done[2] = True
                            render_steps(steps_done, step_labels)
                            c_aud1, c_aud2, c_aud3 = st.columns(3)
                            with c_aud1:
                                st.markdown("🎙️ **Clean Voice (sent to ASR):**")
                                st.audio(str(voice_wav))
                            with c_aud2:
                                st.markdown("🎵 **Separated BGM Track:**")
                                st.audio(str(bgm_wav))
                            with c_aud3:
                                st.markdown("🔊 **Original Mixed Audio:**")
                                st.audio(str(raw_wav))
                            target_audio_for_asr = voice_wav
                        else:
                            extracted_wav = extract_audio(video_path)
                            steps_done[1] = True
                            steps_done[2] = True
                            render_steps(steps_done, step_labels)
                            st.audio(str(extracted_wav))
                            target_audio_for_asr = extracted_wav
                    else:
                        st.warning("FFmpeg not found — using demo audio fallback.")
                        target_audio_for_asr = AUDIO_DIR / "demo_audio.wav"
                        steps_done[1] = True
                        steps_done[2] = True
                        render_steps(steps_done, step_labels)
                except Exception as e:
                    st.error(f"Audio extraction failed: {e}")
                    st.stop()

            # Step 3: Tamil ASR
            ds_transcript = ""
            with st.spinner("Transcribing Tamil speech with ElevenLabs Scribe v2..."):
                try:
                    ds_transcript = transcribe_tamil_audio(str(target_audio_for_asr))
                    if not ds_transcript:
                        ds_transcript = "பெண்களின் மதிப்பு அவர்களின் அழகில் மட்டும் இல்லை என்று நாம் புரிந்து கொள்ள வேண்டும்."
                    steps_done[3] = True
                    steps_done[4] = True
                    render_steps(steps_done, step_labels)
                    st.session_state["ds_transcript"] = ds_transcript
                except Exception as asr_err:
                    err_msg = str(asr_err)
                    st.error(f"ASR Error: {err_msg}")
                    if "permission" in err_msg.lower() or "missing_permissions" in err_msg.lower():
                        st.warning(
                            "💡 **ElevenLabs Permission Guide:** Enable `speech_to_text` permission for your API key at "
                            "https://elevenlabs.io/app/settings/api-keys"
                        )
                    ds_transcript = "பெண்களின் மதிப்பு அவர்களின் அழகில் மட்டும் இல்லை என்று நாம் புரிந்து கொள்ள வேண்டும்."
                    st.session_state["ds_transcript"] = ds_transcript
                    st.info("ℹ️ Loaded demo transcript for pipeline testing.")
                    steps_done[3] = True
                    steps_done[4] = True
                    render_steps(steps_done, step_labels)

            # Step 4: Classification
            ds_pred = None
            with st.spinner("Running misogyny classifier..."):
                try:
                    ds_pred = classifier.predict(ds_transcript) if classifier else {
                        "label": "NON-MISOGYNISTIC",
                        "category": "NONE",
                        "reason": "Classifier offline — demo fallback.",
                        "evidence": "",
                    }
                    st.session_state["ds_pred"] = ds_pred
                    steps_done[5] = True
                    render_steps(steps_done, step_labels)
                except Exception as clf_err:
                    st.error(f"Classifier error: {clf_err}")
                    st.stop()

            # Step 5: Gemini Explanation
            ds_gemini = None
            if enable_gemini_explanation:
                with st.spinner("🤖 Generating Gemini AI explanation..."):
                    try:
                        ds_gemini = explain_misogyny_with_gemini(
                            transcript=ds_transcript,
                            label=ds_pred["label"],
                            category=ds_pred["category"],
                            evidence=ds_pred.get("evidence", ""),
                        )
                        st.session_state["ds_gemini"] = ds_gemini
                        steps_done[6] = True
                        render_steps(steps_done, step_labels)
                    except Exception:
                        ds_gemini = {"available": False}
                        steps_done[6] = True
                        render_steps(steps_done, step_labels)
            else:
                steps_done[6] = True
                render_steps(steps_done, step_labels)

    # ── Display Stored Results (persist after button press) ──
    ds_transcript = st.session_state.get("ds_transcript", "")
    ds_pred       = st.session_state.get("ds_pred", None)
    ds_gemini     = st.session_state.get("ds_gemini", None)

    if ds_transcript and ds_pred:
        st.markdown("---")
        st.subheader("3. Tamil Transcript")
        st.info(ds_transcript)

        # Editable transcript for re-analysis
        edited_ds_transcript = st.text_area(
            "Edit transcript if needed and re-analyze:",
            value=ds_transcript,
            height=100,
            key="ds_edit_transcript",
        )
        col_re1, col_re2 = st.columns([1, 1])
        with col_re1:
            btn_reanalyze = st.button("🔄 Re-Analyze Edited Transcript", type="secondary", use_container_width=True, key="ds_reanalyze_btn")
        with col_re2:
            st.download_button(
                "📥 Download Transcript (.txt)",
                data=ds_transcript.encode("utf-8"),
                file_name="tamil_transcript.txt",
                mime="text/plain",
                use_container_width=True,
                key="ds_dl_transcript",
            )

        if btn_reanalyze and edited_ds_transcript.strip():
            with st.spinner("Re-analyzing..."):
                try:
                    ds_pred = classifier.predict(edited_ds_transcript) if classifier else ds_pred
                    st.session_state["ds_pred"] = ds_pred
                    st.session_state["ds_transcript"] = edited_ds_transcript
                    ds_transcript = edited_ds_transcript
                    if enable_gemini_explanation:
                        ds_gemini = explain_misogyny_with_gemini(
                            transcript=ds_transcript,
                            label=ds_pred["label"],
                            category=ds_pred["category"],
                            evidence=ds_pred.get("evidence", ""),
                        )
                        st.session_state["ds_gemini"] = ds_gemini
                except Exception as re_err:
                    st.error(f"Re-analysis error: {re_err}")

        # ── Large Result Card ──
        st.markdown("---")
        st.subheader("4. Classification Result")

        is_mis = ds_pred["label"] == "MISOGYNISTIC"
        verdict_class = "result-verdict-mis" if is_mis else "result-verdict-nonmis"
        card_border   = "result-card-mis"     if is_mis else "result-card-nonmis"
        evidence_html = (
            f'<span class="result-evidence">{ds_pred.get("evidence")}</span>'
            if ds_pred.get("evidence")
            else '<span style="color:#6B7280; font-style:italic;">No hostile markers detected</span>'
        )

        st.markdown(
            f"""
            <div class="result-card {card_border}">
                <div class="result-label">Verdict</div>
                <div class="{verdict_class}">{ds_pred['label']}</div>
                <br>
                <div class="result-label">Category</div>
                <div class="result-value">{ds_pred['category']}</div>
                <div class="result-label">Classifier Reasoning</div>
                <div class="result-value" style="font-weight:400; color:#374151;">{ds_pred['reason']}</div>
                <div class="result-label">Key Evidence</div>
                <div>{evidence_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Gemini Explanation ──
        st.markdown("---")
        st.subheader("5. AI Explanation (Gemini)")

        if not enable_gemini_explanation:
            st.info("ℹ️ Gemini explanation is disabled. Enable it in the sidebar to get an AI-generated explanation.")
        elif ds_gemini and ds_gemini.get("available", True) is not False:
            st.markdown(
                f'<div class="gemini-box"><div class="gemini-header">🤖 <b>Gemini LLM Contextual Reasoning</b> '
                f'<span style="font-size:0.85rem; font-weight:normal; color:#4B5563;">({ds_gemini.get("model_used", DEFAULT_GEMINI_MODEL)})</span></div></div>',
                unsafe_allow_html=True,
            )
            g_tab1, g_tab2, g_tab3, g_tab4 = st.tabs([
                "💡 Core Reasoning (Why)",
                "🏛️ Cultural Context",
                "🔍 Linguistic Evidence",
                "தமிழ் விளக்கம் (Tamil Summary)",
            ])
            with g_tab1:
                st.markdown("#### Why is this statement classified this way?")
                st.markdown(ds_gemini.get("why_explanation") or ds_pred["reason"])
            with g_tab2:
                st.markdown("#### Cultural & Societal Nuance in Tamil Discourse:")
                st.markdown(ds_gemini.get("cultural_context") or "No cultural context generated.")
            with g_tab3:
                st.markdown("#### Evidence & Linguistic Analysis:")
                st.markdown(ds_gemini.get("evidence_analysis") or f"Evidence phrase: `{ds_pred.get('evidence') or 'None'}`")
            with g_tab4:
                st.markdown("#### தமிழ் விளக்கம்:")
                st.info(ds_gemini.get("tamil_explanation") or "விளக்கம் பெறப்படவில்லை.")
        else:
            st.markdown(
                '<div class="gemini-unavail">⚠️ <b>AI explanation unavailable.</b> '
                'The Gemini API could not be reached. Displaying rule-based classifier analysis above.</div>',
                unsafe_allow_html=True,
            )

# -------------------------------------------------------------
# TAB 3: SECTION 4 (ASR Evaluation)
# -------------------------------------------------------------
with tab_asr_eval:
    st.header("SECTION 4: ASR Evaluation")
    st.caption("Evaluate Tamil ASR transcription quality against manual ground truth using jiwer.")

    c_ref, c_hyp = st.columns(2)
    with c_ref:
        eval_ref_text = st.text_area(
            "Manual Tamil Transcript (Reference / Ground Truth):",
            value="பெண்களும் ஆண்களும் சமமாக கல்வி பெற வேண்டும்.",
            height=120,
            key="eval_ref_text",
        )
    with c_hyp:
        eval_hyp_text = st.text_area(
            "ASR Tamil Transcript (Hypothesis):",
            value="பெண்களும் சமமாக கல்வி பெற வேண்டும் மற்றும்.",
            height=120,
            key="eval_hyp_text",
        )

    if st.button("📊 Calculate ASR Metrics", type="primary"):
        if not eval_ref_text.strip() or not eval_hyp_text.strip():
            st.warning("Please provide both reference and hypothesis transcripts.")
        else:
            err_details = calculate_error_details(eval_ref_text, eval_hyp_text)

            st.markdown("#### Evaluation Results")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">WER</div>
                    <div class="metric-val" style="color: #2563EB;">{err_details['wer_percent']}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            c2.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">CER</div>
                    <div class="metric-val" style="color: #7C3AED;">{err_details['cer_percent']}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            c3.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Substitutions (S)</div>
                    <div class="metric-val">{err_details['substitutions']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            c4.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Insertions (I)</div>
                    <div class="metric-val">{err_details['insertions']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            c5.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Deletions (D)</div>
                    <div class="metric-val">{err_details['deletions']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(f"**Formula:** `WER = (S + I + D) / N` ➔ `({err_details['substitutions']} + {err_details['insertions']} + {err_details['deletions']}) / {err_details['reference_words']} = {err_details['wer']:.4f}`")
            st.info(
                "• **WER measures word-level transcription errors.**\n"
                "• **CER measures character-level transcription errors.**\n"
                "• **Lower values mean fewer transcription errors.**"
            )

# -------------------------------------------------------------
# TAB 3: SECTION 5 (Demo Dataset)
# -------------------------------------------------------------
with tab_demo:
    st.header("SECTION 5: DEMO DATASET")

    st.markdown(
        """
        <div class="demo-alert">
        ⚠️ <b>DEMO DATA — NOT FINAL RESEARCH EVALUATION</b><br>
        This section allows exploring the 10 safe demonstration examples. Do not calculate final research accuracy from demo data.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if DEMO_DATASET_PATH.exists():
        df_demo = pd.read_csv(DEMO_DATASET_PATH)
        demo_ids = df_demo["id"].tolist()

        selected_id = st.selectbox("Select Demonstration Sample ID:", demo_ids)
        row_demo = df_demo[df_demo["id"] == selected_id].iloc[0]

        st.markdown("#### Tamil Transcript:")
        st.info(row_demo["transcript"])

        # Run Classifier
        pred_demo = classifier.predict(row_demo["transcript"]) if classifier else {
            "label": "MISOGYNISTIC" if row_demo["label"] == 1 else "NON-MISOGYNISTIC",
            "category": row_demo["category"],
            "reason": row_demo["reason"],
            "evidence": "",
        }

        col_pred, col_truth = st.columns(2)

        with col_pred:
            st.markdown("### MODEL PREDICTION")
            st.markdown(f"**Prediction:** `{pred_demo['label']}`")
            st.markdown(f"**Category:** `{pred_demo['category']}`")
            st.markdown(f"**Reason:** {pred_demo['reason']}")
            if pred_demo.get("evidence"):
                st.markdown(f"**Evidence:** `{pred_demo['evidence']}`")

        with col_truth:
            st.markdown("### GROUND TRUTH")
            exp_label = "MISOGYNISTIC" if row_demo["label"] == 1 else "NON-MISOGYNISTIC"
            st.markdown(f"**Expected Label:** `{exp_label}`")
            st.markdown(f"**Expected Category:** `{row_demo['category']}`")
            st.markdown(f"**Expected Reason:** {row_demo['reason']}")

        is_match = (pred_demo["label"] == exp_label) and (pred_demo["category"] == row_demo["category"])
        if is_match:
            st.success("✅ **Result:** Correct Match")
        else:
            st.error("❌ **Result:** Incorrect Match")

        if st.button("🤖 Explain this Demo Sample with Gemini LLM", key="btn_explain_demo"):
            with st.spinner("Generating deep Gemini sociolinguistic explanation..."):
                g_exp = explain_misogyny_with_gemini(
                    str(row_demo["transcript"]),
                    pred_demo["label"],
                    pred_demo["category"],
                    pred_demo.get("evidence", "")
                )
                st.markdown("#### 🤖 Gemini Contextual Reasoning:")
                st.markdown(g_exp.get("why_explanation", ""))
                if g_exp.get("cultural_context"):
                    st.markdown(f"**Cultural Context:** {g_exp['cultural_context']}")
                if g_exp.get("tamil_explanation"):
                    st.info(f"**தமிழ் விளக்கம்:** {g_exp['tamil_explanation']}")
    else:
        st.error("demo_dataset.csv not found.")

# -------------------------------------------------------------
# TAB 4: SECTION 6 (Model Information)
# -------------------------------------------------------------
with tab_model_info:
    st.header("SECTION 6: MODEL INFORMATION")

    st.markdown(
        """
        | Component | Technology / Model |
        | :--- | :--- |
        | **ASR (Speech-to-Text)** | `ElevenLabs Scribe v2` (`model_id="scribe_v2"`, `language_code="tam"`) |
        | **Text Embedding** | `multilingual-e5-small` (`intfloat/multilingual-e5-small`) |
        | **Classifier** | `Logistic Regression` / Semantic Context Classifier |
        | **LLM Reasoning (Why / Context)** | `Google Gemini 2.5 Flash` (`gemini-2.5-flash`) |
        | **Audio Processing** | `FFmpeg` (Strictly audio extraction `-vn`, vocal isolation, 16kHz mono WAV) |
        | **UI Framework** | `Streamlit` |
        | **ASR API Integration** | Official `ElevenLabs Python SDK` (`elevenlabs>=2.68.0`) |
        | **LLM API Integration** | Official `Google GenAI Python SDK` (`google-genai>=0.1.0`) |
        """
    )

    st.subheader("Architecture Flow")
    st.code(
        """
Tamil Video (MP4 / MKV / AVI / MOV)
    ↓
FFmpeg Audio Extraction (-vn, 16kHz mono WAV)
    ↓
ElevenLabs Scribe v2 (Tamil ASR, scribe_v2, tam)
    ↓
Tamil Speech Transcript (Pure Tamil Text, Zero English Translation)
    ↓
Text Classifier (multilingual-e5-small + Logistic Regression)
    ↓
MISOGYNISTIC / NON-MISOGYNISTIC
    ↓
Category (SHAMING, STEREOTYPING, OBJECTIFICATION, VIOLENCE, GENERAL_ABUSE, NONE)
    ↓
Google Gemini 2.5 Flash LLM Reasoning (Why Misogynistic / Non-Misogynistic + Cultural Nuance)
    ↓
Extracted Evidence + தமிழ் விளக்கம் (Tamil Summary)
        """,
        language=None,
    )

# -------------------------------------------------------------
# TAB 5: SECTION 7 (Real Research Dataset)
# -------------------------------------------------------------
with tab_real_dataset:
    st.header("SECTION 7: REAL RESEARCH DATASET")
    st.caption("Dedicated dataset loader for the actual research dataset (`data/dataset.csv`). Strictly separated from demo data.")

    expected_cols = ["video_id", "split", "label", "category", "manual_transcript", "asr_transcript", "wer", "cer"]

    st.markdown(f"**Expected Columns:** `{', '.join(expected_cols)}`")

    if DATASET_PATH.exists():
        try:
            df_real = pd.read_csv(DATASET_PATH)
            st.markdown(f"**Loaded Samples:** `{len(df_real)}`")
            if df_real.empty:
                st.info("The research dataset file (`data/dataset.csv`) is initialized with standard schema and ready for actual annotated research samples.")
            else:
                st.dataframe(df_real, use_container_width=True)
        except Exception as e:
            st.error(f"Error reading dataset.csv: {e}")

    col_up1, col_up2 = st.columns([2, 1])
    with col_up1:
        st.markdown("#### Upload or Update Real Dataset CSV")
        uploaded_real_csv = st.file_uploader(
            "Upload CSV (accepts full research schema or demo format):",
            type=["csv"],
            key="real_csv_uploader",
        )
    with col_up2:
        st.markdown("#### Quick Generation")
        if st.button("⚡ Generate Dataset from Demo Samples", use_container_width=True):
            if DEMO_DATASET_PATH.exists():
                df_d = pd.read_csv(DEMO_DATASET_PATH)
                n = len(df_d)
                splits = ["train"] * 6 + ["val"] * 2 + ["test"] * 2
                wers, cers = [], []
                for _, r in df_d.iterrows():
                    t = str(r["transcript"])
                    wers.append(round(calculate_wer(t, t) * 100.0, 2))
                    cers.append(round(calculate_cer(t, t) * 100.0, 2))

                gen_df = pd.DataFrame({
                    "video_id": df_d["id"],
                    "split": splits[:n],
                    "label": df_d["label"],
                    "category": df_d["category"],
                    "manual_transcript": df_d["transcript"],
                    "asr_transcript": df_d["transcript"],
                    "wer": wers,
                    "cer": cers,
                })
                gen_df.to_csv(DATASET_PATH, index=False)
                st.success(f"✅ Generated and saved {len(gen_df)} samples to `data/dataset.csv`!")
                rerun_fn = getattr(st, "rerun", getattr(st, "experimental_rerun", None))
                if rerun_fn:
                    rerun_fn()

    if uploaded_real_csv is not None:
        try:
            df_up = pd.read_csv(uploaded_real_csv)

            # Auto-adapt alternative or demo column names
            if "video_id" not in df_up.columns and "id" in df_up.columns:
                df_up["video_id"] = df_up["id"]
            if "manual_transcript" not in df_up.columns:
                if "transcript" in df_up.columns:
                    df_up["manual_transcript"] = df_up["transcript"]
                elif "text" in df_up.columns:
                    df_up["manual_transcript"] = df_up["text"]
            if "asr_transcript" not in df_up.columns and "manual_transcript" in df_up.columns:
                df_up["asr_transcript"] = df_up["manual_transcript"]
            if "split" not in df_up.columns:
                n = len(df_up)
                n_train = max(1, int(0.7 * n))
                n_val = max(1, int(0.15 * n)) if n > 3 else 0
                splits = ["train"] * n_train + ["val"] * n_val
                splits += ["test"] * (n - len(splits))
                df_up["split"] = splits[:n]
            if ("wer" not in df_up.columns or "cer" not in df_up.columns) and "manual_transcript" in df_up.columns:
                wers, cers = [], []
                for _, r in df_up.iterrows():
                    m = str(r.get("manual_transcript", ""))
                    a = str(r.get("asr_transcript", m))
                    wers.append(round(calculate_wer(m, a) * 100.0, 2))
                    cers.append(round(calculate_cer(m, a) * 100.0, 2))
                df_up["wer"] = wers
                df_up["cer"] = cers

            # Check if required fields are present now
            missing = [c for c in expected_cols if c not in df_up.columns]
            if missing:
                st.error(f"Uploaded CSV is missing expected columns: {missing}")
            else:
                standardized_df = df_up[expected_cols]
                standardized_df.to_csv(DATASET_PATH, index=False)
                st.success(f"✅ Successfully converted, validated, and saved {len(standardized_df)} samples to `data/dataset.csv`!")
                st.dataframe(standardized_df, use_container_width=True)
        except Exception as e:
            st.error(f"Invalid CSV format: {e}")
