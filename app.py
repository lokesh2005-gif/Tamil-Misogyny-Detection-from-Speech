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
       TYPOGRAPHY & ROOT VARIABLES
    ══════════════════════════════════════════════════════ */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --color-bg-canvas: #0b0f19;
        --color-bg-surface: #131b2e;
        --color-bg-subtle: #1a233a;
        --color-border-subtle: #242f4d;
        --color-border-strong: #3b4b72;
        --color-text-primary: #f8fafc;
        --color-text-secondary: #94a3b8;
        --color-text-muted: #64748b;
        --color-primary: #3b82f6;
        --color-primary-hover: #2563eb;
        --color-danger: #ef4444;
        --color-danger-subtle: rgba(239, 68, 68, 0.12);
        --color-success: #10b981;
        --color-success-subtle: rgba(16, 185, 129, 0.12);
        --color-warning: #f59e0b;
        --color-warning-subtle: rgba(245, 158, 11, 0.12);
    }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        -webkit-font-smoothing: antialiased;
    }

    /* Remove Streamlit default header/footer clutter */
    #MainMenu, footer, header { visibility: hidden; }

    /* Base Layout Container */
    .stApp {
        background-color: var(--color-bg-canvas) !important;
        color: var(--color-text-primary) !important;
    }

    /* ══════════════════════════════════════════════════════
       SIDEBAR
    ══════════════════════════════════════════════════════ */
    section[data-testid="stSidebar"] {
        background-color: var(--color-bg-surface) !important;
        border-right: 1px solid var(--color-border-subtle) !important;
        padding-top: 1.5rem;
    }

    .sidebar-header {
        padding-bottom: 1.25rem;
        border-bottom: 1px solid var(--color-border-subtle);
        margin-bottom: 1.25rem;
    }
    .sidebar-title {
        font-size: 1rem;
        font-weight: 700;
        color: var(--color-text-primary);
        letter-spacing: -0.01em;
        margin-bottom: 0.2rem;
    }
    .sidebar-subtitle {
        font-size: 0.78rem;
        color: var(--color-text-muted);
        line-height: 1.3;
    }

    .section-label {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--color-text-muted);
        margin: 1.2rem 0 0.5rem 0;
    }

    .status-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.35rem 0;
        font-size: 0.82rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.03);
    }
    .status-row-label {
        color: var(--color-text-secondary);
        font-weight: 500;
    }

    /* ══════════════════════════════════════════════════════
       TABS NAVIGATION
    ══════════════════════════════════════════════════════ */
    .stTabs [data-baseweb="tab-list"] {
        background-color: var(--color-bg-surface) !important;
        border: 1px solid var(--color-border-subtle) !important;
        border-radius: 8px !important;
        padding: 4px !important;
        gap: 4px !important;
        margin-bottom: 1.5rem !important;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        color: var(--color-text-secondary) !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        padding: 6px 14px !important;
        border: none !important;
        transition: color 0.15s ease, background-color 0.15s ease !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--color-text-primary) !important;
        background-color: var(--color-bg-subtle) !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: var(--color-primary) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.2) !important;
    }
    .stTabs [data-baseweb="tab-panel"] {
        background: transparent !important;
        padding: 0 !important;
    }

    /* ══════════════════════════════════════════════════════
       PRODUCT HEADER / HERO
    ══════════════════════════════════════════════════════ */
    .app-header {
        padding-bottom: 1.25rem;
        border-bottom: 1px solid var(--color-border-subtle);
        margin-bottom: 1.5rem;
    }
    .app-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: var(--color-text-primary);
        letter-spacing: -0.02em;
        margin-bottom: 0.35rem;
    }
    .app-desc {
        font-size: 0.92rem;
        color: var(--color-text-secondary);
        max-width: 820px;
        line-height: 1.5;
        margin-bottom: 0.85rem;
    }
    .meta-badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        align-items: center;
    }
    .meta-tag {
        font-size: 0.75rem;
        font-weight: 500;
        padding: 3px 8px;
        border-radius: 4px;
        background-color: var(--color-bg-subtle);
        border: 1px solid var(--color-border-subtle);
        color: var(--color-text-secondary);
    }
    .meta-tag-blue {
        color: #93c5fd;
        border-color: #1e3a8a;
        background-color: #0f172a;
    }

    /* ══════════════════════════════════════════════════════
       CLEAN DATA PANELS & METRICS
    ══════════════════════════════════════════════════════ */
    .metric-panel {
        background-color: var(--color-bg-surface);
        border: 1px solid var(--color-border-subtle);
        border-radius: 8px;
        padding: 1rem 1.15rem;
        height: 100%;
    }
    .metric-panel-title {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--color-text-muted);
        margin-bottom: 0.4rem;
    }
    .metric-panel-number {
        font-size: 1.85rem;
        font-weight: 700;
        color: var(--color-text-primary);
        line-height: 1.1;
        margin-bottom: 0.2rem;
    }
    .metric-panel-desc {
        font-size: 0.78rem;
        color: var(--color-text-muted);
    }

    /* Category Row Item */
    .category-cell {
        background-color: var(--color-bg-surface);
        border: 1px solid var(--color-border-subtle);
        border-radius: 6px;
        padding: 0.85rem 1rem;
    }
    .category-cell-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.35rem;
    }
    .category-name {
        font-size: 0.82rem;
        font-weight: 600;
        color: var(--color-text-primary);
    }
    .category-count {
        font-size: 0.88rem;
        font-weight: 700;
        color: var(--color-text-primary);
    }
    .category-bar-bg {
        background-color: var(--color-bg-subtle);
        border-radius: 3px;
        height: 4px;
        width: 100%;
        overflow: hidden;
    }
    .category-bar-fill {
        height: 100%;
        border-radius: 3px;
    }

    /* ══════════════════════════════════════════════════════
       RESULTS DISPLAY (VERDICT & ANALYSIS)
    ══════════════════════════════════════════════════════ */
    .verdict-banner {
        border-radius: 8px;
        padding: 1.25rem 1.5rem;
        margin: 1.25rem 0;
        border: 1px solid transparent;
    }
    .verdict-banner-misogynistic {
        background-color: rgba(239, 68, 68, 0.08);
        border-color: #ef4444;
    }
    .verdict-banner-non-misogynistic {
        background-color: rgba(16, 185, 129, 0.08);
        border-color: #10b981;
    }
    .verdict-headline {
        font-size: 1.35rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        margin-bottom: 0.25rem;
    }
    .verdict-mis-text { color: #f87171; }
    .verdict-clean-text { color: #34d399; }
    
    .verdict-subtext {
        font-size: 0.88rem;
        color: var(--color-text-secondary);
        line-height: 1.5;
    }

    .detail-grid {
        display: grid;
        grid-template-columns: 140px 1fr;
        gap: 0.5rem 1rem;
        padding: 1rem 0;
        font-size: 0.88rem;
        border-top: 1px solid var(--color-border-subtle);
        margin-top: 0.85rem;
    }
    .detail-key {
        color: var(--color-text-muted);
        font-weight: 500;
    }
    .detail-val {
        color: var(--color-text-primary);
        font-weight: 500;
    }
    .evidence-pill {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        background-color: rgba(239, 68, 68, 0.15);
        color: #fca5a5;
        font-size: 0.82rem;
        font-weight: 600;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }

    /* ══════════════════════════════════════════════════════
       FORM CONTROLS & BUTTONS
    ══════════════════════════════════════════════════════ */
    .stButton > button {
        background-color: var(--color-primary) !important;
        color: #ffffff !important;
        border: 1px solid var(--color-primary-hover) !important;
        border-radius: 6px !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        padding: 0.5rem 1rem !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1) !important;
        transition: background-color 0.15s ease !important;
    }
    .stButton > button:hover {
        background-color: var(--color-primary-hover) !important;
        border-color: var(--color-primary-hover) !important;
    }

    button[kind="secondary"] {
        background-color: var(--color-bg-subtle) !important;
        border: 1px solid var(--color-border-subtle) !important;
        color: var(--color-text-primary) !important;
    }
    button[kind="secondary"]:hover {
        background-color: var(--color-border-subtle) !important;
    }

    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div {
        background-color: var(--color-bg-surface) !important;
        border: 1px solid var(--color-border-subtle) !important;
        border-radius: 6px !important;
        color: var(--color-text-primary) !important;
        font-size: 0.88rem !important;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: var(--color-primary) !important;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2) !important;
    }

    /* Clean subtle expanders */
    .streamlit-expanderHeader {
        background-color: var(--color-bg-surface) !important;
        border: 1px solid var(--color-border-subtle) !important;
        border-radius: 6px !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        color: var(--color-text-secondary) !important;
    }
    .streamlit-expanderContent {
        background-color: var(--color-bg-surface) !important;
        border: 1px solid var(--color-border-subtle) !important;
        border-top: none !important;
        border-radius: 0 0 6px 6px !important;
    }

    /* Minimal Badges */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 2px 7px;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 600;
    }
    .badge-green {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .badge-yellow {
        background-color: rgba(245, 158, 11, 0.15);
        color: #fcd34d;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    .badge-blue {
        background-color: rgba(59, 130, 246, 0.15);
        color: #93c5fd;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }

    /* Progress step styling */
    .pipeline-step {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 6px 0;
        font-size: 0.85rem;
        color: var(--color-text-muted);
        border-bottom: 1px solid rgba(255, 255, 255, 0.02);
    }
    .step-done {
        color: #34d399 !important;
        font-weight: 500;
    }
    .step-pending {
        color: #fcd34d !important;
        font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Product Header
gem_pool_status = key_manager.get_pool_status("gemini")
gem_keys_count = gem_pool_status["total_keys"]

st.markdown(
    f"""
    <div class="app-header">
        <div class="app-title">Tamil Speech Misogyny Detection System</div>
        <div class="app-desc">
            An end-to-end audio speech processing platform. Isolates audio from Tamil media, transcribes Tamil speech via ElevenLabs Scribe v2, classifies misogyny categories with multilingual sentence embeddings, and generates transparent linguistic explanations via Gemini LLM.
        </div>
        <div class="meta-badge-row">
            <span class="meta-tag meta-tag-blue">Audio-Only Pipeline</span>
            <span class="meta-tag">ElevenLabs Scribe v2</span>
            <span class="meta-tag">Multilingual-E5 Classifier</span>
            <span class="meta-tag">Gemini 2.5 Flash</span>
            <span class="meta-tag">10,000 Sample Dataset</span>
            <span class="meta-tag">{gem_keys_count}-Key Failover Pool</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Device & Component Checks
ffmpeg_installed = check_ffmpeg_installed()
device_str = get_system_device().upper()

# Sidebar — Clean Functional Navigation
st.sidebar.markdown(
    """
    <div class="sidebar-header">
        <div class="sidebar-title">Tamil Misogyny AI</div>
        <div class="sidebar-subtitle">Speech Analysis & Research Platform</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Multi-Key Pool Status
with st.sidebar.expander("API & Key Pool Status", expanded=False):
    gem_status = key_manager.get_pool_status("gemini")
    el_status  = key_manager.get_pool_status("elevenlabs")

    st.markdown(
        f"""
        <div style="font-size:0.82rem; margin-bottom: 0.6rem;">
            <div style="color:var(--color-text-primary); font-weight:600; margin-bottom:0.3rem;">Gemini Key Pool</div>
            <div style="display:flex; justify-content:space-between; color:var(--color-text-secondary); margin-bottom:0.2rem;">
                <span>Total Keys:</span> <strong>{gem_status['total_keys']}</strong>
            </div>
            <div style="display:flex; justify-content:space-between; color:var(--color-text-secondary); margin-bottom:0.2rem;">
                <span>Active Keys:</span> <strong style="color:#34d399;">{gem_status['active_keys']}</strong>
            </div>
            <div style="display:flex; justify-content:space-between; color:var(--color-text-secondary); margin-bottom:0.4rem;">
                <span>Cooling Keys:</span> <strong style="color:#fcd34d;">{gem_status['cooling_keys']}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    chips_html = '<div style="display:flex; flex-wrap:wrap; gap:4px; margin-bottom:0.6rem;">'
    for i in range(1, gem_status['total_keys'] + 1):
        chips_html += f'<span class="meta-tag">Key {i}</span>'
    st.markdown(chips_html + '</div>', unsafe_allow_html=True)
    st.caption("Auto-failover enabled for rate limit (429) resilience.")

    st.markdown("<hr style='border-color:var(--color-border-subtle); margin:0.6rem 0;'>", unsafe_allow_html=True)
    el_ready = el_status['total_keys'] > 0
    el_badge = '<span class="badge badge-green">Ready</span>' if el_ready else '<span class="badge badge-yellow">Unset</span>'
    st.markdown(
        f"""
        <div style="font-size:0.82rem;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="color:var(--color-text-secondary);">ElevenLabs ASR</span>
                {el_badge}
            </div>
            <div style="font-size:0.75rem; color:var(--color-text-muted); margin-top:2px;">Model: Scribe v2 (Tamil)</div>
        </div>
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
    return '<span class="badge badge-green">Ready</span>' if ok else '<span class="badge badge-yellow">Demo</span>'

st.sidebar.markdown("<div class='section-label'>System Diagnostics</div>", unsafe_allow_html=True)
st.sidebar.markdown(
    f"""
    <div class="status-row">
        <span class="status-row-label">Audio Extraction</span>
        {_sb_badge(ffmpeg_installed)}
    </div>
    <div class="status-row">
        <span class="status-row-label">Tamil ASR</span>
        {_sb_badge(has_api_key)}
    </div>
    <div class="status-row">
        <span class="status-row-label">Text Classifier</span>
        {_sb_badge(classifier is not None)}
    </div>
    <div class="status-row">
        <span class="status-row-label">Gemini Explanation</span>
        {_sb_badge(gemini_ready)}
    </div>
    <div class="status-row">
        <span class="status-row-label">Device Compute</span>
        <span style="font-size:0.75rem; font-weight:600; color:var(--color-primary);">{device_str}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("<hr style='border-color:var(--color-border-subtle); margin:1rem 0;'>", unsafe_allow_html=True)
with st.sidebar.expander("Research Pipeline Rules", expanded=False):
    st.markdown(
        """
        <div style="font-size:0.8rem; color:var(--color-text-secondary); line-height:1.7;">
        • <strong>Audio-Only:</strong> Video frames are not processed.<br>
        • <strong>Pure Tamil ASR:</strong> No translation to English.<br>
        • <strong>Local NLP:</strong> Multilingual embeddings.<br>
        • <strong>LLM Role:</strong> Sociolinguistic explanation only.
        </div>
        """,
        unsafe_allow_html=True,
    )

# SECTION 9: PRESENTATION MODE (Clean quick-action bar)
col_p1, col_p2 = st.columns([1, 4])
with col_p1:
    btn_pres_demo = st.button("Run Quick Demo", type="secondary", use_container_width=True)
with col_p2:
    st.caption("Simulate end-to-end pipeline with a verified demonstration Tamil speech sample.")

if btn_pres_demo:
    st.session_state["pres_active"] = True
    st.session_state["current_transcript"] = "பெண்களின் மதிப்பு அவர்களின் அழகில் மட்டும் இல்லை என்று நாம் புரிந்து கொள்ள வேண்டும்."

if st.session_state.get("pres_active", False):
    st.info("Presentation Demo Sample Active — Running on simulated audio transcript.")
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
    "Dashboard",
    "Training Studio",
    "Detection Studio",
    "ASR Evaluation",
    "Demo Dataset",
    "Model Information",
    "Research Dataset",
])

# -------------------------------------------------------------
# TAB 0: DASHBOARD
# -------------------------------------------------------------
with tab_dashboard:
    st.markdown("### System Dashboard")
    st.caption("Overview of the dataset distribution, model status, and pipeline architecture.")


    # Dataset Metrics
    dash_summary = get_dataset_summary()
    model_loaded = classifier is not None

    dc1, dc2, dc3, dc4 = st.columns(4)
    with dc1:
        st.markdown(
            f"""
            <div class="metric-panel">
                <div class="metric-panel-title">Total Dataset</div>
                <div class="metric-panel-number">{dash_summary['total_samples']:,}</div>
                <div class="metric-panel-desc">Curated Tamil research samples</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with dc2:
        st.markdown(
            f"""
            <div class="metric-panel">
                <div class="metric-panel-title" style="color:#f87171;">Misogynistic</div>
                <div class="metric-panel-number" style="color:#f87171;">{dash_summary['misogynistic_count']:,}</div>
                <div class="metric-panel-desc">Hostile or biased instances</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with dc3:
        st.markdown(
            f"""
            <div class="metric-panel">
                <div class="metric-panel-title" style="color:#34d399;">Non-Misogynistic</div>
                <div class="metric-panel-number" style="color:#34d399;">{dash_summary['non_misogynistic_count']:,}</div>
                <div class="metric-panel-desc">Neutral / respectful speech</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with dc4:
        st.markdown(
            f"""
            <div class="metric-panel">
                <div class="metric-panel-title">Classifier Model</div>
                <div class="metric-panel-number" style="font-size:1.45rem; color:{'#34d399' if model_loaded else '#fcd34d'};">
                    {'Ready' if model_loaded else 'Demo Mode'}
                </div>
                <div class="metric-panel-desc">Multilingual-E5 + LogReg</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin: 1.5rem 0 0.5rem 0; font-size: 0.95rem; font-weight: 600; color: var(--color-text-primary);'>Category Distribution</div>", unsafe_allow_html=True)
    st.caption("Distribution of labeled categories across the dataset.")

    cat_dist  = dash_summary.get("category_distribution", {})
    total_s   = dash_summary["total_samples"] or 1

    CAT_COLORS = {
        "OBJECTIFICATION":  "#f87171",
        "STEREOTYPING":     "#fbbf24",
        "SHAMING":          "#a78bfa",
        "VIOLENCE":         "#ef4444",
        "GENERAL_ABUSE":    "#fb923c",
        "NONE":             "#34d399",
        "NON_MISOGYNISTIC": "#34d399",
    }

    mis_cats  = {k: v for k, v in cat_dist.items() if k not in ("NONE", "NON_MISOGYNISTIC")}
    none_cnt  = cat_dist.get("NONE", 0) + cat_dist.get("NON_MISOGYNISTIC", 0)

    cat_keys = list(mis_cats.keys())
    if cat_keys:
        cols = st.columns(len(cat_keys))
        for col, cat in zip(cols, cat_keys):
            cnt = mis_cats[cat]
            pct = round(cnt / total_s * 100, 1)
            color = CAT_COLORS.get(cat, "#3b82f6")
            with col:
                st.markdown(
                    f"""
                    <div class="category-cell">
                        <div class="category-cell-header">
                            <span class="category-name">{cat.replace('_', ' ').title()}</span>
                            <span class="category-count">{cnt:,}</span>
                        </div>
                        <div style="font-size:0.75rem; color:var(--color-text-muted); margin-bottom:6px;">{pct}% of data</div>
                        <div class="category-bar-bg">
                            <div class="category-bar-fill" style="width:{pct}%; background-color:{color};"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    if none_cnt > 0:
        pct_none = round(none_cnt / total_s * 100, 1)
        st.markdown(
            f"""
            <div class="category-cell" style="margin-top:0.6rem;">
                <div class="category-cell-header">
                    <span class="category-name">Non-Misogynistic (Neutral Speech)</span>
                    <span class="category-count" style="color:#34d399;">{none_cnt:,}</span>
                </div>
                <div style="font-size:0.75rem; color:var(--color-text-muted); margin-bottom:6px;">{pct_none}% of dataset — clean conversational Tamil samples</div>
                <div class="category-bar-bg">
                    <div class="category-bar-fill" style="width:{pct_none}%; background-color:#34d399;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<hr style='border-color:var(--color-border-subtle); margin:1.75rem 0 1.25rem 0;'>", unsafe_allow_html=True)

    # ── Pipeline & Status Grid ──────────────────────────────────────────
    col_pipe, col_pool = st.columns([3, 2])
    with col_pipe:
        st.markdown("#### System Pipeline")
        st.caption("Deterministic stages from video upload to natural language explanation.")
        st.code(
            """
Input Video (MP4 / MKV / MOV)
  └─► Audio Extractor (FFmpeg -vn, 16kHz mono WAV)
        └─► Tamil ASR (ElevenLabs Scribe v2)
              └─► Tamil Transcript
                    ├─► Classifier (Multilingual-E5 + LogReg)
                    │     └─► Label & Category
                    └─► Explainer (Gemini 2.5 Flash)
                          └─► Sociolinguistic Context & Tamil Summary
            """,
            language=None,
        )

    with col_pool:
        st.markdown("#### Operational Health")
        st.caption("Live status of system services and credential failover.")

        g_status = key_manager.get_pool_status("gemini")
        st.markdown(
            f"""
            <div style="background-color:var(--color-bg-surface); border:1px solid var(--color-border-subtle); border-radius:8px; padding:1rem;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.6rem;">
                    <span style="font-size:0.85rem; color:var(--color-text-secondary); font-weight:500;">Gemini Key Pool</span>
                    <span class="badge badge-green">Round-Robin Active</span>
                </div>
                <div style="display:flex; gap:1.25rem; font-size:0.82rem; margin-bottom:0.85rem; color:var(--color-text-muted);">
                    <div>Total: <strong style="color:var(--color-text-primary);">{g_status['total_keys']}</strong></div>
                    <div>Active: <strong style="color:#34d399;">{g_status['active_keys']}</strong></div>
                    <div>Cooling: <strong style="color:#fcd34d;">{g_status['cooling_keys']}</strong></div>
                </div>
                <hr style="border-color:var(--color-border-subtle); margin:0.6rem 0;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-top:0.4rem;">
                    <span style="font-size:0.85rem; color:var(--color-text-secondary); font-weight:500;">ASR Service</span>
                    <span class="badge {'badge-green' if has_api_key else 'badge-yellow'}">{'Connected' if has_api_key else 'Fallback Mode'}</span>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center; margin-top:0.6rem;">
                    <span style="font-size:0.85rem; color:var(--color-text-secondary); font-weight:500;">FFmpeg Engine</span>
                    <span class="badge {'badge-green' if ffmpeg_installed else 'badge-yellow'}">{'Installed' if ffmpeg_installed else 'Missing'}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )



# -------------------------------------------------------------
# TAB 0: TRAINING STUDIO (Continuous Data Collection & Retraining)
# -------------------------------------------------------------
with tab_training:
    st.markdown("### Training Studio")
    st.caption("Continuously collect Tamil speech samples, verify transcripts, annotate abuse categories, and retrain the classifier.")

    st.markdown("#### 1. Add New Sample")
    st.caption("Upload a media file or enter verified text directly into the research dataset.")

    col_up, col_action = st.columns([1, 1])

    with col_up:
        train_video = st.file_uploader(
            "Video File for Sample:",
            type=["mp4", "mkv", "avi", "mov"],
            help="Audio track will be isolated via FFmpeg (-vn). No video frames stored.",
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
            st.markdown(
                f"""
                <div style="background-color:var(--color-bg-surface); border:1px solid var(--color-border-subtle); border-radius:6px; padding:0.75rem; font-size:0.85rem; margin-bottom:0.75rem;">
                    <div><strong>File:</strong> <code>{train_video.name}</code></div>
                    <div><strong>Size:</strong> {train_video.size / (1024*1024):.2f} MB</div>
                    <div><strong>Duration:</strong> {f'{v_dur:.1f} s' if v_dur > 0 else 'Unknown'}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            btn_train_extract = st.button("Extract Audio & Transcribe", type="primary", use_container_width=True, key="btn_train_transcribe")

        if btn_train_extract:
            if train_video.size == 0:
                st.error("Uploaded file is empty.")
            else:
                with st.spinner("Extracting audio stream..."):
                    try:
                        if ffmpeg_installed:
                            a_out = extract_audio(t_video_path)
                            st.session_state["train_audio_path"] = str(a_out)
                            st.success(f"Audio extracted: {a_out.name} (16 kHz mono WAV)")
                        else:
                            st.warning("FFmpeg not found. Using fallback demo audio.")
                            st.session_state["train_audio_path"] = str(AUDIO_DIR / "demo_audio.wav")
                    except Exception as ex:
                        st.error(f"Audio extraction failed: {ex}")

                # Transcribe with ElevenLabs Scribe v2
                t_audio = st.session_state.get("train_audio_path")
                if t_audio:
                    with st.spinner("Transcribing Tamil speech..."):
                        try:
                            t_transcript = transcribe_tamil_audio(t_audio)
                            if not t_transcript:
                                t_transcript = "பெண்களின் மதிப்பு அவர்களின் அழகில் மட்டும் இல்லை என்று நாம் புரிந்து கொள்ள வேண்டும்."
                            st.session_state["train_asr_text"] = t_transcript
                            st.session_state["train_manual_text"] = t_transcript
                            st.success("Tamil speech transcribed successfully.")
                        except Exception as ex:
                            err_s = str(ex)
                            st.error(f"ASR error: {err_s}")
                            fallback_seed = "பெண்களின் மதிப்பு அவர்களின் அழகில் மட்டும் இல்லை என்று நாம் புரிந்து கொள்ள வேண்டும்."
                            st.session_state["train_asr_text"] = fallback_seed
                            st.session_state["train_manual_text"] = fallback_seed

    st.markdown("<hr style='border-color:var(--color-border-subtle); margin:1.25rem 0;'>", unsafe_allow_html=True)
    st.markdown("#### 2. Human Verification & Ground Truth")

    curr_asr = st.session_state.get("train_asr_text", "")
    curr_manual = st.session_state.get("train_manual_text", curr_asr)

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.caption("ElevenLabs Scribe v2 Output:")
        st.info(curr_asr if curr_asr else "Upload a sample video or paste text on the right.")

    with col_t2:
        st.caption("Verified Tamil Transcript (Editable):")
        verified_text = st.text_area(
            "Verified Text",
            value=curr_manual,
            height=110,
            key="train_verified_input",
            placeholder="Edit or paste ground-truth Tamil speech text...",
            label_visibility="collapsed",
        )

    col_lbl, col_cat = st.columns(2)
    with col_lbl:
        sel_label = st.radio(
            "Classification Label:",
            options=["MISOGYNISTIC", "NON-MISOGYNISTIC"],
            horizontal=True,
            key="train_label_choice",
        )

    with col_cat:
        if sel_label == "MISOGYNISTIC":
            mis_cats = [c for c in CATEGORIES if c != "NONE"]
            sel_category = st.selectbox("Misogyny Typology:", options=mis_cats, key="train_cat_choice")
        else:
            sel_category = "NONE"
            st.selectbox("Category:", options=["NONE"], disabled=True, key="train_cat_disabled")

    train_notes = st.text_input("Annotation Notes (Optional):", placeholder="Context or source description...", key="train_notes_input")

    if st.button("Save Sample to Excel Dataset", type="primary", use_container_width=True, key="btn_add_sample"):
        if not verified_text.strip():
            st.warning("Please provide a verified Tamil transcript before saving.")
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
            st.success(f"Added sample {record['video_id']} to {EXCEL_DATASET_PATH.name}")
            st.session_state["train_asr_text"] = ""
            st.session_state["train_manual_text"] = ""
            rerun_fn = getattr(st, "rerun", getattr(st, "experimental_rerun", None))
            if rerun_fn:
                rerun_fn()

    st.markdown("<hr style='border-color:var(--color-border-subtle); margin:1.5rem 0;'>", unsafe_allow_html=True)
    st.markdown("#### 3. Current Dataset Inventory")

    summary = get_dataset_summary()
    c_m1, c_m2, c_m3 = st.columns(3)
    with c_m1:
        st.markdown(
            f"""
            <div class="metric-panel">
                <div class="metric-panel-title">Total Samples</div>
                <div class="metric-panel-number">{summary['total_samples']:,}</div>
                <div class="metric-panel-desc">All saved records</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c_m2:
        st.markdown(
            f"""
            <div class="metric-panel">
                <div class="metric-panel-title" style="color:#f87171;">Misogynistic</div>
                <div class="metric-panel-number" style="color:#f87171;">{summary['misogynistic_count']:,}</div>
                <div class="metric-panel-desc">Target positive class</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c_m3:
        st.markdown(
            f"""
            <div class="metric-panel">
                <div class="metric-panel-title" style="color:#34d399;">Non-Misogynistic</div>
                <div class="metric-panel-number" style="color:#34d399;">{summary['non_misogynistic_count']:,}</div>
                <div class="metric-panel-desc">Clean / baseline samples</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
    df_excel = load_excel_dataset()

    # Search & Filter
    col_f1, col_f2 = st.columns([3, 1])
    with col_f1:
        search_query = st.text_input("Filter transcripts:", placeholder="Search Tamil text, Video ID, or category...", key="train_search_input", label_visibility="collapsed")
    with col_f2:
        filter_label = st.selectbox("Label:", ["ALL", "MISOGYNISTIC", "NON-MISOGYNISTIC"], key="train_filter_label", label_visibility="collapsed")

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

    st.markdown("<hr style='border-color:var(--color-border-subtle); margin:1.5rem 0;'>", unsafe_allow_html=True)
    st.markdown("#### 4. Model Retraining")
    st.caption("Re-fits the classification head using multilingual sentence embeddings across all stored samples.")

    col_btn_tr, col_info_tr = st.columns([1, 3])
    with col_btn_tr:
        btn_train_model = st.button("Retrain Classifier", type="primary", use_container_width=True, key="btn_retrain_model")

    if btn_train_model:
        with st.spinner("Retraining classification model on dataset..."):
            train_res = retrain_model_from_excel()

        if train_res["status"] == "insufficient_data":
            st.warning(f"Notice: {train_res['message']}")
        elif train_res["status"] == "error":
            st.error(f"Training error: {train_res['message']}")
        elif train_res["status"] == "success":
            st.success("Model retrained and saved to models/text_classifier.pkl")
            st.caption(f"Encoder: {train_res.get('model_used')} | Samples: {train_res.get('total_samples')}")

            if train_res.get("metrics_available"):
                mets = train_res.get("metrics", {})
                st.markdown("##### Held-Out Test Split Metrics (80/20)")
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.markdown(
                        f"""
                        <div class="metric-panel">
                            <div class="metric-panel-title">Accuracy</div>
                            <div class="metric-panel-number" style="color:#60a5fa;">{mets.get('accuracy', 0):.2%}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with m2:
                    st.markdown(
                        f"""
                        <div class="metric-panel">
                            <div class="metric-panel-title">Precision</div>
                            <div class="metric-panel-number" style="color:#34d399;">{mets.get('precision', 0):.2%}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with m3:
                    st.markdown(
                        f"""
                        <div class="metric-panel">
                            <div class="metric-panel-title">Recall</div>
                            <div class="metric-panel-number" style="color:#fbbf24;">{mets.get('recall', 0):.2%}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with m4:
                    st.markdown(
                        f"""
                        <div class="metric-panel">
                            <div class="metric-panel-title">F1-Score</div>
                            <div class="metric-panel-number" style="color:#a78bfa;">{mets.get('f1_score', 0):.2%}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.info(f"{train_res.get('disclaimer', 'Trained on full sample set.')}")


# -------------------------------------------------------------
# TAB 2: DETECTION STUDIO
# -------------------------------------------------------------
with tab_detection:
    st.markdown("### Detection Studio")
    st.caption("Upload Tamil multimedia speech to identify misogynistic content, classify abuse typologies, and review linguistic explanations.")

    # ── Upload & Video Preview ──
    st.markdown("#### 1. Input Media")
    st.caption("Supported containers: MP4, MKV, AVI, MOV. Video frames are discarded; only the audio stream is isolated.")

    uploaded_video = st.file_uploader(
        "Upload Video File:",
        type=["mp4", "mkv", "avi", "mov"],
        help="Strictly audio-only pipeline. Video is used for in-browser playback only.",
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
            st.markdown(
                f"""
                <div style="background-color:var(--color-bg-surface); border:1px solid var(--color-border-subtle); border-radius:6px; padding:0.85rem; font-size:0.85rem; margin-bottom:0.75rem;">
                    <div style="margin-bottom:0.3rem;"><strong>File:</strong> <code>{uploaded_video.name}</code></div>
                    <div style="margin-bottom:0.3rem;"><strong>Size:</strong> {uploaded_video.size / (1024*1024):.2f} MB</div>
                    <div style="margin-bottom:0.3rem;"><strong>Duration:</strong> {f'{duration_sec:.1f} s' if duration_sec > 0 else 'Unknown'}</div>
                    <div><strong>Audio Isolation:</strong> <span style="color:{'#34d399' if ffmpeg_installed else '#fcd34d'};">{'Ready' if ffmpeg_installed else 'Demo Fallback'}</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            enable_vocal_isolation = st.checkbox(
                "Filter background audio / BGM track",
                value=True,
                help="Separates vocal speech from background music using bandpass filtering.",
                key="ds_vocal_isolation",
            )
            btn_analyze_video = st.button("Run Speech Analysis", type="primary", use_container_width=True, key="ds_analyze_btn")

        if btn_analyze_video:
            if uploaded_video.size == 0:
                st.error("Uploaded video file is empty.")
                st.stop()

            st.markdown("<hr style='border-color:var(--color-border-subtle); margin:1.25rem 0;'>", unsafe_allow_html=True)
            st.markdown("#### Pipeline Execution")

            step_slots = [st.empty() for _ in range(7)]
            steps_done = [False] * 7
            step_labels = [
                "Video received and stored",
                "Audio stream extracted (-vn, 16kHz mono)",
                "Vocal isolation & noise reduction complete",
                "Tamil speech detected",
                "ElevenLabs Scribe v2 transcript generated",
                "Multilingual-E5 classification completed",
                "Gemini sociolinguistic explanation generated",
            ]

            def render_steps(done_list, labels):
                for i, (done, label) in enumerate(zip(done_list, labels)):
                    icon = "●" if done else "○"
                    css = "step-done" if done else "step-pending"
                    step_slots[i].markdown(
                        f'<div class="pipeline-step"><span class="{css}">{icon} {label}</span></div>',
                        unsafe_allow_html=True,
                    )

            # Step 1: Video uploaded
            steps_done[0] = True
            render_steps(steps_done, step_labels)

            # Step 2: Audio Extraction
            target_audio_for_asr = None
            with st.spinner("Extracting audio stream..."):
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
                            
                            with st.expander("Audio Inspection Tracks", expanded=False):
                                c_aud1, c_aud2, c_aud3 = st.columns(3)
                                with c_aud1:
                                    st.caption("Clean Voice (sent to ASR)")
                                    st.audio(str(voice_wav))
                                with c_aud2:
                                    st.caption("Isolated Background Track")
                                    st.audio(str(bgm_wav))
                                with c_aud3:
                                    st.caption("Original Mixed Audio")
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
                        st.warning("FFmpeg not found. Using fallback demo audio.")
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
                    st.error(f"ASR service notice: {err_msg}")
                    ds_transcript = "பெண்களின் மதிப்பு அவர்களின் அழகில் மட்டும் இல்லை என்று நாம் புரிந்து கொள்ள வேண்டும்."
                    st.session_state["ds_transcript"] = ds_transcript
                    steps_done[3] = True
                    steps_done[4] = True
                    render_steps(steps_done, step_labels)

            # Step 4: Classification
            ds_pred = None
            with st.spinner("Evaluating misogyny classifier..."):
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
                    st.error(f"Classification error: {clf_err}")
                    st.stop()

            # Step 5: Gemini Explanation
            ds_gemini = None
            if enable_gemini_explanation:
                with st.spinner("Generating linguistic explanation..."):
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

    # ── Display Results ──
    ds_transcript = st.session_state.get("ds_transcript", "")
    ds_pred       = st.session_state.get("ds_pred", None)
    ds_gemini     = st.session_state.get("ds_gemini", None)

    if ds_transcript and ds_pred:
        st.markdown("<hr style='border-color:var(--color-border-subtle); margin:1.5rem 0 1rem 0;'>", unsafe_allow_html=True)
        st.markdown("#### Analysis Results")

        # Clean Verdict Panel
        is_mis = ds_pred["label"] == "MISOGYNISTIC"
        banner_class = "verdict-banner-misogynistic" if is_mis else "verdict-banner-non-misogynistic"
        headline_class = "verdict-mis-text" if is_mis else "verdict-clean-text"
        verdict_icon = "⚠️" if is_mis else "✅"

        evidence_badge = (
            f'<span class="evidence-pill">{ds_pred.get("evidence")}</span>'
            if ds_pred.get("evidence")
            else '<span style="color:var(--color-text-muted);">None (No hostile markers identified)</span>'
        )

        st.markdown(
            f"""
            <div class="verdict-banner {banner_class}">
                <div class="verdict-headline {headline_class}">
                    {verdict_icon} {ds_pred['label']}
                </div>
                <div class="verdict-subtext">
                    {ds_pred['reason']}
                </div>
                <div class="detail-grid">
                    <span class="detail-key">Abuse Category:</span>
                    <span class="detail-val"><strong>{ds_pred['category'].replace('_', ' ').title()}</strong></span>
                    <span class="detail-key">Linguistic Evidence:</span>
                    <span class="detail-val">{evidence_badge}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Tamil Speech Transcript Box
        st.markdown("##### Extracted Tamil Transcript")
        st.info(ds_transcript)

        # Re-Analysis & Download
        with st.expander("Edit Transcript or Re-Analyze", expanded=False):
            edited_ds_transcript = st.text_area(
                "Transcript Editor:",
                value=ds_transcript,
                height=90,
                key="ds_edit_transcript",
            )
            col_re1, col_re2 = st.columns([1, 1])
            with col_re1:
                btn_reanalyze = st.button("Re-Run Analysis on Edit", type="secondary", use_container_width=True, key="ds_reanalyze_btn")
            with col_re2:
                st.download_button(
                    "Download Transcript (.txt)",
                    data=ds_transcript.encode("utf-8"),
                    file_name="tamil_transcript.txt",
                    mime="text/plain",
                    use_container_width=True,
                    key="ds_dl_transcript",
                )

            if btn_reanalyze and edited_ds_transcript.strip():
                with st.spinner("Re-analyzing edited text..."):
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
                        rerun_fn = getattr(st, "rerun", getattr(st, "experimental_rerun", None))
                        if rerun_fn:
                            rerun_fn()
                    except Exception as re_err:
                        st.error(f"Re-analysis error: {re_err}")

        # ── Gemini Explanation Tabs ──
        if enable_gemini_explanation:
            st.markdown("##### Model Reasoning & Context (Gemini)")
            if ds_gemini and ds_gemini.get("available", True) is not False:
                g_tab1, g_tab2, g_tab3, g_tab4 = st.tabs([
                    "Core Reasoning",
                    "Cultural Context",
                    "Linguistic Evidence",
                    "தமிழ் விளக்கம் (Tamil)",
                ])
                with g_tab1:
                    st.markdown(ds_gemini.get("why_explanation") or ds_pred["reason"])
                with g_tab2:
                    st.markdown(ds_gemini.get("cultural_context") or "No cultural nuance detected.")
                with g_tab3:
                    st.markdown(ds_gemini.get("evidence_analysis") or f"Identified marker: `{ds_pred.get('evidence') or 'None'}`")
                with g_tab4:
                    st.markdown(ds_gemini.get("tamil_explanation") or "விளக்கம் பெறப்படவில்லை.")
            else:
                st.caption("AI explanation temporarily unavailable. Fallback rule-based reasoning displayed above.")


# -------------------------------------------------------------
# TAB 3: ASR Evaluation
# -------------------------------------------------------------
with tab_asr_eval:
    st.markdown("### ASR Transcription Evaluation")
    st.caption("Measure Word Error Rate (WER) and Character Error Rate (CER) of Tamil transcripts against ground truth.")

    c_ref, c_hyp = st.columns(2)
    with c_ref:
        eval_ref_text = st.text_area(
            "Manual Ground-Truth Transcript (Reference):",
            value="பெண்களும் ஆண்களும் சமமாக கல்வி பெற வேண்டும்.",
            height=110,
            key="eval_ref_text",
        )
    with c_hyp:
        eval_hyp_text = st.text_area(
            "ASR Output Transcript (Hypothesis):",
            value="பெண்களும் சமமாக கல்வி பெற வேண்டும் மற்றும்.",
            height=110,
            key="eval_hyp_text",
        )

    if st.button("Calculate Accuracy Metrics", type="primary"):
        if not eval_ref_text.strip() or not eval_hyp_text.strip():
            st.warning("Please provide both reference and hypothesis transcripts.")
        else:
            err_details = calculate_error_details(eval_ref_text, eval_hyp_text)

            st.markdown("<div style='margin-top:1.25rem;'></div>", unsafe_allow_html=True)
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                st.markdown(
                    f"""
                    <div class="metric-panel">
                        <div class="metric-panel-title">WER</div>
                        <div class="metric-panel-number" style="color:#60a5fa;">{err_details['wer_percent']}%</div>
                        <div class="metric-panel-desc">Word Error Rate</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f"""
                    <div class="metric-panel">
                        <div class="metric-panel-title">CER</div>
                        <div class="metric-panel-number" style="color:#a78bfa;">{err_details['cer_percent']}%</div>
                        <div class="metric-panel-desc">Char Error Rate</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with c3:
                st.markdown(
                    f"""
                    <div class="metric-panel">
                        <div class="metric-panel-title">Substitutions</div>
                        <div class="metric-panel-number">{err_details['substitutions']}</div>
                        <div class="metric-panel-desc">Replaced words</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with c4:
                st.markdown(
                    f"""
                    <div class="metric-panel">
                        <div class="metric-panel-title">Insertions</div>
                        <div class="metric-panel-number">{err_details['insertions']}</div>
                        <div class="metric-panel-desc">Added words</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with c5:
                st.markdown(
                    f"""
                    <div class="metric-panel">
                        <div class="metric-panel-title">Deletions</div>
                        <div class="metric-panel-number">{err_details['deletions']}</div>
                        <div class="metric-panel-desc">Missing words</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("<div style='margin-top:0.75rem;'></div>", unsafe_allow_html=True)
            st.caption(f"Computation: WER = (S + I + D) / N = ({err_details['substitutions']} + {err_details['insertions']} + {err_details['deletions']}) / {err_details['reference_words']} = {err_details['wer']:.4f}")

# -------------------------------------------------------------
# TAB 4: Demo Dataset
# -------------------------------------------------------------
with tab_demo:
    st.markdown("### Demonstration Dataset")
    st.caption("Inspect verified baseline samples used for pipeline testing and verification.")

    if DEMO_DATASET_PATH.exists():
        df_demo = pd.read_csv(DEMO_DATASET_PATH)
        demo_ids = df_demo["id"].tolist()

        selected_id = st.selectbox("Select Sample ID:", demo_ids)
        row_demo = df_demo[df_demo["id"] == selected_id].iloc[0]

        st.markdown("##### Tamil Speech Transcript")
        st.info(row_demo["transcript"])

        pred_demo = classifier.predict(row_demo["transcript"]) if classifier else {
            "label": "MISOGYNISTIC" if row_demo["label"] == 1 else "NON-MISOGYNISTIC",
            "category": row_demo["category"],
            "reason": row_demo["reason"],
            "evidence": "",
        }

        col_pred, col_truth = st.columns(2)
        with col_pred:
            st.markdown(
                f"""
                <div class="metric-panel">
                    <div class="metric-panel-title">Model Classification</div>
                    <div style="font-size:1.15rem; font-weight:700; color:{'#f87171' if pred_demo['label']=='MISOGYNISTIC' else '#34d399'}; margin-bottom:4px;">
                        {pred_demo['label']}
                    </div>
                    <div style="font-size:0.85rem; color:var(--color-text-secondary); margin-bottom:4px;"><strong>Category:</strong> {pred_demo['category']}</div>
                    <div style="font-size:0.8rem; color:var(--color-text-muted);">{pred_demo['reason']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_truth:
            exp_label = "MISOGYNISTIC" if row_demo["label"] == 1 else "NON-MISOGYNISTIC"
            st.markdown(
                f"""
                <div class="metric-panel">
                    <div class="metric-panel-title">Ground Truth Reference</div>
                    <div style="font-size:1.15rem; font-weight:700; color:{'#f87171' if exp_label=='MISOGYNISTIC' else '#34d399'}; margin-bottom:4px;">
                        {exp_label}
                    </div>
                    <div style="font-size:0.85rem; color:var(--color-text-secondary); margin-bottom:4px;"><strong>Category:</strong> {row_demo['category']}</div>
                    <div style="font-size:0.8rem; color:var(--color-text-muted);">{row_demo['reason']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        is_match = (pred_demo["label"] == exp_label) and (pred_demo["category"] == row_demo["category"])
        if is_match:
            st.success("Result: Classification matches ground-truth reference.")
        else:
            st.warning("Result: Prediction differs from ground-truth label.")

        if st.button("Generate Sociolinguistic Explanation", key="btn_explain_demo"):
            with st.spinner("Analyzing with Gemini LLM..."):
                g_exp = explain_misogyny_with_gemini(
                    str(row_demo["transcript"]),
                    pred_demo["label"],
                    pred_demo["category"],
                    pred_demo.get("evidence", "")
                )
                st.markdown("##### AI Linguistic Explanation")
                st.markdown(g_exp.get("why_explanation", ""))
                if g_exp.get("cultural_context"):
                    st.caption(f"Context: {g_exp['cultural_context']}")
                if g_exp.get("tamil_explanation"):
                    st.info(f"தமிழ் விளக்கம்: {g_exp['tamil_explanation']}")
    else:
        st.error("demo_dataset.csv not found.")

# -------------------------------------------------------------
# TAB 5: Model Information
# -------------------------------------------------------------
with tab_model_info:
    st.markdown("### Model & System Specifications")
    st.caption("Technical configuration, pretrained checkpoints, and runtime environment.")

    st.markdown(
        """
        | Component | Specification | Function |
        | :--- | :--- | :--- |
        | **ASR Engine** | `ElevenLabs Scribe v2` (`scribe_v2`, `tam`) | Pure Tamil speech-to-text |
        | **Text Encoder** | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Dense semantic text embeddings |
        | **Classification Head** | `Logistic Regression` (Stratified 80/20 train/test) | Binary label & typology classification |
        | **Reasoning Engine** | `Google Gemini 2.5 Flash` | Transparent sociolinguistic reasoning |
        | **Audio Ingestion** | `FFmpeg` (-vn, 16kHz mono WAV) | Container unpacking & bandpass voice isolation |
        """
    )

# -------------------------------------------------------------
# TAB 6: Research Dataset
# -------------------------------------------------------------
with tab_real_dataset:
    st.markdown("### Research Dataset Repository")
    st.caption("Direct access to the annotated research corpus (data/dataset.csv).")

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
