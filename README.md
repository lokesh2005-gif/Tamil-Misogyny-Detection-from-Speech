# Tamil Misogyny Detection from Speech

A Production-Grade Multimodal Speech Analysis System for Tamil Content with Context-Aware NLP and Gemini AI Explanations.

> [!IMPORTANT]
> **Strictly Audio/Speech Only Architecture**:
> This system operates **exclusively on the audio/speech stream**. Video files serve purely as containers for audio extraction (`-vn`). Video frames, image models, and computer vision are strictly excluded.

---

## 🔬 System Pipeline Architecture

```
Tamil Video (MP4 / MKV / AVI / MOV)
        ↓
FFmpeg Audio Extraction (-vn, 16kHz Mono WAV)
        ↓
ElevenLabs Scribe v2 (Tamil ASR · Model: scribe_v2 · Lang: tam)
        ↓
Pure Tamil Speech Transcript (Zero English translation)
        ↓
Text Misogyny Classifier (multilingual-e5-small + Logistic Regression)
        ↓
┌──────────────────────────────────────────────┐
│  MISOGYNISTIC / NON-MISOGYNISTIC             │
│  Category: SHAMING | STEREOTYPING |          │
│            OBJECTIFICATION | VIOLENCE |      │
│            GENERAL_ABUSE | NONE              │
└──────────────────────────────────────────────┘
        ↓
Google Gemini 2.5 Flash (Explanation · Cultural Context · Evidence)
[Multi-Key Pool with Round-Robin Failover & Zero Downtime]
        ↓
Final Result Card + தமிழ் விளக்கம்
```


---

## 1. Final Project Structure

```
tamil_misogyny_detector/
│
├── app.py                     # Integrated Streamlit web application
├── config.py                  # Project paths, audio configs (16kHz mono), categories
├── requirements.txt           # Python dependency specifications
├── README.md                  # Comprehensive project documentation & guide
├── train_text_model.py        # Script to train classifier on real dataset
├── evaluate_text_model.py     # Script to evaluate classifier performance
│
├── src/
│   ├── __init__.py            # Package root
│   ├── audio_processing.py    # FFmpeg-based 16kHz mono audio extraction (-vn)
│   ├── asr.py                 # Local Tamil Speech-to-Text inference engine
│   ├── classifier.py          # Text misogyny & category classification engine
│   ├── metrics.py             # WER/CER (jiwer) & Classification metrics (scikit-learn)
│   └── explanation.py         # Category rationale & linguistic cue explainer
│
├── models/                    # Storage directory for local model checkpoints
│
├── data/
│   ├── videos/                # Uploaded input media storage
│   ├── audio/                 # Extracted 16kHz mono WAV storage
│   ├── transcripts/           # Saved ASR text transcripts (.txt)
│   ├── demo_dataset.csv       # 10 safe demonstration examples (Demo Only)
│   └── dataset.csv            # Template for actual research dataset
│
├── outputs/                   # Export directory for analysis reports
│
└── tests/
    └── test_pipeline.py       # Comprehensive unit and integration test suite
```

---

## 2. Installation Command

```bash
pip install -r requirements.txt
```

Prerequisite: Ensure FFmpeg is installed and accessible in system PATH.
```bash
ffmpeg -version
```

---

## 3. Run Command

Navigate to the project directory and start the Streamlit dashboard:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 4. Required Dependencies

- `Python` >= 3.10
- `streamlit` >= 1.22.0
- `torch` >= 2.0.0
- `transformers` >= 4.30.0
- `sentence-transformers` >= 2.2.0
- `scikit-learn` >= 1.2.0
- `pandas` >= 2.0.0
- `jiwer` >= 3.0.0
- `soundfile` >= 0.12.0
- `FFmpeg` (system binary)

---

## 5. Models Used

1. **Tamil ASR (Speech-to-Text)**:
   - Primary: `AI4Bharat IndicConformerASR` (`ai4bharat/indicconformer_stt_ta_hybrid_ctc_rnnt_large`)
   - Fallback: `openai/whisper-tiny` (forced to `ta` language, `task="transcribe"`)
2. **Text Semantic Embedding**:
   - `intfloat/multilingual-e5-small` (via `sentence-transformers`)
3. **Text Classification Head**:
   - `LogisticRegression` (scikit-learn) / Prototype Semantic Contextual Matcher

---

## 6. How to Run Demo Mode

1. **Instant Presentation Demo**:
   Click the **"🚀 Presentation Demo"** button on the top banner of the Streamlit application for an immediate simulated walkthrough.
2. **Interactive Demo Samples**:
   In the **"📂 Demo Dataset Explorer"** tab, select any ID from `D001` to `D010` to view the Tamil transcript, run the classifier, and compare prediction with ground truth.
3. **Manual Transcript Override**:
   Under the **"🎙️ Live Speech Pipeline"** tab, open **"🛠️ DEMO / MANUAL TRANSCRIPT MODE"** and paste any Tamil transcript directly to test classification without waiting for audio processing.

---

## 7. How to Add Real Tamil Data

Place your collected, ethically annotated research corpus into `data/dataset.csv`.

**Required CSV Columns:**
```csv
video_id,split,label,category,manual_transcript,asr_transcript,wer,cer
```

- `video_id`: Unique identifier (e.g., `VID_001`).
- `split`: `train`, `val`, or `test`.
- `label`: `1` (MISOGYNISTIC) or `0` (NON-MISOGYNISTIC).
- `category`: `SHAMING`, `STEREOTYPING`, `OBJECTIFICATION`, `VIOLENCE`, `GENERAL_ABUSE`, or `NONE`.
- `manual_transcript`: Ground truth Tamil transcription.
- `asr_transcript`: ASR-generated Tamil transcription.
- `wer`: Word Error Rate calculated on the transcripts.
- `cer`: Character Error Rate calculated on the transcripts.

---

## 8. How to Calculate WER / CER

### In Streamlit:
Navigate to the **"📐 ASR Evaluation"** tab:
1. Enter or upload the ground truth manual transcript (Reference).
2. Enter or upload the ASR-generated transcript (Hypothesis).
3. Click **"📊 Calculate ASR Metrics"**.
4. View **WER (%)**, **CER (%)**, **Substitutions ($S$)**, **Insertions ($I$)**, and **Deletions ($D$)**.

### In Python / CLI:
```python
from src.metrics import calculate_error_details

ref = "பெண்களும் ஆண்களும் சமமாக கல்வி பெற வேண்டும்."
hyp = "பெண்களும் சமமாக கல்வி பெற வேண்டும் மற்றும்."

results = calculate_error_details(ref, hyp)
print(f"WER: {results['wer_percent']}%")
print(f"CER: {results['cer_percent']}%")
print(f"S={results['substitutions']}, I={results['insertions']}, D={results['deletions']}")
```

---

## 9. How to Train the Actual Classifier

When your annotated research dataset is ready in `data/dataset.csv`:
```bash
python train_text_model.py --data data/dataset.csv --output models/text_classifier.pkl
```
This script encodes the transcripts with `multilingual-e5-small`, trains balanced `LogisticRegression` models for binary label and category classification, and saves the pipeline to `models/text_classifier.pkl`.

---

## 10. How to Evaluate the Final Model

To evaluate the trained model on test data:
```bash
python evaluate_text_model.py --data data/dataset.csv
```
This outputs:
- **Accuracy**, **Precision**, **Recall**, **F1-Score**, **Macro-F1**
- **Confusion Matrix**
- **Per-Class Classification Report**

---

## Running Automated Tests

```bash
python -m unittest discover -s tests -v
```
All 10 unit and integration tests run across all phases.
