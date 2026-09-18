"""
Comprehensive Unit and Integration Tests for Phases 1, 2, 3, 4, and 5.
"""

import sys
import unittest
import subprocess
from pathlib import Path
import pandas as pd

# Add project root to sys.path
TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    DEMO_DATASET_PATH,
    DATASET_PATH,
    AUDIO_DIR,
    VIDEOS_DIR,
    TRANSCRIPTS_DIR,
)
from src.audio_processing import (
    check_ffmpeg_installed,
    extract_audio,
    get_media_duration,
    separate_voice_and_bgm,
)
from unittest.mock import patch, MagicMock
from src.asr import TamilASR, get_system_device, transcribe_tamil_audio, get_elevenlabs_api_key
from src.classifier import TamilMisogynyClassifier, CATEGORIES
from src.explainer import explain_misogyny_with_gemini, is_gemini_available
from src.metrics import (
    calculate_wer,
    calculate_cer,
    calculate_error_details,
    compute_classification_metrics,
)


class TestTamilMisogynyPipeline(unittest.TestCase):

    def setUp(self):
        self.asr = TamilASR()
        self.classifier = TamilMisogynyClassifier()
        AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
        TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # PHASE 1 & DATASET TESTS
    # -------------------------------------------------------------
    def test_demo_dataset_structure(self):
        """Verify demo dataset contains 10 safe demo examples and correct columns."""
        self.assertTrue(DEMO_DATASET_PATH.exists(), "demo_dataset.csv must exist")
        df = pd.read_csv(DEMO_DATASET_PATH)

        expected_cols = ["id", "transcript", "label", "category", "reason"]
        self.assertListEqual(list(df.columns), expected_cols)
        self.assertEqual(len(df), 10, "demo_dataset.csv must contain exactly 10 demonstration examples")

        for cat in df["category"]:
            self.assertIn(cat, CATEGORIES)

        for label in df["label"]:
            self.assertIn(label, [0, 1])

    def test_real_dataset_schema(self):
        """Verify real dataset template schema."""
        self.assertTrue(DATASET_PATH.exists(), "dataset.csv must exist")
        df = pd.read_csv(DATASET_PATH)
        expected_cols = ["video_id", "split", "label", "category", "manual_transcript", "asr_transcript", "wer", "cer"]
        self.assertListEqual(list(df.columns), expected_cols)

    # -------------------------------------------------------------
    # PHASE 2 TESTS: Audio Extraction (MP4, MKV, AVI, MOV) & ASR
    # -------------------------------------------------------------
    def test_audio_extraction_formats(self):
        """Verify FFmpeg extracts 16kHz mono WAV from MP4, MKV, AVI, MOV."""
        self.assertTrue(check_ffmpeg_installed(), "FFmpeg should be in system PATH")

        for ext in [".mp4", ".mkv", ".avi", ".mov"]:
            test_video = AUDIO_DIR / f"test_sample{ext}"
            test_out = AUDIO_DIR / f"test_extracted_{ext[1:]}.wav"

            create_cmd = [
                "ffmpeg",
                "-y",
                "-f", "lavfi",
                "-i", "sine=frequency=1000:duration=0.5",
                "-f", "lavfi",
                "-i", "color=c=blue:s=64x64:d=0.5",
                "-shortest",
                str(test_video),
            ]
            subprocess.run(create_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            self.assertTrue(test_video.exists())

            extracted_path = extract_audio(test_video, test_out)
            self.assertTrue(extracted_path.exists())
            self.assertGreater(extracted_path.stat().st_size, 0)

            dur = get_media_duration(test_video)
            self.assertGreater(dur, 0.0)

            if test_video.exists():
                test_video.unlink()
            if test_out.exists():
                test_out.unlink()

    def test_separate_voice_and_bgm(self):
        """Verify vocal speech separation from background music."""
        test_in = AUDIO_DIR / "test_mixed_audio.wav"
        # Generate mixed tone (speech 400Hz + bass 80Hz + high 6000Hz)
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "sine=frequency=400:duration=0.5",
            "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            str(test_in)
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

        sep_info = separate_voice_and_bgm(test_in)
        self.assertIn("voice_audio", sep_info)
        self.assertIn("bgm_audio", sep_info)
        self.assertTrue(Path(sep_info["voice_audio"]).exists())
        self.assertTrue(Path(sep_info["bgm_audio"]).exists())

        # Clean up
        if test_in.exists():
            test_in.unlink()
        if Path(sep_info["voice_audio"]).exists():
            Path(sep_info["voice_audio"]).unlink()
        if Path(sep_info["bgm_audio"]).exists():
            Path(sep_info["bgm_audio"]).unlink()
        if Path(sep_info["raw_audio"]).exists():
            Path(sep_info["raw_audio"]).unlink()

    def test_asr_manual_mode_and_save_txt(self):
        """Test TamilASR manual/demo mode and TXT saving."""
        dummy_wav = AUDIO_DIR / "dummy_speech.wav"
        cmd = [
            "ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=0.5",
            "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(dummy_wav)
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

        manual_text = "பெண்களும் ஆண்களும் சமமாக கல்வி பெற வேண்டும்."
        res = self.asr.transcribe(dummy_wav, save_txt=True, manual_override_text=manual_text)

        self.assertEqual(res["transcript"], manual_text)
        self.assertEqual(res["backend"], "manual_input")
        self.assertTrue(Path(res["saved_file"]).exists())

        if dummy_wav.exists():
            dummy_wav.unlink()
        if Path(res["saved_file"]).exists():
            Path(res["saved_file"]).unlink()

    def test_transcribe_tamil_audio_not_found(self):
        """transcribe_tamil_audio must raise FileNotFoundError for missing files."""
        with self.assertRaises(FileNotFoundError):
            transcribe_tamil_audio("non_existent_audio_file.wav")

    def test_transcribe_tamil_audio_empty_file(self):
        """transcribe_tamil_audio must raise ValueError for empty audio files."""
        empty_wav = AUDIO_DIR / "empty_test.wav"
        empty_wav.touch()
        try:
            with self.assertRaises(ValueError):
                transcribe_tamil_audio(str(empty_wav))
        finally:
            if empty_wav.exists():
                empty_wav.unlink()

    @patch("src.asr.get_elevenlabs_api_key")
    def test_transcribe_tamil_audio_missing_api_key(self, mock_get_key):
        """transcribe_tamil_audio must raise RuntimeError when API key is missing."""
        mock_get_key.return_value = None
        dummy_wav = AUDIO_DIR / "dummy_key_test.wav"
        dummy_wav.write_bytes(b"RIFFdummydata")
        try:
            with self.assertRaises(RuntimeError) as ctx:
                transcribe_tamil_audio(str(dummy_wav))
            self.assertIn("ELEVENLABS_API_KEY is not configured", str(ctx.exception))
        finally:
            if dummy_wav.exists():
                dummy_wav.unlink()

    @patch("elevenlabs.client.ElevenLabs")
    def test_transcribe_tamil_audio_mocked_success(self, mock_elevenlabs_cls):
        """Verify transcribe_tamil_audio uses scribe_v2, tam, and returns strictly str."""
        mock_client = MagicMock()
        mock_elevenlabs_cls.return_value = mock_client
        mock_result = MagicMock()
        mock_result.text = "பெண்கள் சம உரிமை பெற வேண்டும்."
        mock_client.speech_to_text.convert.return_value = mock_result

        dummy_wav = AUDIO_DIR / "dummy_mock_test.wav"
        dummy_wav.write_bytes(b"RIFFtestheaderdata12345")
        try:
            out = transcribe_tamil_audio(str(dummy_wav))
            self.assertIsInstance(out, str, "transcribe_tamil_audio must return strictly str")
            self.assertEqual(out, "பெண்கள் சம உரிமை பெற வேண்டும்.")

            # Check that client.speech_to_text.convert was called with model_id="scribe_v2" and language_code="tam"
            mock_client.speech_to_text.convert.assert_called_once()
            _, kwargs = mock_client.speech_to_text.convert.call_args
            self.assertEqual(kwargs.get("model_id"), "scribe_v2")
            self.assertEqual(kwargs.get("language_code"), "tam")
        finally:
            if dummy_wav.exists():
                dummy_wav.unlink()

    # -------------------------------------------------------------
    # PHASE 3 TESTS: ASR Evaluation (WER, CER, S, I, D)
    # -------------------------------------------------------------
    def test_calculate_wer_and_cer(self):
        """Test WER & CER calculation."""
        text = "பெண்களும் ஆண்களும் சமமாக கல்வி பெற வேண்டும்."
        self.assertEqual(calculate_wer(text, text), 0.0)
        self.assertEqual(calculate_cer(text, text), 0.0)

    def test_calculate_error_details_breakdown(self):
        """Verify WER formula (S + I + D) / N."""
        ref = "பெண்கள் வீட்டை மட்டும் கவனிக்க வேண்டும் என்றனர்"
        hyp = "பெண்கள் மட்டுமே கவனிக்க வேண்டும் என்றனர் கூறப்பட்டது"

        details = calculate_error_details(ref, hyp)
        s = details["substitutions"]
        i = details["insertions"]
        d = details["deletions"]
        n = details["reference_words"]

        expected_wer = (s + i + d) / n
        self.assertAlmostEqual(details["wer"], expected_wer, places=4)

    # -------------------------------------------------------------
    # PHASE 4 TESTS: Text Classification & Neutrality Rules
    # -------------------------------------------------------------
    def test_classifier_demo_dataset(self):
        """Verify classifier performance on all 10 demo items."""
        df = pd.read_csv(DEMO_DATASET_PATH)
        for _, row in df.iterrows():
            res = self.classifier.predict(row["transcript"])
            expected_label = "MISOGYNISTIC" if row["label"] == 1 else "NON-MISOGYNISTIC"
            self.assertEqual(res["label"], expected_label, f"Failed on ID {row['id']}")
            self.assertEqual(res["category"], row["category"], f"Category mismatch on ID {row['id']}")
            self.assertIn("reason", res)
            self.assertIn("evidence", res)

            if res["evidence"]:
                self.assertIn(res["evidence"], row["transcript"])

    def test_neutral_gender_words_not_misogyny(self):
        """
        Rule 1, 2, 3: Words like woman, girl, mother alone must NOT trigger misogyny.
        A neutral statement about a woman is NOT misogyny.
        """
        neutral_samples = [
            "ஒரு பெண் சாலையில் நடந்து செல்கிறார்.",
            "என் தாய் உணவு தயாரித்தார்.",
            "அந்த சிறுமி புத்தகம் படிக்கிறாள்.",
            "அம்மா மற்றும் சகோதரி பேசுகிறார்கள்.",
            "அவர் ஒரு பெண் மருத்துவர்.",
        ]
        for s in neutral_samples:
            res = self.classifier.predict(s)
            self.assertEqual(res["label"], "NON-MISOGYNISTIC")
            self.assertEqual(res["category"], "NONE")
            self.assertIn(res["reason"], ["Insufficient evidence.", "The statement is neutral or supports equality."])
            self.assertEqual(res["evidence"], "")

    def test_insufficient_evidence_rule(self):
        """Rule 5: If evidence is insufficient, label=NON-MISOGYNISTIC, category=NONE, reason=Insufficient evidence."""
        res_empty = self.classifier.predict("")
        self.assertEqual(res_empty["label"], "NON-MISOGYNISTIC")
        self.assertEqual(res_empty["category"], "NONE")
        self.assertEqual(res_empty["reason"], "Insufficient evidence.")
        self.assertEqual(res_empty["evidence"], "")

    # -------------------------------------------------------------
    # PHASE 5 TESTS: Complete End-to-End Pipeline Integration
    # VIDEO -> AUDIO -> ASR -> TRANSCRIPT -> CLASSIFIER -> PREDICTION -> CATEGORY -> REASON
    # -------------------------------------------------------------
    def test_full_pipeline_end_to_end(self):
        """
        Test the complete pipeline:
        Video -> FFmpeg Audio Extraction (-vn) -> Tamil ASR -> Transcript -> Classifier -> Prediction + Category + Reason + Evidence
        """
        test_video = VIDEOS_DIR / "e2e_test_video.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "sine=frequency=800:duration=1.0",
            "-f", "lavfi", "-i", "color=c=green:s=128x128:d=1.0",
            "-c:a", "aac", "-c:v", "libx264",
            str(test_video),
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        self.assertTrue(test_video.exists())

        # Step 1: Extract Audio (strictly -vn)
        extracted_audio = extract_audio(test_video)
        self.assertTrue(extracted_audio.exists())
        self.assertGreater(extracted_audio.stat().st_size, 0)

        # Step 2: Tamil ASR (Speech-to-Text)
        demo_text = "பெண்களின் மதிப்பு அவர்களின் அழகில் மட்டும் இல்லை என்று நாம் புரிந்து கொள்ள வேண்டும்."
        asr_res = self.asr.transcribe(extracted_audio, save_txt=True, manual_override_text=demo_text)
        transcript = asr_res["transcript"]
        self.assertEqual(transcript, demo_text)
        self.assertTrue(Path(asr_res["saved_file"]).exists())

        # Step 3: Text Classifier
        pred = self.classifier.predict(transcript)

        # Step 4: Verification of Prediction, Category, Reason, Evidence
        self.assertEqual(pred["label"], "MISOGYNISTIC")
        self.assertEqual(pred["category"], "OBJECTIFICATION")
        self.assertTrue(len(pred["reason"]) > 0)
        self.assertEqual(pred["evidence"], "அழகில் மட்டும்")

        if test_video.exists():
            test_video.unlink()
        if extracted_audio.exists():
            extracted_audio.unlink()
        if Path(asr_res["saved_file"]).exists():
            Path(asr_res["saved_file"]).unlink()

    # -------------------------------------------------------------
    # GEMINI LLM EXPLAINER TESTS
    # -------------------------------------------------------------
    @patch("src.explainer.get_gemini_api_key")
    def test_gemini_explainer_fallback_when_unconfigured(self, mock_get_key):
        """When Gemini API key is missing, explainer must provide a safe rule-based fallback."""
        mock_get_key.return_value = None
        res = explain_misogyny_with_gemini(
            transcript="ஒரு பெண்ணின் உடையை வைத்து அவளை அவமதிப்பது தவறானது.",
            label="MISOGYNISTIC",
            category="SHAMING",
            evidence="உடையை வைத்து"
        )
        self.assertFalse(res["available"])
        self.assertIn("SHAMING", res["why_explanation"])
        self.assertTrue(len(res["tamil_explanation"]) > 0)
        self.assertIn("GEMINI_API_KEY is not configured", res["error"])

    @patch("google.genai.Client")
    @patch("src.explainer.get_gemini_api_key")
    def test_gemini_explainer_mocked_success(self, mock_get_key, mock_client_cls):
        """Verify Gemini explainer parses markdown sections correctly on successful response."""
        mock_get_key.return_value = "AIzaSyTestKeyDummy123"
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_resp = MagicMock()
        mock_resp.text = """### 1. Core Reasoning (Why this Classification)
The statement demeans women by judging modesty.

### 2. Cultural & Societal Context
In Tamil society, moral policing around attire is prevalent.

### 3. Linguistic Evidence Analysis
The phrase "உடையை வைத்து" targets clothing.

### 4. தமிழ் விளக்கம் (Tamil Summary)
இந்த வாக்கியம் உடை சார்ந்த பாலின பாகுபாட்டை குறிக்கிறது.
"""
        mock_client.models.generate_content.return_value = mock_resp

        res = explain_misogyny_with_gemini(
            transcript="ஒரு பெண்ணின் உடையை வைத்து அவளை அவமதிப்பது தவறானது.",
            label="MISOGYNISTIC",
            category="SHAMING",
            evidence="உடையை வைத்து"
        )

        self.assertTrue(res["available"])
        self.assertIn("The statement demeans women", res["why_explanation"])
        self.assertIn("Tamil society", res["cultural_context"])
        self.assertIn("உடையை வைத்து", res["evidence_analysis"])
        self.assertIn("இந்த வாக்கியம்", res["tamil_explanation"])
        self.assertIn("தமிழ் விளக்கம்", res["full_response"])
        self.assertEqual(res["model_used"], "gemini-2.5-flash")

    @patch("src.explainer.explain_misogyny_with_gemini")
    def test_classifier_explain_method(self, mock_explain):
        """Verify TamilMisogynyClassifier.explain delegates to explainer."""
        mock_explain.return_value = {
            "available": True,
            "why_explanation": "Test why",
            "tamil_explanation": "Test tamil",
        }
        exp = self.classifier.explain("பெண்களும் ஆண்களும் சமமாக கல்வி பெற வேண்டும்.")
        self.assertEqual(exp["why_explanation"], "Test why")
        self.assertEqual(exp["tamil_explanation"], "Test tamil")


if __name__ == "__main__":
    unittest.main()
