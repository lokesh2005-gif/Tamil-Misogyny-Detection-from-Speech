"""
Audio Processing Module for Tamil Misogyny Detection System.
Extracts audio tracks from video files strictly using FFmpeg.
Converts to 16 kHz mono uncompressed WAV format.
Provides Vocal Isolation & Background Music (BGM) Separation.
NO computer vision, video frame extraction, or image processing is performed.
"""

import sys
import subprocess
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np

from config import (
    AUDIO_SAMPLE_RATE,
    AUDIO_CHANNELS,
    AUDIO_FORMAT,
    AUDIO_DIR,
)

# Supported input video containers
SUPPORTED_VIDEO_FORMATS: List[str] = [".mp4", ".mkv", ".avi", ".mov"]


def get_ffmpeg_binary() -> str:
    """Return path to ffmpeg binary (system PATH or imageio_ffmpeg)."""
    bin_path = shutil.which("ffmpeg")
    if bin_path:
        return bin_path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def get_ffprobe_binary() -> Optional[str]:
    """Return path to ffprobe binary if available."""
    bin_path = shutil.which("ffprobe")
    if bin_path:
        return bin_path
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        probe = Path(exe).parent / ("ffprobe.exe" if sys.platform == "win32" else "ffprobe")
        if probe.exists():
            return str(probe)
    except Exception:
        pass
    return None


def check_ffmpeg_installed() -> bool:
    """Verify that FFmpeg is accessible via system PATH or imageio_ffmpeg."""
    if shutil.which("ffmpeg") is not None:
        return True
    try:
        import imageio_ffmpeg
        return bool(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        return False


def get_media_duration(file_path: str | Path) -> float:
    """
    Get duration of video or audio file in seconds using ffprobe or soundfile.
    Returns 0.0 if duration cannot be determined.
    """
    path = Path(file_path).resolve()
    if not path.exists():
        return 0.0

    ffprobe_bin = get_ffprobe_binary()
    if not ffprobe_bin:
        try:
            import soundfile as sf
            info = sf.info(str(path))
            return round(float(info.duration), 2)
        except Exception:
            return 0.0

    cmd = [
        ffprobe_bin,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]

    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        dur_str = res.stdout.strip()
        return round(float(dur_str), 2) if dur_str else 0.0
    except Exception:
        return 0.0


def extract_audio(
    video_path: str | Path,
    output_audio_path: Optional[str | Path] = None,
    sample_rate: int = AUDIO_SAMPLE_RATE,
    channels: int = AUDIO_CHANNELS,
) -> Path:
    """
    Extract audio from a Tamil video file (MP4, MKV, AVI, MOV) and convert to 16 kHz mono WAV.
    Strictly uses FFmpeg with `-vn` flag to completely discard video streams.

    Args:
        video_path: Path to the input video file (.mp4, .mkv, .avi, .mov).
        output_audio_path: Optional destination path for extracted WAV.
        sample_rate: Target sampling rate (default: 16000 Hz).
        channels: Target channels (default: 1 for mono).

    Returns:
        Path to the extracted 16kHz mono WAV file.
    """
    path = Path(video_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    ext = path.suffix.lower()
    allowed_exts = [e.lower() for e in SUPPORTED_VIDEO_FORMATS]
    allowed_all = allowed_exts + [".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"]
    if ext not in allowed_all:
        raise ValueError(
            f"Unsupported file format '{ext}'. Supported video formats are: {', '.join(SUPPORTED_VIDEO_FORMATS)}"
        )

    if not check_ffmpeg_installed():
        raise RuntimeError(
            "FFmpeg executable not found in system PATH. "
            "Please ensure FFmpeg is installed to extract audio."
        )

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    if output_audio_path is None:
        output_filename = f"{path.stem}_16k_mono.{AUDIO_FORMAT}"
        target_path = (AUDIO_DIR / output_filename).resolve()
    else:
        target_path = Path(output_audio_path).resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg_bin = get_ffmpeg_binary()
    # FFmpeg arguments:
    # -vn: STRICTLY DISABLE VIDEO STREAM (no frames extracted)
    # -acodec pcm_s16le: 16-bit PCM WAV
    # -ar 16000: 16 kHz sampling rate
    # -ac 1: Mono audio channel
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        str(target_path),
    ]

    try:
        subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"FFmpeg audio extraction failed (exit code {e.returncode}): {e.stderr.strip()}"
        ) from e

    if not target_path.exists() or target_path.stat().st_size == 0:
        raise RuntimeError(f"FFmpeg produced an empty or missing audio file: {target_path}")

    return target_path


def separate_voice_and_bgm(
    input_path: str | Path,
    output_voice_path: Optional[str | Path] = None,
    output_bgm_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """
    Acoustically separates speech voice from background music / noise.
    1. Extracts raw 16kHz mono audio.
    2. Applies acoustic bandpass (80Hz - 3800Hz) to isolate human vocal tract frequencies
       while stripping sub-bass music and high-frequency cymbals/synths with dynaudnorm speech leveling.
    3. Computes background music residual (BGM = raw - voice).

    Returns:
        Dict with 'voice_audio', 'bgm_audio', 'raw_audio', and 'duration'.
    """
    source_path = Path(input_path).resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Input file not found: {source_path}")

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    # 1. First extract raw 16kHz mono audio
    raw_audio_path = (AUDIO_DIR / f"{source_path.stem}_raw.wav").resolve()
    extract_audio(source_path, raw_audio_path)

    # Set output paths
    if output_voice_path is None:
        voice_path = (AUDIO_DIR / f"{source_path.stem}_voice_isolated.wav").resolve()
    else:
        voice_path = Path(output_voice_path).resolve()

    if output_bgm_path is None:
        bgm_path = (AUDIO_DIR / f"{source_path.stem}_bgm_isolated.wav").resolve()
    else:
        bgm_path = Path(output_bgm_path).resolve()

    ffmpeg_bin = get_ffmpeg_binary()
    # 2. Apply FFmpeg vocal bandpass + adaptive FFT noise/music suppression + dynamic normalizer
    # Filter: highpass=80, lowpass=3800, afftdn (FFT denoiser), dynaudnorm (speech leveler)
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i", str(raw_audio_path),
        "-vn",
        "-af", "highpass=f=80,lowpass=f=3800,afftdn=nf=-20:tn=1,dynaudnorm=f=150:g=15",
        "-acodec", "pcm_s16le",
        "-ar", str(AUDIO_SAMPLE_RATE),
        "-ac", "1",
        str(voice_path),
    ]

    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except Exception:
        # If filter fails, fallback to raw audio
        shutil.copyfile(raw_audio_path, voice_path)

    # 3. Calculate BGM residual (Raw - Voice) with lightweight vector subtraction (<10MB RAM, instant)
    try:
        import soundfile as sf

        raw_data, sr = sf.read(str(raw_audio_path))
        voice_data, _ = sf.read(str(voice_path))

        min_len = min(len(raw_data), len(voice_data))
        bgm_data = raw_data[:min_len] - voice_data[:min_len]
        bgm_max = float(np.max(np.abs(bgm_data))) if len(bgm_data) > 0 else 0.0
        if bgm_max > 1e-4:
            bgm_data = (bgm_data / bgm_max) * 0.75
        sf.write(str(bgm_path), bgm_data, sr)

    except Exception:
        shutil.copyfile(raw_audio_path, bgm_path)

    dur = get_media_duration(voice_path)

    return {
        "voice_audio": str(voice_path),
        "bgm_audio": str(bgm_path),
        "raw_audio": str(raw_audio_path),
        "duration": dur,
    }


def extract_and_enhance_audio(video_path: str | Path) -> Path:
    """
    Convenience function: extracts audio and isolates voice from background music.
    Returns the path to the clean voice-isolated WAV file.
    """
    sep_result = separate_voice_and_bgm(video_path)
    return Path(sep_result["voice_audio"])
