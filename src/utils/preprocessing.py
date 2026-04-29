"""Audio preprocessing utilities."""

import subprocess
import tempfile
import torch
import torchaudio
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_RATE = 16000
MAX_DURATION_SEC = 30
MAX_SAMPLES = DEFAULT_SAMPLE_RATE * MAX_DURATION_SEC  # 480,000


def load_and_preprocess(
    path: str | Path,
    target_sr: int = DEFAULT_SAMPLE_RATE,
    max_samples: int = MAX_SAMPLES,
    trim_silence: bool = True,
    trim_threshold_db: float = -40.0,
    normalize: bool = True,
) -> torch.Tensor:
    """Load a WAV file and preprocess for HuBERT.

    Args:
        path: Path to WAV file
        target_sr: Target sample rate
        max_samples: Maximum number of samples (truncate longer)
        trim_silence: Whether to trim leading/trailing silence
        trim_threshold_db: Silence threshold in dB
        normalize: Whether to normalize amplitude to [-1, 1]

    Returns:
        1D tensor of shape (samples,) at target_sr
    """
    ext = Path(path).suffix.lower()
    if ext in (".m4a", ".mp3", ".aac"):
        # Check for pre-converted wav sibling (from convert_m4a_to_wav.py)
        wav_sibling = Path(path).with_suffix(".wav")
        if wav_sibling.exists():
            waveform, sr = torchaudio.load(str(wav_sibling))
        else:
            # Fallback: convert via ffmpeg subprocess (slow)
            waveform, sr = _load_via_ffmpeg(str(path), target_sr)
    else:
        waveform, sr = torchaudio.load(str(path))

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != target_sr and waveform.numel() > 0:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)

    # Squeeze to 1D
    waveform = waveform.squeeze(0)

    # Guard against empty waveforms (corrupt/empty WAV files)
    if waveform.numel() == 0:
        logger.warning("Empty waveform for %s, returning silence", path)
        return torch.zeros(1)

    # Trim silence
    if trim_silence and waveform.numel() > 0:
        waveform = _trim_silence(waveform, trim_threshold_db)

    # Normalize
    if normalize and waveform.numel() > 0:
        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform / peak

    # Truncate or pad
    if waveform.shape[0] > max_samples:
        waveform = waveform[:max_samples]

    return waveform


def _load_via_ffmpeg(path: str, target_sr: int = 16000) -> tuple[torch.Tensor, int]:
    """Load audio via ffmpeg subprocess (for m4a/AAC files that soundfile can't read)."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        try:
            result = subprocess.run(
                ["ffmpeg", "-y", "-i", path, "-ar", str(target_sr), "-ac", "1",
                 "-f", "wav", tmp.name],
                capture_output=True, timeout=30,
            )
        except subprocess.TimeoutExpired:
            logger.warning("ffmpeg timed out for %s, returning silence", path)
            return torch.zeros(1, 1), target_sr
        if result.returncode != 0:
            logger.warning("ffmpeg failed for %s: %s", path, result.stderr[:200])
            return torch.zeros(1, 1), target_sr
        waveform, sr = torchaudio.load(tmp.name)
    return waveform, sr


def _trim_silence(waveform: torch.Tensor, threshold_db: float) -> torch.Tensor:
    """Trim leading and trailing silence from a 1D waveform."""
    threshold = 10 ** (threshold_db / 20)
    above = waveform.abs() > threshold

    if not above.any():
        return waveform

    nonzero = torch.nonzero(above).squeeze()
    if nonzero.dim() == 0:
        return waveform

    start = nonzero[0].item()
    end = nonzero[-1].item() + 1
    return waveform[start:end]


def pad_waveform(waveform: torch.Tensor, target_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad waveform to target length, returning padded waveform and attention mask.

    Args:
        waveform: 1D tensor
        target_length: Desired length

    Returns:
        (padded_waveform, attention_mask) both of shape (target_length,)
    """
    length = waveform.shape[0]
    if length >= target_length:
        return waveform[:target_length], torch.ones(target_length, dtype=torch.long)

    padded = torch.zeros(target_length)
    padded[:length] = waveform
    mask = torch.zeros(target_length, dtype=torch.long)
    mask[:length] = 1
    return padded, mask


def collate_audio(batch: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate variable-length waveforms into a padded batch.

    Args:
        batch: List of 1D tensors

    Returns:
        (padded_batch, attention_masks) of shape (B, max_len)
    """
    max_len = min(max(w.shape[0] for w in batch), MAX_SAMPLES)
    padded = []
    masks = []
    for w in batch:
        p, m = pad_waveform(w, max_len)
        padded.append(p)
        masks.append(m)
    return torch.stack(padded), torch.stack(masks)
