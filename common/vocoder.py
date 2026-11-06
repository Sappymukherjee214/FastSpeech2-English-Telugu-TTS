from pathlib import Path
from typing import Any, Optional

import librosa
import numpy as np
import soundfile as sf


def griffin_lim_vocoder(
    mel_spectrogram: np.ndarray,
    config: dict[str, Any],
    n_iter: int = 60,
) -> np.ndarray:
    """Reconstruct waveform from log mel-spectrogram using Griffin-Lim.

    Args:
        mel_spectrogram: Array of shape [time, n_mels] containing log-magnitude
            mel values.
        config: Full configuration mapping or direct audio configuration mapping.
        n_iter: Number of Griffin-Lim iterations.

    Returns:
        1D float32 numpy array representing the synthesized audio waveform.
    """
    audio_cfg = config.get("audio", config)
    sample_rate = int(audio_cfg["sample_rate"])
    n_fft = int(audio_cfg["n_fft"])
    hop_length = int(audio_cfg["hop_length"])
    n_mels = int(audio_cfg["n_mels"])
    f_min = float(audio_cfg.get("f_min", 0.0))
    f_max_val = audio_cfg.get("f_max", 8000.0)
    f_max = float(f_max_val) if f_max_val is not None else None

    # Transpose back from [time, n_mels] to [n_mels, time] and undo log
    mel_linear = np.exp(np.clip(mel_spectrogram.T, -12.0, 12.0))

    # Compute inverse mel basis to map mel channels back to linear frequency bins
    mel_basis = librosa.filters.mel(
        sr=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=f_min,
        fmax=f_max,
    )
    inv_mel_basis = np.linalg.pinv(mel_basis)
    linear_spec = np.maximum(1e-5, np.dot(inv_mel_basis, mel_linear))

    # Reconstruct phase using Griffin-Lim
    audio = librosa.griffinlim(
        linear_spec,
        n_iter=n_iter,
        hop_length=hop_length,
        win_length=n_fft,
        n_fft=n_fft,
    )

    # Normalize amplitude to [-0.95, 0.95] to prevent clipping
    max_val = float(np.max(np.abs(audio)))
    if max_val > 1e-6:
        audio = (audio / max_val) * 0.95

    return audio.astype(np.float32)


def save_audio(
    audio: np.ndarray,
    path: str | Path,
    sample_rate: int,
) -> None:
    """Save 1D audio array to a WAV file."""
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(target_path), audio, sample_rate, subtype="PCM_16")


def synthesize(
    mel_spectrogram: np.ndarray,
    config: dict[str, Any],
    output_path: Optional[str | Path] = None,
    n_iter: int = 60,
) -> np.ndarray:
    """Synthesize audio waveform from mel spectrogram and optionally save it.

    Args:
        mel_spectrogram: Array of shape [time, n_mels].
        config: Configuration dictionary with audio settings.
        output_path: Optional path to save the resulting .wav file.
        n_iter: Number of Griffin-Lim iterations.

    Returns:
        1D float32 numpy array representing the synthesized audio.
    """
    audio = griffin_lim_vocoder(mel_spectrogram, config, n_iter=n_iter)
    if output_path is not None:
        audio_cfg = config.get("audio", config)
        sample_rate = int(audio_cfg["sample_rate"])
        save_audio(audio, output_path, sample_rate)
    return audio
