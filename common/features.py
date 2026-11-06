from pathlib import Path

import librosa
import numpy as np


def load_audio(path: str | Path, sample_rate: int):
    audio, rate = librosa.load(str(path), sr=sample_rate, mono=True)
    if not len(audio):
        raise ValueError(f"Empty audio: {path}")
    return audio.astype(np.float32), rate


def audio_duration(path: str | Path) -> float:
    return float(librosa.get_duration(path=str(path)))


def extract_features(audio, sample_rate, n_fft, hop_length, n_mels, f_min, f_max):
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, fmin=f_min, fmax=f_max, power=1.0,
    )
    mel = np.log(np.maximum(mel, 1e-5)).T.astype(np.float32)
    pitch_fmin = max(50.0, float(f_min)) if f_min else 50.0
    pitch_fmax = min(float(f_max), 600.0) if f_max else 600.0
    pitch = librosa.yin(audio, fmin=pitch_fmin, fmax=pitch_fmax,
                        sr=sample_rate, frame_length=n_fft, hop_length=hop_length)
    energy = librosa.feature.rms(y=audio, frame_length=n_fft, hop_length=hop_length)[0]
    return {
        "mel": mel,
        "pitch": np.nan_to_num(pitch[:len(mel)]).astype(np.float32),
        "energy": np.log(np.maximum(energy[:len(mel)], 1e-5)).astype(np.float32),
    }


def save_features(features, path):
    np.savez_compressed(path, **features)


def load_features(path):
    with np.load(path) as data:
        return {key: data[key] for key in data.files}
