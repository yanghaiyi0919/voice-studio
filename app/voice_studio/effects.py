"""Deterministic pitch/time/filter effects derived from the original project."""

import uuid
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy import signal

from .audio import AudioError


def transform(method, input_path, output_dir):
    audio, rate = librosa.load(input_path, sr=None)
    if method == "spongebob":
        shifted = librosa.effects.pitch_shift(audio, sr=rate, n_steps=4)
        stretched = librosa.effects.time_stretch(shifted, rate=1.2)
        filtered = signal.sosfilt(signal.butter(6, 1500, "high", fs=rate, output="sos"), stretched)
        time = np.arange(len(filtered)) / rate
        result = 0.8 * filtered * (0.9 + 0.1 * np.sin(2 * np.pi * 4 * time)) + 0.2 * stretched
    elif method == "fatcat":
        shifted = librosa.effects.pitch_shift(audio, sr=rate, n_steps=-3.5)
        stretched = librosa.effects.time_stretch(shifted, rate=0.85)
        filtered = signal.sosfilt(signal.butter(6, 500, "low", fs=rate, output="sos"), stretched)
        impulse = np.exp(-np.linspace(0, 5, int(0.1 * rate)))
        impulse /= impulse.sum()
        reverb = signal.fftconvolve(filtered, impulse, mode="same")
        result = (0.65 * stretched + 0.35 * reverb) * (
            0.8 + 0.2 * np.sin(2 * np.pi * 2 * np.arange(len(stretched)) / rate)
        )
    elif method == "minion":
        shifted = librosa.effects.pitch_shift(audio, sr=rate, n_steps=4)
        stretched = librosa.effects.time_stretch(shifted, rate=1.15)
        filtered = signal.sosfilt(signal.butter(8, [800, 3500], "bandpass", fs=rate, output="sos"), stretched)
        mixed = np.round((0.8 * filtered + 0.2 * stretched) * 256) / 256
        result = mixed * (0.8 + 0.2 * np.sin(2 * np.pi * 20 * np.arange(len(mixed)) / rate))
        delay = int(0.03 * rate)
        echo = np.zeros_like(result)
        echo[delay:] = result[:-delay] * 0.15
        result += echo
    else:
        raise AudioError("未知的变声模式。")
    peak = float(np.max(np.abs(result)))
    if not np.isfinite(result).all() or peak < 1e-8:
        raise AudioError("变声未得到有效音频，请换一段清晰的录音。")
    output = Path(output_dir) / (method + "_" + uuid.uuid4().hex + ".wav")
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, result / peak * 0.8, rate, subtype="PCM_16")
    return str(output)
