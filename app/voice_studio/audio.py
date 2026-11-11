"""Audio decoding, normalization and basic enrollment quality checks."""

import hashlib
import math
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000


class AudioError(ValueError):
    pass


@dataclass(frozen=True)
class AudioQuality:
    duration: float
    active_duration: float
    rms_dbfs: float
    clipping_ratio: float
    sample_rate: int = SAMPLE_RATE

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class PreparedAudio:
    path: Path
    sha256: str
    quality: AudioQuality


def check_waveform(audio, sample_rate=SAMPLE_RATE, *, enrollment=False, strict_quality=True):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
        raise AudioError("音频为空或包含无效数值，请重新录制。")
    duration = len(audio) / sample_rate
    minimum, maximum = (2.0, 30.0) if enrollment else (1.0, 60.0)
    if duration < minimum:
        raise AudioError(f"录音太短，请提供至少 {minimum:g} 秒的音频。")
    if duration > maximum:
        raise AudioError(f"录音太长，请截取 {maximum:g} 秒以内的片段。")
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    rms_dbfs = 20 * math.log10(max(rms, 1e-12))
    clipping = float(np.mean(np.abs(audio) >= 0.999))
    frame = int(sample_rate * 0.02)
    padded = np.pad(audio, (0, (-len(audio)) % frame))
    energies = np.sqrt(np.mean(padded.reshape(-1, frame).astype(np.float64) ** 2, axis=1))
    active = float(np.sum(energies > max(0.003, float(energies.max()) * 0.08)) * 0.02)
    if (enrollment or strict_quality) and (rms_dbfs < -45 or active < (1.0 if enrollment else 0.5)):
        raise AudioError("有效声音不足或音量太低，请靠近麦克风，在安静环境中清晰说话。")
    if (enrollment or strict_quality) and clipping > 0.05:
        raise AudioError("录音有明显削波失真，请降低麦克风增益后重录。")
    return AudioQuality(round(duration, 3), round(active, 3), round(rms_dbfs, 2), round(clipping, 5))


def prepare_audio(path, work_dir, *, enrollment=False, strict_quality=True):
    path = Path(path)
    if not path.is_file():
        raise AudioError("请先录制或上传音频。")
    if path.stat().st_size > 50 * 1024 * 1024:
        raise AudioError("文件超过 50 MiB，请先截取需要的语音片段。")
    try:
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
    except Exception:
        try:
            import librosa

            mono, rate = librosa.load(path, sr=None, mono=True)
            audio = mono[:, None]
        except Exception as exc:
            raise AudioError("无法读取音频，请上传有效的 WAV、MP3 或 FLAC 文件。") from exc
    audio = np.mean(audio, axis=1)
    if rate != SAMPLE_RATE:
        from scipy.signal import resample_poly

        divisor = math.gcd(int(rate), SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // divisor, int(rate) // divisor).astype(np.float32)
    quality = check_waveform(audio, enrollment=enrollment, strict_quality=strict_quality)
    pcm = np.clip(np.rint(np.clip(audio, -1.0, 1.0) * 32768.0), -32768, 32767).astype("<i2")
    fingerprint = hashlib.sha256(pcm.tobytes()).hexdigest()
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    normalized = work_dir / (uuid.uuid4().hex + ".wav")
    sf.write(normalized, pcm, SAMPLE_RATE, subtype="PCM_16")
    return PreparedAudio(normalized, fingerprint, quality)


def capture_local_microphone(output_dir, seconds=5):
    """Record only when explicitly invoked by the local user's button."""
    if not 1 <= seconds <= 10:
        raise AudioError("本机录音时长应在 1–10 秒之间。")
    try:
        import sounddevice as sd

        rate = SAMPLE_RATE
        try:
            sd.check_input_settings(channels=1, samplerate=rate, dtype="float32")
        except sd.PortAudioError:
            rate = int(sd.query_devices(kind="input")["default_samplerate"])
        audio = sd.rec(int(seconds * rate), samplerate=rate, channels=1, dtype="float32")
        sd.wait()
        if not np.isfinite(audio).all() or not audio.size:
            raise AudioError("麦克风没有返回有效音频，请检查输入设备。")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        file = output_dir / ("microphone_" + uuid.uuid4().hex + ".wav")
        sf.write(file, audio, rate, subtype="PCM_16")
        return str(file)
    except Exception as exc:
        if isinstance(exc, AudioError):
            raise
        raise AudioError("本机麦克风暂不可用，请检查设备和系统权限，或改用音频上传。") from exc
