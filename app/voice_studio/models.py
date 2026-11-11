"""Lazy, serialized adapters around the original pretrained speech models."""

import hashlib
import json
import threading
from pathlib import Path

import numpy as np
import soundfile as sf

from .speaker_store import ProfileError, unit_vector


class ModelHub:
    def __init__(self, paths, device="auto"):
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("VOICE_DEVICE must be auto, cpu or cuda.")
        self.paths = paths
        self.requested_device = device
        self._speaker = None
        self._emotion = None
        self._asr = None
        self._fingerprint = None
        self._lock = threading.RLock()

    @property
    def device(self):
        import torch

        if self.requested_device == "cuda" and not torch.cuda.is_available():
            raise ProfileError("当前环境无法使用 CUDA，请选择 CPU 或检查显卡运行环境。")
        return (
            ("cuda" if torch.cuda.is_available() else "cpu")
            if self.requested_device == "auto"
            else self.requested_device
        )

    @property
    def model_id(self):
        if self._fingerprint is None:
            file = Path(self.paths.models["speaker"]) / "campplus_cn_common.bin"
            if not file.is_file():
                raise ProfileError("声纹模型尚未准备，请按照 README 完成首次安装。")
            sha = hashlib.sha256()
            with file.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    sha.update(block)
            self._fingerprint = "cam++:cn-common:" + sha.hexdigest()
        return self._fingerprint

    def _auto_model(self, key):
        if not Path(self.paths.models[key]).is_dir():
            raise ProfileError("模型文件尚未准备，请按照 README 完成首次安装。")
        from funasr import AutoModel

        return AutoModel(
            model=self.paths.models[key], hub="ms", device=self.device, disable_update=True, disable_pbar=True
        )

    def speaker_embedding(self, path):
        import torch

        with self._lock, torch.inference_mode():
            if self._speaker is None:
                self._speaker = self._auto_model("speaker")
            output = self._speaker.generate(str(path), output_dir=None, extract_embedding=True, disable_pbar=True)
            vector = output[0]["spk_embedding"]
            if isinstance(vector, torch.Tensor):
                vector = vector.detach().cpu().numpy()
            return unit_vector(vector)

    def emotion(self, path):
        import torch

        with self._lock, torch.inference_mode():
            if self._emotion is None:
                self._emotion = self._auto_model("emotion")
            output = self._emotion.generate(
                str(path), granularity="utterance", extract_embedding=False, disable_pbar=True
            )
            result = output[0]
            scores = np.asarray(result["scores"], dtype=float)
            if not np.isfinite(scores).all() or not scores.size:
                raise ProfileError("情绪模型未返回有效结果。")
            index = int(scores.argmax())
            return {"label": str(result["labels"][index]), "score": float(scores[index])}

    def transcribe(self, path):
        from vosk import KaldiRecognizer, Model

        with self._lock:
            if self._asr is None:
                directory = Path(self.paths.models["asr"])
                if not (directory / "am" / "final.mdl").is_file():
                    raise ProfileError("转写模型尚未准备，请按照 README 完成首次安装。")
                self._asr = Model(str(directory))
            audio, rate = sf.read(path, dtype="int16")
            if rate != 16000 or audio.ndim != 1:
                raise ProfileError("转写需要标准化后的单声道音频。")
            recognizer = KaldiRecognizer(self._asr, rate)
            pieces = []
            for offset in range(0, len(audio), 4000):
                if recognizer.AcceptWaveform(audio[offset : offset + 4000].tobytes()):
                    pieces.append(json.loads(recognizer.Result()).get("text", ""))
            pieces.append(json.loads(recognizer.FinalResult()).get("text", ""))
            return "".join(pieces).replace(" ", "") or "未识别到有效内容，请换一段清晰的语音。"

    def warm_up(self):
        from vosk import Model

        with self._lock:
            self._speaker = self._auto_model("speaker")
            self._emotion = self._auto_model("emotion")
            self._asr = Model(self.paths.models["asr"])
