"""Enrollment and profile operations, independent of the Gradio interface."""

import shutil
import uuid
from pathlib import Path

import numpy as np

from .audio import prepare_audio
from .recognition import calibrate, identify
from .speaker_store import ProfileError, Sample, unit_vector


class SpeakerService:
    def __init__(self, store, models, paths):
        self.store, self.models, self.paths = store, models, paths

    @staticmethod
    def _check_consistency(vectors):
        if len(vectors) < 2:
            return
        matrix = np.stack([unit_vector(vector) for vector in vectors])
        values = (matrix @ matrix.T)[np.triu_indices(len(matrix), 1)]
        if float(values.min()) < 0.20:
            raise ProfileError("参考录音的声纹差异较大，请确认所有片段来自同一个人。")

    def _samples(self, files, retain_audio):
        files = list(files or [])
        if not files or len(files) > 20:
            raise ProfileError("请提供 1–20 段参考录音。")
        prepared, copied, samples = [], [], []
        try:
            for index, file in enumerate(files, 1):
                try:
                    audio = prepare_audio(file, self.paths.runtime / "enrollment-tmp", enrollment=True)
                    prepared.append(audio.path)
                    vector = self.models.speaker_embedding(audio.path)
                except ValueError as exc:
                    raise ProfileError(f"第 {index} 段录音：{exc}") from exc
                saved = None
                if retain_audio:
                    self.paths.retained_audio.mkdir(parents=True, exist_ok=True)
                    saved = self.paths.retained_audio / (uuid.uuid4().hex + ".wav")
                    shutil.copy2(audio.path, saved)
                    copied.append(saved)
                samples.append(Sample(vector, audio.sha256, audio.quality.to_dict(), str(saved) if saved else None))
            self._check_consistency([sample.embedding for sample in samples])
            return samples
        except Exception:
            self._purge(copied)
            raise
        finally:
            for path in prepared:
                path.unlink(missing_ok=True)

    def _purge(self, paths):
        failures = []
        for value in paths:
            if not value:
                continue
            path = Path(value).resolve()
            if not path.is_relative_to(self.paths.retained_audio.resolve()) or path.suffix != ".wav":
                failures.append("有文件不在本项目录音目录内，已保留。")
                continue
            try:
                path.unlink(missing_ok=True)
            except OSError:
                failures.append("有录音文件正在使用，未能删除，请稍后手动处理。")
        return failures

    def enroll(self, speaker_id, display_name, files, retain_audio=False):
        self.store._identity(speaker_id, display_name)
        if self.store.get_profile(str(speaker_id).strip().lower()):
            raise ProfileError("这个用户 ID 已存在，请选择更新档案或使用其他 ID。")
        if not 3 <= len(files or []) <= 20:
            raise ProfileError("登记需要 3–20 段不同的参考录音，建议每段 5–10 秒。")
        model_id = self.models.model_id
        samples = self._samples(files, retain_audio)
        try:
            return self.store.enroll(speaker_id, display_name, samples, model_id)
        except Exception:
            self._purge([sample.audio_path for sample in samples])
            raise

    def update(self, speaker_id, files, expected_revision, *, replace=False, retain_audio=False):
        model_id = self.models.model_id
        samples = self._samples(files, retain_audio)
        try:
            profiles, _ = self.store.snapshot()
            existing = next((profile for profile in profiles if profile["speaker_id"] == speaker_id), None)
            if existing and not replace:
                if existing["model_id"] != model_id:
                    raise ProfileError("声纹模型已变化，请替换全部参考录音。")
                self._check_consistency(existing["samples"] + [sample.embedding for sample in samples])
            old_audio = self.store.update_samples(speaker_id, samples, model_id, expected_revision, replace=replace)
        except Exception:
            self._purge([sample.audio_path for sample in samples])
            raise
        return self._purge(old_audio)

    def delete(self, speaker_id, expected_revision):
        return self._purge(self.store.delete(speaker_id, expected_revision))

    def identify(self, path):
        audio = prepare_audio(path, self.paths.runtime / "inputs", strict_quality=False)
        profiles, fingerprint = self.store.snapshot()
        settings = self.store.get_settings()
        if audio.quality.rms_dbfs < -45 or audio.quality.active_duration < 0.5 or audio.quality.clipping_ratio > 0.05:
            result = {
                "status": "invalid_audio",
                "speaker_id": None,
                "reason": "音量太低、有效声音不足或失真较多，暂不判断身份。可先尝试增强或重新录音。",
            }
        elif profiles:
            result = identify(
                self.models.speaker_embedding(audio.path),
                profiles,
                self.models.model_id,
                settings["threshold"],
                settings["margin"],
            )
        else:
            result = {"status": "empty", "reason": "尚未登记任何用户，请先到说话人管理中登记。", "speaker_id": None}
        calibration = settings.get("calibration")
        result["calibration"] = (
            "未校准"
            if not calibration
            else ("有效" if calibration["fingerprint"] == fingerprint else "档案已变化，建议重新校准")
        )
        result["quality"] = audio.quality.to_dict()
        return str(audio.path), result

    def calibrate(self):
        profiles, fingerprint = self.store.snapshot()
        settings = self.store.get_settings()
        report = calibrate(profiles, self.models.model_id, fingerprint)
        self.store.save_settings(report["threshold"], settings["margin"], report, expected_fingerprint=fingerprint)
        return report
