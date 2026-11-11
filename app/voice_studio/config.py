"""Portable paths; all personal data and model files stay outside source control."""

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    root: Path
    private: Path
    runtime: Path
    models: dict

    @property
    def database(self):
        return self.private / "speakers.sqlite3"

    @property
    def retained_audio(self):
        return self.private / "audio"

    @classmethod
    def load(cls, root=None):
        source = Path(__file__).resolve()
        default_root = source.parents[2] if source.parents[1].name == "app" else Path.cwd()
        root = Path(root or os.environ.get("VOICE_PROJECT_DIR", default_root)).resolve()
        private = Path(os.environ.get("VOICE_DATA_DIR", root / "private")).resolve()
        runtime = Path(os.environ.get("VOICE_RUNTIME_DIR", root / "runtime")).resolve()
        for directory in (private, runtime, private / "audio", runtime / "outputs"):
            directory.mkdir(parents=True, exist_ok=True)
        manifest_path = root / "runtime" / "models.json"
        models = json.loads(manifest_path.read_text(encoding="utf-8-sig")) if manifest_path.is_file() else {}
        defaults = {
            "speaker": root / "models" / "speech_campplus_sv_zh-cn_16k-common",
            "emotion": root / "models" / "emotion2vec_plus_large",
            "asr": root / "models" / "vosk-model-small-cn-0.22",
        }
        if not defaults["asr"].exists() and (root / "app" / "model").is_dir():
            defaults["asr"] = root / "app" / "model"
        for key, default in defaults.items():
            candidate = Path(models.get(key, default))
            models[key] = str((candidate if candidate.is_absolute() else root / candidate).resolve())
        return cls(root, private, runtime, models)
