"""Optional real-model smoke test, using local recordings and an isolated database."""

import argparse
import json
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

import numpy as np
import soundfile as sf
from voice_studio.audio import prepare_audio
from voice_studio.config import Paths
from voice_studio.models import ModelHub
from voice_studio.speaker_service import SpeakerService
from voice_studio.speaker_store import ProfileError, SpeakerStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, required=True, help="Local JSON containing at least three profile folders."
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    configured = Paths.load(ROOT)
    result = []
    with tempfile.TemporaryDirectory(prefix="speaker-integration-", dir=configured.runtime) as directory:
        work = Path(directory)
        paths = Paths(ROOT, work / "private", work / "runtime", configured.models)
        paths.runtime.mkdir(parents=True)
        models = ModelHub(paths)
        store = SpeakerStore(paths.database)
        service = SpeakerService(store, models, paths)
        groups = []
        for profile in manifest["profiles"][:3]:
            prepared = []
            for file in sorted(Path(profile["folder"]).glob("*.wav"))[:5]:
                with sf.SoundFile(file) as stream:
                    audio = stream.read(frames=int(stream.samplerate * 20), dtype="float32")
                    candidate = work / (uuid.uuid4().hex + ".wav")
                    sf.write(candidate, audio, stream.samplerate)
                prepared.append(prepare_audio(candidate, work / "inputs", enrollment=True).path)
            groups.append(prepared)

        def passed(name, details=None):
            result.append({"name": name, "passed": True, "details": details})
            print(name + ": PASS", flush=True)

        service.enroll("example_a", "示例用户 A", groups[0][:3])
        service.enroll("example_b", "示例用户 B", groups[1][:3], retain_audio=True)
        passed("register_two_new_users")
        calibration = service.calibrate()
        passed("calibrate_and_persist_threshold", round(calibration["threshold"], 4))
        _, known = service.identify(groups[0][0])
        assert known["speaker_id"] == "example_a", known
        passed("identify_registered_reference", known["score"])
        _, unknown = service.identify(groups[2][0])
        assert unknown["status"] == "unknown", unknown
        passed("reject_unregistered_speaker", unknown["score"])
        store.rename("example_a", "示例用户 A 已改名", 1)
        service.update("example_a", [groups[0][3]], 2)
        assert store.get_profile("example_a")["sample_count"] == 4
        passed("rename_and_append")
        service.update("example_a", groups[0][:3], 3, replace=True)
        assert store.get_profile("example_a")["sample_count"] == 3
        passed("replace_reference_recordings")
        check = (
            "import sys;sys.path.insert(0,sys.argv[1]);"
            "from voice_studio.speaker_store import SpeakerStore;"
            "s=SpeakerStore(sys.argv[2]);"
            "assert len(s.list_profiles())==2;"
            "assert s.get_profile('example_a')['sample_count']==3;"
            "assert s.get_settings()['calibration'] is not None;"
            "print('persisted')"
        )
        subprocess.run(
            [sys.executable, "-c", check, str(ROOT / "app"), str(paths.database)], check=True, capture_output=True
        )
        passed("reload_features_in_new_process")
        service.delete("example_b", 1)
        assert store.get_profile("example_b") is None
        assert not list(paths.retained_audio.glob("*.wav"))
        _, deleted = service.identify(groups[1][0])
        assert deleted["status"] == "unknown", deleted
        passed("delete_profile_features_and_saved_audio")
        silent = work / "silence.wav"
        sf.write(silent, np.zeros(48000), 16000)
        try:
            service.enroll("invalid", "静音测试", [silent] * 3)
        except ProfileError:
            pass
        else:
            raise AssertionError("Silent enrollment was accepted")
        assert store.get_profile("invalid") is None
        passed("reject_silent_registration")
    (configured.runtime / "real-speaker-validation.json").write_text(
        json.dumps(
            {"tests": result, "scope": "Functional smoke tests, not an accuracy benchmark."},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("REAL_MODEL_SPEAKER_VALIDATION_PASSED", flush=True)


if __name__ == "__main__":
    main()
