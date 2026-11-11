"""One-time local migration from an explicitly supplied, untracked JSON manifest."""

import argparse
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

import soundfile as sf
from voice_studio.audio import prepare_audio
from voice_studio.config import Paths
from voice_studio.models import ModelHub
from voice_studio.speaker_service import SpeakerService
from voice_studio.speaker_store import SpeakerStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "private" / "legacy_import.json")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    paths = Paths.load(ROOT)
    store = SpeakerStore(paths.database)
    service = SpeakerService(store, ModelHub(paths), paths)
    migration_dir = paths.runtime / "migration"
    migration_dir.mkdir(exist_ok=True)
    report = []
    for index, profile in enumerate(manifest["profiles"], 1):
        speaker_id = profile["speaker_id"]
        if store.get_profile(speaker_id):
            print(f"Profile {index}: already imported.", flush=True)
            continue
        prepared, generated, rejected = [], [], []
        try:
            for source in sorted(Path(profile["folder"]).glob("*.wav")):
                candidate = source
                with sf.SoundFile(source) as stream:
                    if stream.frames / stream.samplerate > 30:
                        audio = stream.read(frames=int(stream.samplerate * 20), dtype="float32")
                        candidate = migration_dir / (uuid.uuid4().hex + ".wav")
                        sf.write(candidate, audio, stream.samplerate)
                        generated.append(candidate)
                try:
                    asset = prepare_audio(candidate, migration_dir, enrollment=True)
                    prepared.append(asset.path)
                    generated.append(asset.path)
                except ValueError as error:
                    rejected.append({"file": source.name, "reason": str(error)})
            service.enroll(speaker_id, profile["display_name"], prepared)
            report.append({"speaker_id": speaker_id, "imported": len(prepared), "rejected": rejected})
            print(
                f"Profile {index}: persisted {len(prepared)} reference vectors; rejected {len(rejected)} recordings.",
                flush=True,
            )
        finally:
            for path in generated:
                path.unlink(missing_ok=True)
    if len(store.list_profiles()) >= 2:
        calibration = service.calibrate()
        print("Calibrated threshold:", round(calibration["threshold"], 4), flush=True)
    (paths.private / "migration-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("PERSISTED_PROFILES:", len(store.list_profiles()), flush=True)


if __name__ == "__main__":
    main()
