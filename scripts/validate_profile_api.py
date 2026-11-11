"""Optional UI API smoke test with a disposable profile and local reference audio."""

import argparse
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

import soundfile as sf
from gradio_client import Client, handle_file
from voice_studio.config import Paths
from voice_studio.models import ModelHub
from voice_studio.speaker_service import SpeakerService
from voice_studio.speaker_store import SpeakerStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:7860")
    args = parser.parse_args()
    paths = Paths.load(ROOT)
    store = SpeakerStore(paths.database)
    helper = SpeakerService(store, ModelHub(paths), paths)
    before = len(store.list_profiles())
    user = "ui_smoke_" + uuid.uuid4().hex[:12]
    queue_user = user + "_queue"
    work = paths.runtime / "profile-api-validation"
    work.mkdir(exist_ok=True)
    audio_files = []
    for index, source in enumerate(sorted(args.audio_dir.glob("*.wav"))[:4]):
        with sf.SoundFile(source) as stream:
            stream.seek(int(0.2 * stream.samplerate))
            audio = stream.read(int(5 * stream.samplerate))
            target = work / f"clip_{index}.wav"
            sf.write(target, audio, stream.samplerate)
            audio_files.append(target)
    assert len(audio_files) == 4, "Four local reference recordings are required."
    client = Client(args.url, verbose=False)
    tests = []

    def passed(name):
        tests.append(name)
        print(name + ": PASS", flush=True)

    def save(mode, files, selected=None):
        return client.predict(
            mode,
            user if mode == "登记新用户" else "",
            "临时功能验证档案" if mode == "登记新用户" else "",
            selected,
            [handle_file(str(file)) for file in files],
            True,
            api_name="/save_speaker_profile",
        )

    try:
        save("登记新用户", audio_files[:3])
        assert store.get_profile(user)["sample_count"] == 3
        passed("ui_register_new_user")
        try:
            save("登记新用户", audio_files[:3])
        except Exception as error:
            assert "已存在" in str(error), str(error)
        else:
            raise AssertionError("Duplicate ID was accepted")
        passed("ui_reject_duplicate_id")
        client.predict(user, api_name="/select_speaker")
        client.predict(user, "临时档案已改名", api_name="/rename_speaker")
        assert store.get_profile(user)["display_name"] == "临时档案已改名"
        passed("ui_rename")
        save("追加参考录音", [audio_files[3]], user)
        assert store.get_profile(user)["sample_count"] == 4
        passed("ui_append_references")
        save("替换全部参考录音", audio_files[:3], user)
        assert store.get_profile(user)["sample_count"] == 3
        passed("ui_replace_references")
        try:
            client.predict(user, False, api_name="/delete_speaker")
        except Exception as error:
            assert "勾选" in str(error), str(error)
        else:
            raise AssertionError("Unconfirmed deletion was accepted")
        passed("ui_requires_delete_confirmation")
        client.predict(user, True, api_name="/delete_speaker")
        assert store.get_profile(user) is None
        passed("ui_delete_profile")
        assert len(store.list_profiles()) == before
        passed("original_profiles_preserved")
        for file in audio_files[:3]:
            client.predict(handle_file(str(file)), api_name="/add_reference_recording")
        passed("ui_collect_multiple_recordings")
        try:
            client.predict(handle_file(str(audio_files[0])), api_name="/add_reference_recording")
        except Exception as error:
            assert "已经加入" in str(error), str(error)
        else:
            raise AssertionError("Duplicate pending recording was accepted")
        passed("ui_reject_duplicate_pending_recording")
        client.predict("登记新用户", queue_user, "逐段录音验证", None, None, False, api_name="/save_speaker_profile")
        assert store.get_profile(queue_user)["sample_count"] == 3
        passed("ui_register_from_recording_queue")
        client.predict(queue_user, True, api_name="/delete_speaker")
        client.predict(api_name="/clear_reference_recordings")
        passed("ui_clear_queue_and_remove_disposable_profile")
        assert len(store.list_profiles()) == before
    finally:
        profile = store.get_profile(user)
        if profile:
            helper.delete(user, profile["revision"])
        profile = store.get_profile(queue_user)
        if profile:
            helper.delete(queue_user, profile["revision"])
    (paths.runtime / "profile-api-validation.json").write_text(json.dumps({"tests": tests}, indent=2), encoding="utf-8")
    print("PROFILE_API_VALIDATION_PASSED", flush=True)


if __name__ == "__main__":
    main()
