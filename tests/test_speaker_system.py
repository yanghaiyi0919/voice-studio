"""Model-free regression tests; no personal recordings or weights are required."""

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from voice_studio.audio import AudioError, check_waveform, prepare_audio
from voice_studio.recognition import calibrate, identify
from voice_studio.speaker_store import ProfileError, RevisionConflict, Sample, SpeakerStore, centroid, unit_vector

MODEL = "test-model-v1"


def samples(prefix, vector, count=3):
    result = []
    for index in range(count):
        delta = np.zeros(len(vector), dtype=np.float32)
        delta[-1] = (index - 1) * 0.015
        result.append(
            Sample(
                unit_vector(np.asarray(vector) + delta),
                hashlib.sha256(f"{prefix}-{index}".encode()).hexdigest(),
                {"duration": 5.0},
            )
        )
    return result


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.database = Path(self.temp.name) / "profiles.sqlite3"
        self.store = SpeakerStore(self.database)
        self.store.enroll("alice", "示例用户 A", samples("a", [1, 0, 0]), MODEL)

    def tearDown(self):
        self.temp.cleanup()

    def test_persists_across_store_instances(self):
        reloaded = SpeakerStore(self.database)
        self.assertEqual(reloaded.get_profile("alice")["sample_count"], 3)
        self.assertEqual(len(reloaded.snapshot()[0][0]["samples"]), 3)

    def test_duplicate_id_is_rejected(self):
        with self.assertRaises(ProfileError):
            self.store.enroll("ALICE", "另一个名字", samples("other", [0, 1, 0]), MODEL)

    def test_duplicate_recording_rolls_back_whole_enrollment(self):
        with self.assertRaises(ProfileError):
            self.store.enroll("bob", "示例用户 B", samples("a", [0, 1, 0]), MODEL)
        self.assertIsNone(self.store.get_profile("bob"))
        self.assertEqual(len(self.store.snapshot()[0]), 1)

    def test_duplicate_inside_batch_is_rejected(self):
        values = samples("new", [0, 1, 0])
        values[2].sha256 = values[0].sha256
        with self.assertRaises(ProfileError):
            self.store.enroll("bob", "示例用户 B", values, MODEL)

    def test_requires_three_recordings(self):
        with self.assertRaises(ProfileError):
            self.store.enroll("bob", "示例用户 B", samples("b", [0, 1, 0], 2), MODEL)

    def test_invalid_id_is_rejected(self):
        with self.assertRaises(ProfileError):
            self.store.enroll("../escape", "测试", samples("b", [0, 1, 0]), MODEL)

    def test_rename_persists_and_changes_revision(self):
        self.store.rename("alice", "新显示名称", 1)
        self.assertEqual(self.store.get_profile("alice")["display_name"], "新显示名称")
        self.assertEqual(self.store.get_profile("alice")["revision"], 2)

    def test_stale_update_is_rejected_without_changes(self):
        self.store.rename("alice", "新显示名称", 1)
        with self.assertRaises(RevisionConflict):
            self.store.update_samples("alice", samples("append", [1, 0, 0], 1), MODEL, 1)
        self.assertEqual(self.store.get_profile("alice")["sample_count"], 3)

    def test_append_and_replace(self):
        self.store.update_samples("alice", samples("append", [1, 0, 0], 1), MODEL, 1)
        self.assertEqual(self.store.get_profile("alice")["sample_count"], 4)
        self.store.update_samples("alice", samples("replace", [0, 1, 0]), "model-v2", 2, replace=True)
        profile = self.store.get_profile("alice")
        self.assertEqual((profile["sample_count"], profile["model_id"]), (3, "model-v2"))

    def test_append_wrong_model_is_rejected(self):
        with self.assertRaises(ProfileError):
            self.store.update_samples("alice", samples("append", [1, 0, 0], 1), "model-v2", 1)

    def test_delete_cascades_samples(self):
        self.store.delete("alice", 1)
        with self.store.connection() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM samples").fetchone()[0], 0)
        self.assertEqual(self.store.list_profiles(), [])

    def test_stale_delete_does_not_delete(self):
        with self.assertRaises(RevisionConflict):
            self.store.delete("alice", 99)
        self.assertIsNotNone(self.store.get_profile("alice"))

    def test_thresholds_persist_and_reject_invalid_values(self):
        self.store.save_settings(0.7, 0.08)
        self.assertEqual(SpeakerStore(self.database).get_settings()["threshold"], 0.7)
        for value in [-0.1, 1.1, float("nan")]:
            with self.assertRaises(ProfileError):
                self.store.save_settings(value, 0.05)

    def test_calibration_revision_guard(self):
        _, fingerprint = self.store.snapshot()
        self.store.rename("alice", "新显示名称", 1)
        with self.assertRaises(RevisionConflict):
            self.store.save_settings(0.6, 0.05, {}, expected_fingerprint=fingerprint)


class RecognitionTests(unittest.TestCase):
    def profile(self, user, vector):
        values = samples(user, vector)
        return {
            "speaker_id": user,
            "display_name": user,
            "model_id": MODEL,
            "centroid": centroid([s.embedding for s in values]),
            "samples": [s.embedding for s in values],
        }

    def test_known_speaker(self):
        result = identify([1, 0, 0], [self.profile("a", [1, 0, 0]), self.profile("b", [0, 1, 0])], MODEL, 0.6, 0.05)
        self.assertEqual(result["speaker_id"], "a")

    def test_unknown_is_rejected(self):
        result = identify([0, 0, 1], [self.profile("a", [1, 0, 0])], MODEL, 0.6, 0.05)
        self.assertEqual((result["status"], result["speaker_id"]), ("unknown", None))

    def test_close_candidates_are_rejected(self):
        result = identify([1, 0, 0], [self.profile("a", [1, 0, 0]), self.profile("b", [1, 0.01, 0])], MODEL, 0.6, 0.05)
        self.assertEqual(result["status"], "unknown")

    def test_single_profile_still_requires_absolute_threshold(self):
        result = identify([0, 1, 0], [self.profile("a", [1, 0, 0])], MODEL, 0.6, 0.05)
        self.assertEqual(result["status"], "unknown")

    def test_empty_registry(self):
        self.assertEqual(identify([1, 0, 0], [], MODEL, 0.6, 0.05)["status"], "empty")

    def test_model_mismatch(self):
        self.assertEqual(
            identify([1, 0, 0], [self.profile("a", [1, 0, 0])], "different-model", 0.6, 0.05)["status"],
            "model_mismatch",
        )

    def test_invalid_vectors(self):
        for value in [[0, 0], [float("nan"), 1], [float("inf"), 1], [1e30, 1e30]]:
            with self.assertRaises(ProfileError):
                unit_vector(value)

    def test_calibration_uses_held_out_recordings(self):
        report = calibrate([self.profile("a", [1, 0, 0]), self.profile("b", [0, 1, 0])], MODEL, "fingerprint")
        self.assertEqual(report["genuine_trials"], 6)
        self.assertEqual(report["estimated_far"], 0)
        self.assertGreater(report["threshold"], 0.1)
        self.assertLess(report["threshold"], 0.9)
        self.assertEqual(report["fingerprint"], "fingerprint")

    def test_calibration_needs_multiple_speakers(self):
        with self.assertRaises(ProfileError):
            calibrate([self.profile("a", [1, 0, 0])], MODEL, "fp")


class AudioTests(unittest.TestCase):
    def wave(self, seconds=3):
        time = np.arange(int(16000 * seconds)) / 16000
        return (0.12 * np.sin(2 * np.pi * 220 * time)).astype(np.float32)

    def test_silence_is_rejected(self):
        with self.assertRaises(AudioError):
            check_waveform(np.zeros(48000), enrollment=True)

    def test_short_audio_is_rejected(self):
        with self.assertRaises(AudioError):
            check_waveform(self.wave(1), enrollment=True)

    def test_clipped_audio_is_rejected(self):
        with self.assertRaises(AudioError):
            check_waveform(np.ones(48000), enrollment=True)

    def test_valid_waveform_metrics(self):
        quality = check_waveform(self.wave(), enrollment=True)
        self.assertEqual(quality.duration, 3)
        self.assertGreater(quality.active_duration, 2)

    def test_normalization_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.wav"
            sf.write(source, self.wave(), 16000, subtype="PCM_16")
            first = prepare_audio(source, Path(directory) / "first", enrollment=True)
            second = prepare_audio(first.path, Path(directory) / "second", enrollment=True)
            self.assertEqual(first.sha256, second.sha256)

    def test_resamples_stereo(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "stereo.wav"
            time = np.arange(44100 * 3) / 44100
            wave = 0.1 * np.sin(2 * np.pi * 220 * time)
            sf.write(source, np.stack([wave, wave], axis=1), 44100)
            prepared = prepare_audio(source, Path(directory) / "out", enrollment=True)
            audio, rate = sf.read(prepared.path)
            self.assertEqual((rate, audio.ndim, len(audio)), (16000, 1, 48000))


if __name__ == "__main__":
    unittest.main()
