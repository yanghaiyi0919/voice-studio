"""Registration/retention tests with a fake encoder, not model benchmarks."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from voice_studio.config import Paths
from voice_studio.presentation import analysis_card
from voice_studio.speaker_service import SpeakerService
from voice_studio.speaker_store import ProfileError, SpeakerStore


class FakeEncoder:
    model_id = "fake-encoder"

    def __init__(self):
        self.vector = np.array([1, 0, 0], dtype=np.float32)
        self.sequence = []
        self.calls = 0

    def speaker_embedding(self, path):
        self.calls += 1
        return self.sequence.pop(0) if self.sequence else self.vector.copy()


class EnrollmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.paths = Paths(self.root, self.root / "private", self.root / "runtime", {})
        self.store = SpeakerStore(self.paths.database)
        self.models = FakeEncoder()
        self.service = SpeakerService(self.store, self.models, self.paths)
        self.files = []
        time = np.arange(16000 * 3) / 16000
        for index in range(7):
            file = self.root / f"synthetic_{index}.wav"
            sf.write(file, 0.12 * np.sin(2 * np.pi * (220 + index * 11) * time), 16000)
            self.files.append(file)

    def tearDown(self):
        self.temp.cleanup()

    def test_enrollment_defaults_to_embedding_only(self):
        self.service.enroll("user_a", "示例", self.files[:3])
        self.assertEqual(self.store.get_profile("user_a")["sample_count"], 3)
        self.assertEqual(list(self.paths.retained_audio.glob("*.wav")), [])

    def test_new_store_can_identify_without_reenrollment(self):
        self.service.enroll("user_a", "示例", self.files[:3])
        reloaded = SpeakerService(SpeakerStore(self.paths.database), FakeEncoder(), self.paths)
        _, result = reloaded.identify(self.files[3])
        self.assertEqual(result["speaker_id"], "user_a")

    def test_duplicate_id_does_not_run_encoder(self):
        self.service.enroll("user_a", "示例", self.files[:3])
        calls = self.models.calls
        with self.assertRaises(ProfileError):
            self.service.enroll("user_a", "重复", self.files[3:6])
        self.assertEqual(self.models.calls, calls)

    def test_retained_audio_is_removed_on_profile_delete(self):
        self.service.enroll("user_a", "示例", self.files[:3], retain_audio=True)
        self.assertEqual(len(list(self.paths.retained_audio.glob("*.wav"))), 3)
        self.service.delete("user_a", 1)
        self.assertEqual(list(self.paths.retained_audio.glob("*.wav")), [])
        self.assertTrue(all(path.is_file() for path in self.files))

    def test_failed_enrollment_does_not_leave_audio_or_profile(self):
        self.models.sequence = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 1, 0])]
        with self.assertRaises(ProfileError):
            self.service.enroll("mixed", "混合", self.files[:3], retain_audio=True)
        self.assertEqual(self.store.list_profiles(), [])
        self.assertEqual(list(self.paths.retained_audio.glob("*.wav")), [])

    def test_duplicate_recordings_roll_back_retained_files(self):
        with self.assertRaises(ProfileError):
            self.service.enroll("user_a", "示例", [self.files[0]] * 3, retain_audio=True)
        self.assertEqual(self.store.list_profiles(), [])
        self.assertEqual(list(self.paths.retained_audio.glob("*.wav")), [])

    def test_append_changes_reference_count(self):
        self.service.enroll("user_a", "示例", self.files[:3])
        self.service.update("user_a", [self.files[3]], 1)
        self.assertEqual(self.store.get_profile("user_a")["sample_count"], 4)

    def test_appending_other_voice_is_rejected(self):
        self.service.enroll("user_a", "示例", self.files[:3])
        self.models.vector = np.array([0, 1, 0], dtype=np.float32)
        with self.assertRaises(ProfileError):
            self.service.update("user_a", [self.files[3]], 1)
        self.assertEqual(self.store.get_profile("user_a")["sample_count"], 3)

    def test_unknown_returns_no_identity(self):
        self.service.enroll("user_a", "示例", self.files[:3])
        self.models.vector = np.array([0, 1, 0], dtype=np.float32)
        _, result = self.service.identify(self.files[3])
        self.assertEqual(result["status"], "unknown")
        self.assertIsNone(result["speaker_id"])

    def test_quiet_input_can_still_be_enhanced_without_identity_guess(self):
        self.service.enroll("user_a", "示例", self.files[:3])
        quiet = self.root / "quiet.wav"
        sf.write(quiet, np.sin(np.arange(48000) / 20) * 0.0001, 16000)
        calls = self.models.calls
        normalized, result = self.service.identify(quiet)
        self.assertEqual(result["status"], "invalid_audio")
        self.assertTrue(Path(normalized).is_file())
        self.assertEqual(self.models.calls, calls)

    def test_cleanup_cannot_remove_files_outside_owned_audio_directory(self):
        warnings = self.service._purge([self.files[0]])
        self.assertTrue(warnings)
        self.assertTrue(self.files[0].is_file())

    def test_display_name_is_html_escaped(self):
        html = analysis_card({"status": "recognized", "display_name": "<script>alert(1)</script>", "reason": ""})
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)


if __name__ == "__main__":
    unittest.main()
