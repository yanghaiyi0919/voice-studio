"""Exercise the recording adapter with a fake device; never capture real audio."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from voice_studio.audio import AudioError, capture_local_microphone


class LocalRecordingTests(unittest.TestCase):
    def fake_device(self, unsupported=False):
        check = Mock(side_effect=ValueError("unsupported") if unsupported else None)
        return SimpleNamespace(
            check_input_settings=check,
            PortAudioError=ValueError,
            query_devices=Mock(return_value={"default_samplerate": 48000}),
            rec=Mock(side_effect=lambda frames, **kwargs: np.zeros((frames, 1), dtype=np.float32)),
            wait=Mock(),
        )

    def test_records_only_on_explicit_call(self):
        device = self.fake_device()
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"sounddevice": device}):
            file = capture_local_microphone(directory)
            audio, rate = sf.read(file)
            self.assertEqual((rate, len(audio)), (16000, 80000))
            device.rec.assert_called_once()
            device.wait.assert_called_once()

    def test_falls_back_to_device_sample_rate(self):
        device = self.fake_device(unsupported=True)
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"sounddevice": device}):
            file = capture_local_microphone(directory)
            audio, rate = sf.read(file)
            self.assertEqual((rate, len(audio)), (48000, 240000))

    def test_refuses_unbounded_recording(self):
        with self.assertRaises(AudioError):
            capture_local_microphone(".", seconds=100)

    def test_device_failure_returns_actionable_error(self):
        device = self.fake_device()
        device.rec.side_effect = RuntimeError("device unavailable")
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"sounddevice": device}):
            with self.assertRaises(AudioError) as error:
                capture_local_microphone(directory)
            self.assertIn("本机麦克风暂不可用", str(error.exception))


if __name__ == "__main__":
    unittest.main()
