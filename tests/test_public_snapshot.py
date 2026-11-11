"""A staged secret must still be rejected after the working copy is redacted."""

import importlib.util
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("git"), "Git is required for staged-source validation.")
class PublicSnapshotTests(unittest.TestCase):
    def test_checks_staged_bytes_instead_of_working_copy(self):
        spec = importlib.util.spec_from_file_location("public_check", ROOT / "scripts" / "check_public_files.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            file = root / "config.txt"
            file.write_text("token=" + "sk-" + "A" * 25, encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "config.txt"], check=True, capture_output=True)
            file.write_text("token=removed", encoding="utf-8")
            module.ROOT = root
            with self.assertRaises(SystemExit) as error:
                module.main()
            self.assertIn("Possible credential", str(error.exception))


if __name__ == "__main__":
    unittest.main()
