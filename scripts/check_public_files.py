"""Reject personal data, weights, archives and credentials in the Git index."""

import json
import re
import subprocess
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = {
    "private",
    "runtime",
    "logs",
    "models",
    "original-source",
    ".venv",
    ".cache",
    "dist",
    "tools",
}
SUFFIXES = {
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".ogg",
    ".db",
    ".sqlite",
    ".sqlite3",
    ".npz",
    ".npy",
    ".pt",
    ".pth",
    ".bin",
    ".mdl",
    ".fst",
    ".zip",
    ".pem",
    ".pkl",
}
SENSITIVE = re.compile(r"sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN (?:[A-Z ]* )?PRIVATE KEY-----")


def tracked_files():
    output = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout
    return [name.decode("utf-8") for name in output.split(b"\0") if name]


def tracked_bytes(name):
    return subprocess.run(["git", "show", ":" + name], cwd=ROOT, capture_output=True, check=True).stdout


def main():
    problems = []
    names = tracked_files()
    private_names = []
    local_manifest = ROOT / "private" / "legacy_import.json"
    if local_manifest.is_file():
        private_names = [
            profile["display_name"] for profile in json.loads(local_manifest.read_text(encoding="utf-8"))["profiles"]
        ]
    if not names:
        raise SystemExit("No tracked source files. Stage the intended public files before checking.")
    for name in names:
        relative = PurePosixPath(name)
        file = ROOT / name
        if relative.parts[0] in FORBIDDEN or name.startswith(("app/data/", "app/model/", "app/emotion2vec-main/")):
            problems.append(f"Private/generated directory: {name}")
        if file.suffix.lower() in SUFFIXES or file.name == ".env":
            problems.append(f"Private/generated file type: {name}")
        content = tracked_bytes(name)
        if len(content) > 2 * 1024 * 1024:
            problems.append(f"File exceeds public source limit: {name}")
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if SENSITIVE.search(text):
            problems.append(f"Possible credential: {name}")
        if any(private_name in text for private_name in private_names):
            problems.append(f"Local private display name: {name}")
        if re.search(r"[A-Z]:[\\/](?:Users[\\/]|wechat\s)", text, re.IGNORECASE):
            problems.append(f"Personal absolute path: {name}")
    if problems:
        raise SystemExit("\n".join(problems))
    print(f"PUBLIC_SOURCE_CHECK_PASSED: {len(names)} tracked files; no private assets or credential patterns found.")


if __name__ == "__main__":
    main()
