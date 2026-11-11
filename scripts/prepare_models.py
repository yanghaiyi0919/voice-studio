"""Download the original app's two models and verify upstream SHA-256 hashes."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "emotion": ("iic/emotion2vec_plus_large", ["configuration.json", "config.yaml", "tokens.txt", "model.pt"]),
    "speaker": (
        "iic/speech_campplus_sv_zh-cn_16k-common",
        ["configuration.json", "config.yaml", "campplus_cn_common.bin"],
    ),
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fetch_file(model_id: str, info: dict, target: Path) -> None:
    expected_size = info["Size"]
    expected_hash = info["Sha256"]
    if target.is_file() and target.stat().st_size == expected_size and digest(target) == expected_hash:
        print(f"Verified cached file: {target.name}", flush=True)
        return
    partial = target.with_name(target.name + ".part")
    query = urllib.parse.urlencode({"Revision": info["Revision"], "FilePath": info["Path"]})
    url = f"https://modelscope.cn/api/v1/models/{model_id}/repo?{query}"
    for attempt in range(1, 6):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == expected_size:
            if digest(partial) == expected_hash:
                partial.replace(target)
                return
            partial.unlink()
            offset = 0
        headers = {"User-Agent": "VoiceDetection-LocalDeployment/1.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
                if offset and response.status != 206:
                    offset = 0
                mode = "ab" if offset else "wb"
                started = time.monotonic()
                last_report = started
                transferred = 0
                with partial.open(mode) as stream:
                    while block := response.read(1024 * 1024):
                        stream.write(block)
                        offset += len(block)
                        transferred += len(block)
                        now = time.monotonic()
                        if now - last_report >= 10:
                            rate = transferred / max(now - started, 0.001) / 1024**2
                            print(
                                f"{target.name}: {offset / 1024**2:.0f}/{expected_size / 1024**2:.0f} MiB, {rate:.1f} MiB/s",
                                flush=True,
                            )
                            last_report = now
            if partial.stat().st_size != expected_size:
                raise RuntimeError(f"Incomplete download for {target.name}")
            if digest(partial) != expected_hash:
                partial.unlink()
                raise RuntimeError(f"Checksum mismatch for {target.name}")
            partial.replace(target)
            print(f"Downloaded and verified: {target.name}", flush=True)
            return
        except Exception as error:
            print(f"Download attempt {attempt}/5 failed: {error}", flush=True)
            if attempt == 5:
                raise
            time.sleep(min(2 * attempt, 8))


def prepare_vosk() -> str:
    name = "vosk-model-small-cn-0.22"
    destination = ROOT / "models" / name
    legacy = ROOT / "app" / "model"
    for candidate in [destination, legacy]:
        if (candidate / "am" / "final.mdl").is_file() and (candidate / "conf" / "model.conf").is_file():
            return str(candidate)
    archive = ROOT / ".cache" / (name + ".zip")
    archive.parent.mkdir(parents=True, exist_ok=True)
    partial = archive.with_suffix(".zip.part")
    if not archive.exists():
        url = f"https://alphacephei.com/vosk/models/{name}.zip"
        for attempt in range(5):
            try:
                offset = partial.stat().st_size if partial.exists() else 0
                headers = {"Range": f"bytes={offset}-"} if offset else {}
                with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
                    if offset and response.status != 206:
                        offset = 0
                    with partial.open("ab" if offset else "wb") as stream:
                        for block in iter(lambda: response.read(1024 * 1024), b""):
                            stream.write(block)
                with ZipFile(partial) as check:
                    if check.testzip():
                        raise RuntimeError("Vosk archive CRC check failed")
                partial.replace(archive)
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2)
    with ZipFile(archive) as source:
        if sum(entry.file_size for entry in source.infolist()) > 300 * 1024 * 1024:
            raise RuntimeError("Unexpected Vosk archive size")
        for entry in source.infolist():
            parts = Path(entry.filename.replace("\\", "/")).parts
            if not parts or parts[0] != name or ".." in parts or any(":" in part for part in parts):
                raise RuntimeError("Unsafe Vosk archive path")
            if stat.S_ISLNK(entry.external_attr >> 16):
                raise RuntimeError("Vosk archive contains an unexpected symbolic link")
            target = (ROOT / "models").joinpath(*parts).resolve()
            if not target.is_relative_to((ROOT / "models").resolve()):
                raise RuntimeError("Vosk archive path escaped model directory")
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read(entry))
    (destination / "download-manifest.json").write_text(
        json.dumps({"source": f"https://alphacephei.com/vosk/models/{name}.zip", "sha256": digest(archive)}, indent=2),
        encoding="utf-8",
    )
    return str(destination)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    state = {}
    for kind, (model_id, needed) in MODELS.items():
        target_dir = ROOT / "models" / model_id.split("/")[-1]
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"Preparing {model_id}", flush=True)
        url = f"https://modelscope.cn/api/v1/models/{model_id}/repo/files?Revision=master&Recursive=true"
        with urllib.request.urlopen(url, timeout=60) as response:
            manifest = json.load(response)
        files = {info["Path"]: info for info in manifest["Data"]["Files"] if info["Type"] == "blob"}
        chosen = {name: files[name] for name in needed}
        (target_dir / "download-manifest.json").write_text(
            json.dumps({"model_id": model_id, "files": chosen}, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        for name in needed:
            fetch_file(model_id, chosen[name], target_dir / name)
        state[kind] = str(target_dir)
    state["asr"] = prepare_vosk()
    import imageio_ffmpeg

    tools = ROOT / "tools"
    tools.mkdir(exist_ok=True)
    ffmpeg = tools / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    if not ffmpeg.exists():
        shutil.copy2(imageio_ffmpeg.get_ffmpeg_exe(), ffmpeg)
    runtime = ROOT / "runtime"
    runtime.mkdir(exist_ok=True)
    (runtime / "models.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    print("All model files are ready.", flush=True)


if __name__ == "__main__":
    main()
