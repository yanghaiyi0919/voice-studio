"""Exercise the running app with a five-second sample from the supplied archive."""

import argparse
import json
import os
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runtime" / "validation"
OUT.mkdir(parents=True, exist_ok=True)
os.environ["TMP"] = str(OUT)
os.environ["TEMP"] = str(OUT)
tempfile.tempdir = str(OUT)

import numpy as np
import soundfile as sf
from gradio_client import Client, handle_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True, help="Your local reference WAV file.")
    parser.add_argument("--url", default=None)
    args = parser.parse_args()
    state_file = ROOT / "runtime" / "server.json"
    state = json.loads(state_file.read_text(encoding="utf-8-sig")) if state_file.exists() else {}
    url = args.url or state.get("url", "http://127.0.0.1:7860")
    with urllib.request.urlopen(url + "/config", timeout=10) as response:
        config = json.load(response)
    buttons = {
        component["props"].get("value", ""): component["id"]
        for component in config["components"]
        if component["type"] == "button"
    }

    def button_api(label_fragment):
        component_id = next(value for label, value in buttons.items() if label_fragment in label)
        dependency = next(
            dep for dep in config["dependencies"] if any(target[0] == component_id for target in dep.get("targets", []))
        )
        return "/" + dependency["api_name"]

    upload_api = "/" + next(
        dep["api_name"] for dep in config["dependencies"] if dep.get("api_name") == "on_file_upload"
    )
    source = args.audio
    with sf.SoundFile(source) as audio_file:
        audio = audio_file.read(frames=int(audio_file.samplerate * 5), dtype="float32")
        rate = audio_file.samplerate
    sample = OUT / "input_sample.wav"
    sf.write(sample, audio, rate, subtype="PCM_16")
    client = Client(url, verbose=False)
    results = []

    def case(name, fn):
        start = time.monotonic()
        details = fn()
        item = {"name": name, "passed": True, "seconds": round(time.monotonic() - start, 2), "details": details}
        results.append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)

    def upload():
        output = client.predict(handle_file(str(sample)), api_name=upload_api)
        report = str(output)
        assert "情绪" in report and "说话人" in report, report
        assert "库为空" not in report and "音频文件不存在" not in report, report
        assert isinstance(output, tuple) and output[1]["emotion"], "Emotion inference did not return a result."
        assert output[1]["speaker"]["status"] in {"recognized", "unknown", "empty"}
        return "Emotion and speaker report returned."

    def transcribe():
        text = client.predict(api_name=button_api("开始识别字幕"))
        assert isinstance(text, str) and text.strip(), str(text)
        assert "❌" not in text and "请先" not in text and "未识别到有效内容" not in text, text
        return {"characters": len(text)}

    def existing_files(value):
        if isinstance(value, str):
            return [Path(value)] if Path(value).is_file() else []
        if isinstance(value, dict):
            return [item for child in value.values() for item in existing_files(child)]
        if isinstance(value, (tuple, list)):
            return [item for child in value for item in existing_files(child)]
        return []

    def enhance(strength):
        output = client.predict(strength, api_name=button_api("执行增强处理"))
        files = existing_files(output)
        wavs = [path for path in files if path.suffix.lower() == ".wav"]
        images = [path for path in files if path.suffix.lower() == ".png"]
        assert wavs and len(images) >= 2, str(output)
        audio, _ = sf.read(wavs[0])
        assert audio.size and np.isfinite(audio).all()
        return {"audio_files": len(wavs), "spectrograms": len(images), "strength": strength}

    def transform(label):
        output = client.predict(api_name=button_api(label))
        files = existing_files(output)
        assert files, str(output)
        audio, _ = sf.read(files[0])
        assert audio.size and np.isfinite(audio).all()
        return {"samples": len(audio)}

    case("upload_and_analyze", upload)
    case("speech_to_text", transcribe)
    case("enhancement_strength_0.8", lambda: enhance(0.8))
    case("enhancement_strength_1.0", lambda: enhance(1.0))
    for label in ["海绵宝宝", "慵懒胖猫", "电子小黄人"]:
        case("voice_transform_" + label, lambda label=label: transform(label))
    (ROOT / "runtime" / "validation.json").write_text(
        json.dumps({"url": url, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("DEPLOYMENT_VALIDATION_PASSED", flush=True)


if __name__ == "__main__":
    main()
