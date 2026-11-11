"""Run the local Voice Studio interface."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("MODELSCOPE_CACHE", str(ROOT / ".cache" / "modelscope"))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("GRADIO_TEMP_DIR", str(ROOT / "runtime" / "gradio"))
os.environ.setdefault("TORCHAUDIO_USE_CODEC", "0")
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))
os.environ["PATH"] = str(ROOT / "tools") + os.pathsep + os.environ.get("PATH", "")

from voice_studio.ui import main

if __name__ == "__main__":
    main()
