"""Build a source-only zip using the checked Git index, never directory recursion."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from check_public_files import main as check_public_files
from check_public_files import tracked_bytes, tracked_files

ROOT = Path(__file__).resolve().parents[1]


def main():
    check_public_files()
    destination = ROOT / "dist" / "voice-studio-source.zip"
    destination.parent.mkdir(exist_ok=True)
    with ZipFile(destination, "w", compression=ZIP_DEFLATED) as archive:
        for name in tracked_files():
            archive.writestr("voice-studio/" + name, tracked_bytes(name))
    print(destination)
    print("SOURCE_ZIP_BYTES:", destination.stat().st_size)


if __name__ == "__main__":
    main()
