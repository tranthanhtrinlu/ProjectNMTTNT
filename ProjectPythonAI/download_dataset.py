# ai-service/download_dataset.py
import os
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "datasets" / "CIFAKE"
ZIP_PATH = ROOT / "cifake.zip"


def run(cmd: str):
    print(f"$ {cmd}")
    code = os.system(cmd)
    if code != 0:
        raise SystemExit(f"Command failed: {cmd}")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not ZIP_PATH.exists():
        run("kaggle datasets download -d nadirphy/cifake-dataset -p . -f data.zip -q")
        # Kaggle file name is data.zip
        if Path("data.zip").exists():
            Path("data.zip").rename(ZIP_PATH)
    print("Extracting...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        zf.extractall(ROOT / "datasets")
    print(f"Done. Check: {DATA_DIR}")


if __name__ == "__main__":
    main()
