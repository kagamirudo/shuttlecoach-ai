from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
VENDOR = ROOT / "vendor" / "courtside"
for directory in (DATA, RESULTS, MODELS, ROOT / ".cache"):
    directory.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache" / "ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
