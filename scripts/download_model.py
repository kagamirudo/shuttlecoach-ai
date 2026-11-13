"""Fetch a specific Ultralytics release asset; keep weights out of source control."""
import hashlib
import json
from pathlib import Path
import urllib.request

root = Path(__file__).resolve().parents[1]
destination = root / "models" / "yolo11m-pose.pt"
url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-pose.pt"
expected_sha256 = "29b17eaf3a3117cbea906090dbedf9159f7c6a49db58ec8b99ed2dfde1cf6eb2"
destination.parent.mkdir(exist_ok=True)
if not destination.exists():
    partial = destination.with_suffix(".part")
    urllib.request.urlretrieve(url, partial)
    if hashlib.sha256(partial.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("Model checksum differs from the verified release asset; download was not activated.")
    partial.replace(destination)
digest = hashlib.sha256(destination.read_bytes()).hexdigest()
if digest != expected_sha256:
    raise ValueError("Existing model checksum is incorrect.")
(destination.parent / "pose-manifest.json").write_text(json.dumps({
    "filename":destination.name, "url":url, "sha256":digest,
    "license":"AGPL-3.0; see https://github.com/ultralytics/ultralytics/blob/main/LICENSE",
}, indent=2)+"\n")
print(f"{destination}\nsha256={digest}")
