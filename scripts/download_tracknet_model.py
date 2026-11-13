"""Download the official public TrackNetV3 checkpoint; verify before activation."""
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen
import zipfile

ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/"models"/"TrackNet_best.pt"
FILE_ID="1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA"
URL=f"https://drive.google.com/file/d/{FILE_ID}/view"
DIGEST="df867641a02712b021f04548ff4b1208ddfdb47f629ab2094ceb978667e83b1a"


class DownloadForm(HTMLParser):
    def __init__(self):
        super().__init__();self.inputs={}

    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if tag=="input" and attrs.get("type")=="hidden" and attrs.get("name") in {"id","export","confirm","uuid"}:
            self.inputs[attrs["name"]]=attrs.get("value","")


if not MODEL.exists():
    archive=ROOT/".cache"/"TrackNetV3_ckpts.zip";archive.parent.mkdir(exist_ok=True)
    with urlopen(f"https://drive.google.com/uc?export=download&id={FILE_ID}",timeout=60) as response:
        if "text/html" in response.headers.get("Content-Type",""):
            form=DownloadForm();form.feed(response.read().decode())
            if form.inputs.get("id")!=FILE_ID or not form.inputs.get("confirm"):
                raise ValueError("Official checkpoint download unavailable; no credentials or access workaround is attempted.")
            download="https://drive.usercontent.google.com/download?"+urlencode(form.inputs)
        else:download=response.url
    partial=archive.with_suffix(".part")
    with urlopen(download,timeout=120) as response,partial.open("wb") as output:
        while chunk:=response.read(1024*1024):output.write(chunk)
    partial.replace(archive)
    with zipfile.ZipFile(archive) as bundle:data=bundle.read("ckpts/TrackNet_best.pt")
    if hashlib.sha256(data).hexdigest()!=DIGEST:raise ValueError("Checkpoint checksum differs; model was not activated.")
    MODEL.parent.mkdir(exist_ok=True);MODEL.write_bytes(data)
if hashlib.sha256(MODEL.read_bytes()).hexdigest()!=DIGEST:raise ValueError("Existing TrackNet checkpoint checksum differs.")
(MODEL.parent/"tracknet-manifest.json").write_text(json.dumps({"filename":MODEL.name,"url":URL,"sha256":DIGEST,"license":"MIT",
    "repository":"https://github.com/qaz812345/TrackNetV3","revision":"6eda442ada1740573f200f836d93edc9a541ee86"},indent=2)+"\n")
print(f"{MODEL}\nsha256={DIGEST}")
