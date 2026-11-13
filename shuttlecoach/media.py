from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import parse_qs, urlparse

import av
import imageio_ffmpeg
import numpy as np

from .paths import DATA


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def youtube_id(url):
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or parsed.username or parsed.password:
        raise ValueError("Use an HTTPS YouTube video URL.")
    if host == "youtu.be":
        video_id = parsed.path.strip("/")
    elif host in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
        video_id = parse_qs(parsed.query).get("v", [""])[0]
    else:
        raise ValueError("Only youtube.com and youtu.be video URLs are accepted.")
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise ValueError("The URL must identify one YouTube video.")
    return video_id


def download(url):
    """Normal public download only; no credentials or access-control workarounds."""
    video_id = youtube_id(url)
    canonical = f"https://www.youtube.com/watch?v={video_id}"
    import yt_dlp
    options = {
        "noplaylist": True, "outtmpl": str(DATA / f"{video_id}.%(ext)s"),
        "format": "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[height<=720]",
        "merge_output_format": "mp4", "ffmpeg_location": imageio_ffmpeg.get_ffmpeg_exe(),
        "writeinfojson": True, "overwrites": False, "socket_timeout": 20, "retries": 2,
        "noprogress": True,
    }
    with yt_dlp.YoutubeDL(options) as downloader:
        info = downloader.extract_info(canonical, download=True)
        candidate = Path(downloader.prepare_filename(info))
    merged = candidate.with_suffix(".mp4")
    path = merged if merged.is_file() else candidate
    if not path.is_file():
        raise ValueError("Download did not produce a video. Use a local MP4/MOV.")
    return path.resolve()


def probe(path):
    """Decode presentation timestamps; never derive a VFR timeline from average FPS."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Video does not exist: {path}")
    from .paths import ROOT
    stat = path.stat()
    cache = ROOT / ".cache" / ("timeline-" + hashlib.sha256(str(path).encode()).hexdigest()[:20] + ".json")
    if cache.is_file():
        try:
            saved = json.loads(cache.read_text())
            if saved["size"] == stat.st_size and saved["mtime_ns"] == stat.st_mtime_ns:
                return saved["metadata"]
        except (ValueError, KeyError):
            pass
    times, durations = [], []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        width, height = stream.width, stream.height
        rate = stream.average_rate or Fraction(30, 1)
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError("Video has missing presentation timestamps; remux it before analysis.")
            times.append(float(frame.pts * frame.time_base))
            durations.append(float(frame.duration * frame.time_base) if frame.duration else 0)
    if len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Video needs at least two frames with increasing timestamps.")
    origin = times[0]
    times = np.asarray(times) - origin
    final_duration = durations[-1] or float(np.median(np.diff(times)))
    metadata = {"path": str(path), "sha256": sha256(path), "width": width, "height": height,
            "fps": float(rate), "rate": str(rate), "frame_count": len(times),
            "duration_s": float(times[-1] + final_duration), "pts_origin_s": origin,
            "times": times.tolist(), "last_frame_duration_s": final_duration}
    write_json(cache, {"size":stat.st_size,"mtime_ns":stat.st_mtime_ns,"metadata":metadata})
    return metadata


def read_frame(path, index, times=None, origin=0):
    # Sequential decoding makes manual reference frames agree with the PTS table.
    with av.open(str(path)) as container:
        if times is not None:
            target = times[index] + origin
            container.seek(max(0, int(target * av.time_base)), backward=True)
            for frame in container.decode(video=0):
                if frame.pts is not None and abs(float(frame.pts * frame.time_base) - target) < 0.0001:
                    return frame.to_ndarray(format="rgb24")
            raise ValueError("Could not recover the exact source frame after seeking.")
        for i, frame in enumerate(container.decode(video=0)):
            if i == index:
                return frame.to_ndarray(format="rgb24")
    raise ValueError("Frame is outside the video.")


def verify_overlay(path, expected_times):
    with av.open(str(path)) as container:
        actual = [float(frame.pts * frame.time_base) for frame in container.decode(video=0)]
    expected = np.asarray(expected_times) - expected_times[0]
    if len(actual) != len(expected):
        raise ValueError("Overlay frame count differs from source interval.")
    error = float(np.max(np.abs(np.asarray(actual) - expected)))
    tolerance = min(float(np.min(np.diff(expected))) / 2, 0.002)
    if error > tolerance:
        raise ValueError(f"Overlay timestamp error {error:.6f}s exceeds {tolerance:.6f}s.")
    return {"frames": len(actual), "max_timestamp_error_s": error, "tolerance_s": tolerance}


def verify_tracker_frames(meta,start,end,stride=1):
    """Check upstream OpenCV frame seeking against the independent review decoder.

    OpenCV's frame seeking can be inaccurate on some variable-rate codecs. Reject
    those inputs instead of attaching correct-looking times to the wrong images.
    """
    import cv2
    capture=cv2.VideoCapture(meta["path"])
    checks=[]
    try:
        if not capture.isOpened() or not capture.set(cv2.CAP_PROP_POS_FRAMES,start):
            raise ValueError("Tracker could not seek to the requested source frame.")
        for index in range(start,min(start+3*stride,end),stride):
            ok,image=capture.read()
            if not ok:raise ValueError("Tracker could not decode the start of the selected interval.")
            expected=read_frame(meta["path"],index,meta["times"],meta["pts_origin_s"])
            error=float(np.mean(np.abs(image.astype(float)-expected[:,:,::-1].astype(float))))
            time_error=abs(capture.get(cv2.CAP_PROP_POS_MSEC)/1000-meta["times"][index])
            interval=meta["times"][index+1]-meta["times"][index] if index+1<len(meta["times"]) else meta["last_frame_duration_s"]
            # Different FFmpeg color-conversion versions can differ by ~2/255
            # without changing frame identity. PTS agreement remains strict.
            if error>4.0 or time_error>max(0.002,interval/2):
                raise ValueError("Tracker and review decoders disagree on source-frame alignment. Use a constant-frame-rate copy and its own timeline; do not reuse these timestamps.")
            checks.append({"source_frame":index,"mean_pixel_difference":error,"timestamp_error_s":time_error})
            for _ in range(stride-1):
                capture.grab()
    finally:
        capture.release()
    return {"status":"passed","probes":checks,"limitation":"Checks initial sequential decoding; overlay timestamps are independently checked across every output frame."}


def write_json(path, value):
    import tempfile
    path = Path(path)
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(mode="w",dir=path.parent,suffix=".tmp",delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(text)
    temporary.replace(path)
