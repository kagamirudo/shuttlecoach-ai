from datetime import datetime, timezone
from fractions import Fraction
import importlib.metadata
import json
from pathlib import Path
import time

import av
import cv2
import numpy as np
import pandas as pd

from .analysis import build_observations, summarize, validate_config, track_reason
from .media import probe, sha256, verify_overlay, verify_tracker_frames, write_json
from .paths import ROOT, RESULTS, VENDOR
from .upstream import bind_run, court, db, detect, draw_skeleton, viz
from .validation import evaluate, sample_template, screen_coverage


def choose_device(requested="auto"):
    import torch
    if requested not in {"auto", "cpu", "mps"}:
        raise ValueError("Choose auto, cpu or mps.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable in this environment. Choose CPU or run in your Mac terminal.")
    return ("mps" if torch.backends.mps.is_available() else "cpu") if requested == "auto" else requested


def raw_tracks(database):
    con = db.connect(database, read_only=True)
    result = con.execute("SELECT * FROM tracks ORDER BY frame_num,player_id").df()
    con.close()
    return result.to_dict("records")


def overlay(video, destination, raw, meta, start, end, homography, stride=1, estimates=()):
    """Reuse upstream drawing; encode using source PTS, including VFR input."""
    indexed = {}
    forecast={}
    for row in estimates:forecast.setdefault(int(row['source_frame']),[]).append(row)
    for row in raw:
        reason=track_reason(row)
        if reason=='accepted' or (reason=='outside_court' and row.get('identity_verified',False)):
            indexed.setdefault(int(row["frame_num"]), []).append(row)
    base = meta["times"][start]
    time_base = Fraction(1, 90000)
    mini_height=min(meta["height"]-48,viz.court_size()[1])
    mini_width=round(viz.court_size()[0]*mini_height/viz.court_size()[1])
    canvas_width=meta["width"]+mini_width+24
    from .timeline import source_seconds,source_frame_number,crosses_cut
    trails={"near":[],"far":[]};previous_frame=start-1
    with av.open(str(video)) as source, av.open(str(destination), "w") as output:
        stream = output.add_stream("libx264", rate=(Fraction(meta["rate"])/stride).limit_denominator(1001))
        # Odd source dimensions need padding for yuv420p.
        stream.width, stream.height = canvas_width+canvas_width%2, meta["height"]+meta["height"]%2
        stream.pix_fmt = "yuv420p"
        stream.time_base = time_base
        stream.codec_context.time_base = time_base
        stream.options = {"crf": "23", "preset": "veryfast", "bf": "0"}
        for index, source_frame in enumerate(source.decode(video=0)):
            if index < start:
                continue
            if index >= end:
                break
            if (index-start)%stride:
                continue
            if crosses_cut(meta,previous_frame,index):
                for trail in trails.values():trail.clear()
            previous_frame=index
            frame = source_frame.to_ndarray(format="bgr24")
            frame=cv2.copyMakeBorder(frame,0,0,0,mini_width+24,cv2.BORDER_CONSTANT,value=(20,26,22))
            # Reproject known court lines into the source view for calibration QA.
            inverse = np.linalg.inv(homography)
            for a, b in viz._segments():
                points = court.image_to_court(np.array([a, b]), inverse).astype(int)
                cv2.line(frame, tuple(points[0]), tuple(points[1]), (100, 210, 100), 1, cv2.LINE_AA)
            mini = viz.render_court()
            present={row["player_id"] for row in indexed.get(index,[])}
            for side in trails:
                if side not in present:
                    trails[side].clear()
            for row in indexed.get(index, []):
                side = row["player_id"]
                color = viz.NEAR_C if side == "near" else viz.FAR_C
                bx, by, bw, bh = row["bbox"]
                cv2.rectangle(frame, (int(bx-bw/2), int(by-bh/2)), (int(bx+bw/2), int(by+bh/2)), color, 2)
                draw_skeleton(frame, row["keypoints"], color)
                cv2.circle(frame, (int(row["img_x"]), int(row["img_y"])), 5, color, -1)
                if not court.in_court((row['court_x'],row['court_y'])):
                    trails[side].clear()
                    continue
                trails[side].append((row["court_x"],row["court_y"]))
                trails[side]=trails[side][-60:]
                for a,b in zip(trails[side],trails[side][1:]):
                    cv2.line(mini,viz.to_px(*a),viz.to_px(*b),color,1,cv2.LINE_AA)
                viz.draw_point(mini, row["court_x"], row["court_y"], color, 5, side)
            for row in forecast.get(index,[]):
                if row['side'] in present:continue
                point=np.array([[row['court_x_m'],row['court_y_m']]])
                pixel=court.image_to_court(point,inverse)[0]
                if np.isfinite(pixel).all() and 0<=pixel[0]<meta['width'] and 0<=pixel[1]<meta['height']:
                    px,py=map(int,pixel)
                    cv2.circle(frame,(px,py),10,(0,215,255),2)
                    cv2.putText(frame,f"ESTIMATED {row['side']} +{row['prediction_age_s']:.1f}s",(px+12,py),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,215,255),1)
                if court.in_court(point[0]):
                    cv2.circle(mini,viz.to_px(*point[0]),8,(0,215,255),2)
            mini = cv2.resize(mini, (mini_width,mini_height))
            frame[8:8+mini_height,meta["width"]+12:meta["width"]+12+mini_width]=mini
            stamp = f"TRACKING | original {source_seconds(meta,index):.3f}s | frame {source_frame_number(meta,index)}"
            if estimates:stamp='ESTIMATE PREVIEW | '+stamp
            cv2.rectangle(frame, (0, meta["height"]-34), (meta["width"], meta["height"]), (20,20,20), -1)
            cv2.putText(frame, stamp, (8, meta["height"]-12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            frame = cv2.copyMakeBorder(frame,0,meta["height"]%2,0,canvas_width%2,cv2.BORDER_CONSTANT)
            encoded = av.VideoFrame.from_ndarray(frame, format="bgr24")
            encoded.pts = round((meta["times"][index] - base) / time_base)
            encoded.time_base = time_base
            for packet in stream.encode(encoded):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    return verify_overlay(destination, meta["times"][start:end:stride])


def heatmaps(observations, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    players = observations.player.unique()
    fig, axes = plt.subplots(1, len(players), figsize=(4*len(players), 7), squeeze=False)
    for ax, player in zip(axes[0], players):
        group = observations[(observations.player == player) & observations.observed]
        viz.mpl_court(ax)
        if len(group):
            ax.hist2d(group.court_x_m.to_numpy(dtype=float), group.court_y_m.to_numpy(dtype=float),
                      bins=[np.linspace(0, 6.1, 18), np.linspace(0, 13.4, 38)],
                      weights=group.sample_duration_s, cmap="magma", alpha=0.8, cmin=0.001)
        ax.set_title(f"{player}\nObserved seconds per cell")
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)


def render_report(run):
    manifest = json.loads((run / "analysis.json").read_text())
    validation = json.loads((run / "validation.json").read_text()) if (run / "validation.json").exists() else {"status":"pending"}
    lines = ["# ShuttleCoach — player occupancy review", "", f"Validation: **{validation['status'].upper()}**", "",
             "Player measurements only. See the separate Phase 2 evidence report for its coaching gate and suggestions.", "",
             f"Source SHA-256: `{manifest['video']['sha256']}`", f"Model SHA-256: `{manifest['model']['sha256']}`", "",
             "Near/far are court slots. Athlete names and rally boundaries are manual review inputs.", "",
             "| Interval | Player | Observed / sampled seconds | Coverage | Front / middle / rear occupancy | Review |",
             "|---|---|---:|---:|---|---|"]
    lines[4:4]=[f"Player positions sampled at {manifest['video']['fps']/manifest.get('tracking_stride',1):.1f} fps. Source timestamps are retained; unprocessed frames are not reported as detections.",""]
    if manifest["config"].get("includes_between_point_pauses"):
        lines[4:4]=["Retained analysis windows include setup and retrieval. These occupancy totals describe the selected windows; the Phase 2 report uses its separately reviewed rally intervals.",""]
    if (run/"screening.json").exists():
        screening=json.loads((run/"screening.json").read_text())
        lines[4:4]=[f"Automatic coverage screen: **{screening['status'].upper()}** — both athletes have reliable tracking in {screening['both_slots_observed_frames']}/{screening['sample_frames']} frozen sample frames. Required: at least 95%. Verified off-court tracking counts here; court occupancy still excludes it. This is an upper bound on correctness; identity and position require manual labels.",""]
    for summary in manifest["summaries"]:
        rally = manifest["config"]["rallies"][summary["rally_id"]-1]
        occupancy = " / ".join(f"{value:.1%}" if value is not None else "unavailable" for value in summary["occupancy"].values())
        # Prevent user-entered labels from introducing Markdown/HTML into the report.
        name = str(summary['player']).replace("|", "\\|").replace("\n", " ").replace("<", "&lt;")
        url = manifest["config"].get("youtube_url")
        original_start=rally.get('original_start_s',rally['start_s'])
        review = f"[source {original_start:.2f}s]({url}&t={int(original_start)}s)" if url else f"source {rally['start_s']:.2f}–{rally['end_s']:.2f}s"
        lines.append(f"| {summary['rally_id']} | {name} | {summary['observed_time_s']:.2f} / {summary['sampled_time_s']:.2f} | {summary['coverage']:.1%} | {occupancy} | {review} |")
    lines += ["", "Occupancy percentages use accepted observations only. Missing/low-confidence/out-of-court positions are excluded and reduce coverage.",
              "Heatmaps use camera coordinates; after an end change, near/far positions do not describe the same tactical orientation.",
              "Projected foot positions are approximate. The tracking overlay shows original source timestamps; this report carries the current validation status.",
              "", "## Review artifacts", "", "- [Overlay video](overlay.mp4)", "- [Occupancy heatmaps](heatmap.png)",
              "- [Timestamped observations](observations.csv)", "- [Manual reference template](reference_template.csv)",
              "- [Coverage screen](screening.json)", "- [Validation results](validation.json)", ""]
    if validation["status"] != "pending":
        lines += ["## Validation", "", "```json", json.dumps(validation, indent=2), "```", ""]
    if (run/"audit"/"provenance.json").exists():
        lines += ["[Visual-reference provenance](audit/provenance.json). The result applies only to the sampled frames in these analysis intervals.",""]
    (run / "report.md").write_text("\n".join(lines))


def analyze(config_path, model, device="auto"):
    config_path, model = Path(config_path).resolve(), Path(model).resolve()
    if not model.is_file():
        raise ValueError("Download the pose model first: python scripts/download_model.py")
    config = json.loads(config_path.read_text())
    meta = probe(config["video"])
    from .timeline import attach_timeline
    meta=attach_timeline(meta,config)
    homography = validate_config(config, meta)
    device = choose_device(device)
    times = np.asarray(meta["times"])
    start, end = [int(np.searchsorted(times, value, side="left")) for value in
                  (config["rallies"][0]["start_s"], config["rallies"][-1]["end_s"])]
    if end-start < 2:
        raise ValueError("Select at least two source frames.")
    stride=config.get("tracking_stride",1)
    if type(stride) is not int or not 1<=stride<=max(1,int(meta["fps"]/5)):
        raise ValueError("Tracking stride must retain at least five samples per second.")
    processed=list(range(start,end,stride))
    if len(processed)<2:
        raise ValueError("Select an interval with at least two tracking samples.")
    alignment=verify_tracker_frames(meta,start,end,stride)
    for relative, digest in json.loads((VENDOR / "PROVENANCE.json").read_text())["files"].items():
        if sha256(VENDOR / relative) != digest:
            raise ValueError(f"Vendored source changed: {relative}. Review provenance before inference.")
    run = RESULTS / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run.mkdir()
    if config.get('source_frame_map'):
        import shutil
        shutil.copy2(config['source_frame_map']['path'],run/'source_frame_map.csv')
    write_json(run / "status.json", {"state":"running", "stage":"tracking", "device":device})
    started = time.perf_counter()
    try:
        with bind_run(Path(meta["path"]), homography.flatten().tolist(), run / "tracks.duckdb"):
            detect.process_video("clip", str(model), start_frame=start, max_frames=len(processed),
                                 device=device, imgsz=1920, stride=stride)
        raw = raw_tracks(run / "tracks.duckdb")
        pd.DataFrame([{k:v for k,v in row.items() if k not in {"bbox","keypoints"}} for row in raw]).to_csv(run / "raw_tracks.csv", index=False)
        observations = build_observations(raw, meta, config["rallies"],processed)
        observations.to_csv(run / "observations.csv", index=False)
        template = sample_template(observations)
        template.to_csv(run / "reference_template.csv", index=False)
        write_json(run/"screening.json",screen_coverage(observations,template))
        write_json(run / "status.json", {"state":"running", "stage":"rendering", "device":device})
        timing = overlay(Path(meta["path"]), run / "overlay.mp4", raw, meta, start, end, homography,stride)
        heatmaps(observations, run / "heatmap.png")
        manifest = {"schema_version":1, "phase":"1", "validation":"pending", "coaching_enabled":False,
                    "config":config, "video":meta, "homography":homography.tolist(), "start_frame":start,
                    "end_frame_exclusive":end, "overlay_source_start_s":meta["times"][start], "overlay_timing":timing,
                    "tracker_frame_alignment":alignment,
                    "model":{"path":str(model),"sha256":sha256(model)}, "device":device,
                    "tracking_imgsz":1920,
                    "tracking_stride":stride,"processed_source_frames":processed,
                    "runtime_s":time.perf_counter()-started, "summaries":summarize(observations),
                    "artifacts_sha256":{name:sha256(run / name) for name in
                                        ("observations.csv","reference_template.csv","raw_tracks.csv","tracks.duckdb")},
                    "upstream":json.loads((VENDOR / "PROVENANCE.json").read_text()),
                    "versions":{name:importlib.metadata.version(name) for name in
                                ("torch","ultralytics","opencv-python","av","duckdb","streamlit")}}
        if config.get('source_frame_map'):
            manifest['artifacts_sha256']['source_frame_map.csv']=sha256(run/'source_frame_map.csv')
        write_json(run / "analysis.json", manifest)
        write_json(run / "validation.json",{"status":"pending","reason":"Complete the frozen manual references and unused calibration checks before evaluating accuracy."})
        render_report(run)
        write_json(run / "status.json", {"state":"complete", "validation":"pending", "runtime_s":manifest["runtime_s"]})
        print(f"RESULT={run}", flush=True)
    except BaseException as error:
        write_json(run / "status.json", {"state":"failed", "error":str(error)})
        raise
    return run


def validate_run(run, references, checks):
    run, references, checks = Path(run).resolve(), Path(references).resolve(), Path(checks).resolve()
    manifest = json.loads((run / "analysis.json").read_text())
    for name, digest in manifest["artifacts_sha256"].items():
        if sha256(run / name) != digest:
            raise ValueError(f"Analysis artifact changed: {name}. Validation requires the original frozen artifacts.")
    source = manifest["video"]
    if sha256(source["path"]) != source["sha256"]:
        raise ValueError("Source video changed since inference. References cannot be aligned safely.")
    labels = pd.read_csv(references)
    for x_field, y_field, table in [("image_x","image_y", labels[labels.visible.astype(str).str.lower().isin(["true","1","1.0"])])]:
        if len(table) and ((pd.to_numeric(table[x_field],errors="coerce") < 0).any()
                           or (pd.to_numeric(table[y_field],errors="coerce") < 0).any()
                           or (pd.to_numeric(table[x_field],errors="coerce") >= source["width"]).any()
                           or (pd.to_numeric(table[y_field],errors="coerce") >= source["height"]).any()):
            raise ValueError("Manual foot references must lie inside the source image.")
    check_data = json.loads(checks.read_text())
    if not isinstance(check_data,list):
        raise ValueError("Calibration checks must be a JSON list of points.")
    for point in check_data:
        if not (0 <= float(point["image_x"]) < source["width"] and 0 <= float(point["image_y"]) < source["height"]):
            raise ValueError("Calibration reference pixels must lie inside the source image.")
    result = evaluate(pd.read_csv(run / "observations.csv"), labels, check_data,
                      np.asarray(manifest["homography"]), pd.read_csv(run / "reference_template.csv"), manifest['config']['calibration'])
    result["reference_sha256"], result["calibration_checks_sha256"] = sha256(references), sha256(checks)
    write_json(run / "validation.json", result)
    # Preserve the submitted review inputs separately from raw detections.
    labels.to_csv(run / "reviewed_references.csv", index=False)
    write_json(run / "reviewed_calibration_checks.json", check_data)
    manifest["validation"]=result["status"]
    write_json(run/"analysis.json",manifest)
    status=json.loads((run/"status.json").read_text())
    status["validation"]=result["status"]
    write_json(run/"status.json",status)
    render_report(run)
    from .coaching import write_coaching_report
    write_coaching_report(run)
    return result


def revise_run(run, rallies, reuse_birdie=True):
    """Correct labels/intervals from cached detections; never rerun the neural model."""
    import shutil
    run = Path(run).resolve()
    manifest = json.loads((run / "analysis.json").read_text())
    meta = manifest["video"]
    if sha256(meta["path"]) != meta["sha256"]:
        raise ValueError("Source video changed since inference.")
    if manifest['config']['rallies']!=rallies:
        manifest['config'].pop('reviewed_rallies',None)
        manifest['config'].pop('rally_review',None)
    manifest["config"]["rallies"] = rallies
    validate_config(manifest["config"],meta)
    times=np.asarray(meta["times"])
    first,last=[int(np.searchsorted(times,t)) for t in (rallies[0]["start_s"],rallies[-1]["end_s"])]
    if first<manifest["start_frame"] or last>manifest["end_frame_exclusive"]:
        raise ValueError("Corrections must stay inside the tracked interval. Run tracking for a wider interval first.")
    for name,digest in manifest["artifacts_sha256"].items():
        if sha256(run/name)!=digest:
            raise ValueError(f"Cached artifact changed: {name}.")
    revised=RESULTS/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    revised.mkdir()
    for name in dict.fromkeys(["tracks.duckdb","raw_tracks.csv","overlay.mp4",*manifest['artifacts_sha256']]):
        shutil.copy2(run/name,revised/name)
    observations=build_observations(raw_tracks(revised/"tracks.duckdb"),meta,rallies,manifest.get("processed_source_frames"))
    observations.to_csv(revised/"observations.csv",index=False)
    sample_template(observations).to_csv(revised/"reference_template.csv",index=False)
    write_json(revised/"screening.json",screen_coverage(observations,sample_template(observations)))
    heatmaps(observations,revised/"heatmap.png")
    manifest["summaries"]=summarize(observations)
    manifest["parent_run"]=str(run)
    manifest["validation"]="pending"
    manifest['coaching_enabled']=False
    manifest['coaching_gate']='blocked'
    manifest["inference_reused"]=True
    manifest["artifacts_sha256"]={name:sha256(revised/name) for name in manifest["artifacts_sha256"]}
    write_json(revised/"analysis.json",manifest)
    write_json(revised/"validation.json",{"status":"pending","reason":"Review corrections create a new validation scope; prior accuracy results are not reused."})
    render_report(revised)
    write_json(revised/"status.json",{"state":"complete","validation":"pending","inference_reused":True})
    if reuse_birdie and (run/"shuttle.json").exists():
        from .shuttle import reuse_shuttle
        if reuse_shuttle(run,revised):
            from .coaching import write_coaching_report
            write_coaching_report(revised)
    return revised
