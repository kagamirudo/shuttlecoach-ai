"""Local review UI. Inference runs in a separate process and persists to run folders."""
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

from shuttlecoach.analysis import MAX_ANALYSIS_SPAN_S, validate_config, player_overview
from shuttlecoach.calibration import LANDMARKS, fit_calibration
from shuttlecoach.media import download, probe, read_frame, write_json, youtube_id
from shuttlecoach.paths import DATA, MODELS, RESULTS, ROOT
from shuttlecoach.runner import validate_run, revise_run
from shuttlecoach.shuttle import court_zone
from shuttlecoach.timeline import source_seconds, source_frame_number
from shuttlecoach.upstream import court, viz

st.set_page_config(page_title="ShuttleCoach · Match review", page_icon="🏸", layout="wide")
st.title("ShuttleCoach")
st.caption("Local singles and birdie tracking · Manual calibration · Coaching gated on validated evidence")


@st.cache_data(show_spinner="Reading the video timeline…")
def metadata(path, size, modified):
    return probe(path)


@st.cache_data(max_entries=24, show_spinner=False)
def source_frame(path, frame, modified, times, origin):
    return read_frame(path, frame, times, origin)


def frame_image(meta, frame):
    return source_frame(meta["path"], int(frame), Path(meta["path"]).stat().st_mtime_ns,
                        tuple(meta["times"]), meta["pts_origin_s"])


def click_image(image, key):
    width = min(960, image.shape[1])
    shown = cv2.resize(image, (width, round(image.shape[0] * width / image.shape[1])))
    click = streamlit_image_coordinates(shown, width="stretch", key=key, cursor="crosshair")
    if click:
        # Widget reports its displayed image dimensions; account for responsive sizing.
        scale_x = image.shape[1] / click.get("width", shown.shape[1])
        scale_y = image.shape[0] / click.get("height", shown.shape[0])
        return {"x":click["x"] * scale_x, "y":click["y"] * scale_y, "event":click.get("unix_time", str(click))}


def current_runs():
    return sorted([p for p in RESULTS.iterdir() if (p / "analysis.json").is_file()], reverse=True)


@st.fragment(run_every="2s")
def job_progress():
    process = st.session_state.get("process")
    if process is None:
        return
    code = process.poll()
    if code is not None and not st.session_state.get('process_completion_refreshed', False):
        st.session_state['process_completion_refreshed'] = True
        st.rerun(scope='app')
    log = Path(st.session_state["log"]).read_text(errors="replace")
    if code is None:
        st.info("Tracking is running locally. Results appear when video and exports finish.")
    elif code:
        st.error("The analysis stopped. Check the details below; no result was marked valid.")
    else:
        st.success("Analysis finished. Manual validation is still required.")
        if st.button("Open completed analysis"):
            result = next((line[7:] for line in log.splitlines() if line.startswith("RESULT=")), None)
            if result:
                st.session_state["selected_run"] = result
                st.session_state.pop("process", None)
                st.rerun()
    with st.expander("Processing log"):
        st.code(log[-8000:] or "Starting…", language=None)


prepare, review = st.tabs(["Prepare a clip", "Review & validate"])
with prepare:
    st.subheader("1 · Choose footage")
    mode = st.radio("Source", ["Local file", "YouTube"], horizontal=True)
    if mode == "YouTube":
        url = st.text_input("YouTube video URL", value="https://www.youtube.com/watch?v=RZaZ7GaPJU8")
        permitted = st.checkbox("I’m authorized to download this footage")
        if st.button("Retrieve video", disabled=not permitted):
            try:
                with st.spinner("Retrieving one video…"):
                    st.session_state["video_path"] = str(download(url))
                    st.session_state["youtube_url"] = f"https://www.youtube.com/watch?v={youtube_id(url)}"
            except Exception as error:
                st.error(f"Download unavailable: {error}. Use a local MP4/MOV.")
    else:
        local = st.text_input("Absolute MP4/MOV path", value=st.session_state.get("video_path", ""))
        upload = st.file_uploader("Or upload a local recording", type=["mp4","mov","m4v"])
        if st.button("Load local video"):
            try:
                if upload:
                    buffer = upload.getbuffer()
                    path = DATA / f"upload-{hashlib.sha256(buffer).hexdigest()[:16]}{Path(upload.name).suffix.lower()}"
                    if not path.exists():
                        path.write_bytes(buffer)
                else:
                    path = Path(local).expanduser().resolve()
                if not path.is_file():
                    raise ValueError("Select an existing local file.")
                st.session_state["video_path"] = str(path)
                st.session_state.pop("youtube_url", None)
            except Exception as error:
                st.error(str(error))
    path = Path(st.session_state.get("video_path", ""))
    if path.is_file():
        try:
            stat = path.stat()
            meta = metadata(str(path), stat.st_size, stat.st_mtime_ns)
            if st.session_state.get("source_hash") != meta["sha256"]:
                st.session_state["source_hash"] = meta["sha256"]
                st.session_state["corners"] = []
                st.session_state["calibration_review"] = False
                st.session_state.pop("cal_frame_previous", None)
            st.caption(f"{meta['width']} × {meta['height']} · {meta['duration_s']:.2f} seconds · {meta['frame_count']:,} timestamped frames")
            st.video(str(path))
            st.subheader("2 · Mark the rallies")
            st.write(f"Select up to {MAX_ANALYSIS_SPAN_S // 60} minutes from one fixed camera view. Times refer to the source video. Player movement is sampled at about 10 fps. Editing the times or player names and running again creates a separate result.")
            rally_table = st.data_editor(pd.DataFrame([{"start_s":0.0,"end_s":min(30.0,meta["duration_s"]),
                                                     "near_player":"Player A","far_player":"Player B"}]),
                                         num_rows="dynamic", key=f"rallies_{meta['sha256']}", hide_index=True)
            singles = st.checkbox("Exactly two singles players are visible")
            steady = st.checkbox("The selected camera view stays fixed without cuts")
            st.subheader("3 · Calibrate the court")
            requested = st.number_input("Calibration time in source seconds", min_value=0.0,
                                        max_value=float(meta["times"][-1]), value=0.0, step=0.1)
            frame = min(int(np.searchsorted(meta["times"], requested)), meta["frame_count"]-1)
            method = st.radio("Calibration method", ["Four outer corners", "Visible line intersections"], horizontal=True)
            intersections = method == "Visible line intersections"
            context = (meta['sha256'], frame, method)
            if st.session_state.get('calibration_context') != context:
                st.session_state['corners'] = []
                st.session_state['landmark_points'] = []
                st.session_state['calibration_review'] = False
                st.session_state['calibration_context'] = context
                st.session_state.pop('last_cal_click', None)
            corners = st.session_state['corners']
            pairs = st.session_state['landmark_points']
            calibration = {"frame":frame, "mode":"intersections" if intersections else "corners"}
            selected = None
            if intersections:
                st.write("Choose a numbered painted-line intersection, then click that same intersection in the video. Mark at least four points spread across court width and depth; six or more is better. Corners may be off-screen.")
                st.caption("Near is the camera-side baseline (bottom of the diagram); left/right are viewed toward the far baseline. Use intersections on the same court, not net posts, line endpoints at the image edge, or neighboring courts.")
                available = [name for name in LANDMARKS if name not in {p['landmark'] for p in pairs}]
                selected = st.selectbox("Intersection to mark", available, format_func=lambda name:f"{list(LANDMARKS).index(name)+1} · {name}", disabled=not available)
                diagram = viz.render_court()[:, :, ::-1].copy()
                for number, (name, (x, y)) in enumerate(LANDMARKS.items(), 1):
                    px, py = viz.to_px(x, y)
                    color = (255, 190, 30) if name == selected else (255, 255, 255)
                    cv2.circle(diagram, (px, py), 3, color, -1)
                    cv2.putText(diagram, str(number), (px-7, py-5), cv2.FONT_HERSHEY_SIMPLEX, .32, color, 1, cv2.LINE_AA)
                with st.expander("Numbered court reference — near baseline at bottom", expanded=True):
                    st.image(diagram, width=350)
                calibration['points'] = pairs
            else:
                st.write("Click the four **outer doubles-court corners**, in order: near-left → near-right → far-right → far-left. All four physical intersections must be visible. If a corner is off-screen, select Visible line intersections.")
                calibration['corners_px'] = corners
            image = frame_image(meta, frame).copy()
            ready = False
            try:
                h, inferred_corners = fit_calibration(calibration, meta)
                calibration['corners_px'] = inferred_corners.tolist()
                inverse = np.linalg.inv(h)
                for a, b in viz._segments():
                    points = court.image_to_court(np.array([a,b]), inverse).astype(int)
                    cv2.line(image, tuple(points[0]), tuple(points[1]), (70,240,110), 2)
                ready = True
            except ValueError as issue:
                st.info(str(issue))
            marked = [(list(LANDMARKS).index(p['landmark'])+1, [p['image_x'],p['image_y']]) for p in pairs] if intersections else list(enumerate(corners,1))
            for number, point in marked:
                cv2.circle(image, tuple(map(round,point)), 6, (255,190,30), -1)
                cv2.putText(image, str(number), tuple(map(round,point)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            click = click_image(image, f"calibration_{meta['sha256']}_{frame}_{intersections}")
            if click and st.session_state.get("last_cal_click") != click["event"]:
                st.session_state["last_cal_click"] = click["event"]
                if intersections and selected:
                    pairs.append({'landmark':selected, 'image_x':click['x'], 'image_y':click['y']})
                    st.session_state["calibration_review"] = False
                    st.rerun()
                elif not intersections and len(corners) < 4:
                    corners.append([click["x"],click["y"]])
                    st.session_state["calibration_review"] = False
                    st.rerun()
            marked_points = pairs if intersections else corners
            if st.button("Undo intersection" if intersections else "Undo corner", disabled=not marked_points):
                marked_points.pop()
                st.session_state["calibration_review"] = False
                st.rerun()
            if st.button("Clear calibration", disabled=not marked_points):
                marked_points.clear()
                st.session_state["calibration_review"] = False
                st.rerun()
            count = f"{len(pairs)} intersections marked" if intersections else f"{len(corners)}/4 corners"
            st.caption(f"Source frame {frame} at {meta['times'][frame]:.3f}s · {count}")
            if intersections and pairs:
                st.dataframe(pd.DataFrame(pairs), hide_index=True)
            st.caption("Green lines must align with the visible painted lines. A fitted preview is not an accuracy pass; later validation needs at least four unused intersections.")
            confirmed = st.checkbox("I reviewed the projected court lines", key="calibration_review", disabled=not ready)
            calibration['reviewed'] = confirmed
            config = {"video":str(path.resolve()), "singles_confirmed":singles, "camera_stable_confirmed":steady,
                      "calibration":calibration, "rallies":rally_table.to_dict("records")}
            config["tracking_stride"]=max(1,round(meta["fps"]/10))
            if st.session_state.get("youtube_url"):
                config["youtube_url"] = st.session_state["youtube_url"]
            error = None
            try:
                validate_config(config, meta)
            except (ValueError, TypeError, KeyError) as issue:
                error = str(issue)
            if not error:
                span=config['rallies'][-1]['end_s']-config['rallies'][0]['start_s']
                st.caption(f"Selected tracking span: {span:.1f}s / {MAX_ANALYSIS_SPAN_S}s maximum per run")
            if error:
                st.info(error)
            devices=["auto","cpu"] if os.environ.get('SHUTTLECOACH_CONTAINER')=='1' else ["auto","cpu","mps"]
            device = st.selectbox("Processing device", devices)
            active = st.session_state.get("process")
            active = active is not None and active.poll() is None
            if st.button("Run player tracking", type="primary", disabled=bool(error) or active):
                job_id = __import__("uuid").uuid4().hex
                job_dir = ROOT / ".cache" / job_id
                job_dir.mkdir()
                config_file, log_file = job_dir / "config.json", job_dir / "worker.log"
                write_json(config_file, config)
                with log_file.open("w") as log:
                    process = subprocess.Popen([sys.executable,"-u","-m","shuttlecoach.cli","analyze",str(config_file),"--device",device],
                                               cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                st.session_state["process"], st.session_state["log"] = process, str(log_file)
                st.session_state['process_completion_refreshed'] = False
                st.rerun()
            st.download_button("Download reviewed configuration", json.dumps(config,indent=2), "clip-config.json", "application/json",on_click="ignore")
        except Exception as error:
            st.error(f"Cannot prepare this video: {error}")
    job_progress()

with review:
    runs = current_runs()
    if not runs:
        st.info("Run a clip first. Tracking outputs will appear here with validation pending.")
    else:
        selected = st.session_state.get("selected_run")
        index = next((i for i,p in enumerate(runs) if str(p)==selected),0)
        run = st.selectbox("Analysis", runs, index=index, format_func=lambda p:p.name)
        manifest = json.loads((run / "analysis.json").read_text())
        validation = json.loads((run / "validation.json").read_text()) if (run / "validation.json").exists() else {"status":"pending"}
        st.subheader(f"Validation: {validation['status']}")
        st.caption("Player and birdie tracking are validated separately. Coaching requires both gates and reviewed rally intervals.")
        if validation.get('criteria'):
            a,b,c=st.columns(3)
            a.metric("Reviewed player pairs usable",f"{validation['both_players_correct_rate']:.1%}")
            error=validation.get('median_projected_error_m')
            b.metric("Median foot-position error",f"{error:.2f} m" if error is not None else 'Not measurable')
            errors=validation.get('calibration_check_errors_m',[])
            c.metric("Largest court-line check error",f"{max(errors):.2f} m" if errors else 'Not checked')
            st.caption("Player-pair target: at least 95%. Median foot error: at most 0.75 m; 90th percentile: at most 1.5 m. Four unused court-line checks must each be within 0.25 m.")
            hidden=2*validation['sample_frames']-validation['visible_references']
            if hidden:
                st.write(f"In the manual sample, **{hidden} player-foot positions were not visible**. Those samples remain in the result and count against the usable-pair requirement.")
            if validation['status']=='failed':
                st.warning("Player tracking has not passed validation. Use the video to inspect gaps and identity errors; positioning numbers are not ready for coaching.")
        if (run/"screening.json").exists():
            screening=json.loads((run/"screening.json").read_text())
            message=f"Coverage screen: {screening['both_slots_observed_frames']}/{screening['sample_frames']} sampled frames contain both reliable athlete positions (95% required). Verified off-court tracking is separate from court occupancy."
            if validation["status"]=="pending":
                message+=" Identity and position accuracy still require manual labels."
            if screening["status"]=="failed":
                st.warning("This run cannot pass the current tracking gate. "+message)
            else:
                st.info(message)
        playback_options=["Analyzed clip","Full source video"]
        motion_path=run/'motion.json'
        motion=json.loads(motion_path.read_text()) if motion_path.exists() else {}
        if motion and (run/'motion_preview.mp4').exists():playback_options.append('Short-gap motion preview')
        if (run/"shuttle_overlay.mp4").exists() and (run/"shuttle.json").exists():playback_options.append("Birdie tracking")
        playback=st.radio("Playback",playback_options,horizontal=True,key=f"playback_{run.name}")
        if playback=="Full source video":
            original=manifest['video'].get('original_source',manifest['video'])
            st.video(original["path"])
            st.caption(f"Full match · {original['duration_s']/60:.2f} minutes · original audio")
            st.caption("Validation applies to the selected analysis intervals, not the entire source video.")
            st.code(original["path"],language=None)
        elif playback=='Short-gap motion preview':
            st.video(str(run/'motion_preview.mp4'))
            st.warning('Yellow hollow markers are estimated positions for gaps up to 0.3 seconds. They are excluded from tracking accuracy, court occupancy, footwork evidence and coaching. Longer occlusions remain empty.')
        elif playback=="Birdie tracking":
            st.video(str(run/"shuttle_overlay.mp4"))
            st.caption("Yellow circles and trails are observed image candidates from TrackNet or source foreground. Validation below measures their accuracy. The mini-court is an image projection through the calibrated floor homography, not the airborne birdie's true floor position. Gaps stay empty.")
        else:
            st.video(str(run / "overlay.mp4"))
            if manifest['video'].get('original_source'):
                intervals=manifest['config'].get('reviewed_rallies',manifest['config']['rallies'])
                seconds=sum(r['end_s']-r['start_s'] for r in intervals)
                st.caption(f'{seconds:.1f} seconds in manually reviewed rally intervals (approximate boundaries), plus retained setup and retrieval context. Watermarks and exported original_source_time_s retain the original match clock. Full source video plays the complete match with audio.')
            else:st.caption(f"This overlay covers source {manifest['overlay_source_start_s']:.3f}–{manifest['video']['times'][manifest['end_frame_exclusive']-1]:.3f}s. Choose Full source video above for the complete match with audio.")
        st.image(str(run / "heatmap.png"))
        st.subheader("Your tracking results")
        measurements=pd.read_csv(run/'observations.csv')
        overview=player_overview(measurements)
        st.write("Coverage is the amount of selected footage with an accepted player position. It does not measure whether the tracker followed the correct person or placed their feet accurately.")
        for column, person in zip(st.columns(max(1,len(overview))),overview):
            with column:
                st.subheader(person['player'])
                sides=['near camera' if side=='near' else 'far side' for side in person['sides']]
                st.caption(' / '.join(sides)+' · names are the labels you entered')
                st.metric("Tracking coverage",f"{person['coverage']:.1%}")
                st.progress(min(1.,max(0.,person['coverage'])))
                st.write(f"Tracked for **{person['tracked_s']:.1f} of {person['selected_s']:.1f} seconds**. {person['unavailable_s']:.1f} seconds have no accepted position.")
                st.markdown("**Where accepted court positions fell**")
                if person['in_court_s']:
                    for zone,label in [('front','Front · near the net'),('middle','Middle'),('rear','Rear · near the baseline')]:
                        share=person['occupancy'][zone]
                        st.write(f"{label}: **{share:.1%}**")
                        st.progress(min(1.,max(0.,share)))
                    st.caption(f"Based on {person['in_court_s']:.1f} seconds inside the court. Missing positions and verified off-court positions are excluded.")
                else:st.info("No accepted in-court positions to summarize.")
                if person['coverage']<.95:
                    st.warning("Too much footage is unavailable to interpret this player's positioning reliably.")
        st.caption(f"{len(manifest['config']['rallies'])} manually selected time window(s), not an automatically detected rally count. Positioning includes pauses in those windows. Front, middle and rear are thirds of each court half measured from the net.")
        if validation['status']!='passed':
            st.info("Next step: inspect the tracking video, then validate player identities, foot positions and unused court-line intersections below. Position percentages are descriptive measurements, not coaching conclusions.")
        with st.expander("Detailed measurements and CSV"):
            st.dataframe(pd.json_normalize(manifest["summaries"]), hide_index=True)
            st.download_button("Download player measurements",(run/'observations.csv').read_bytes(),'player-measurements.csv','text/csv',on_click='ignore')
        from shuttlecoach.report_ui import show_ai_analysis
        running=st.session_state.get('process')
        show_ai_analysis(run,manifest,running is not None and running.poll() is None)
        st.subheader("Phase 2 · Birdie tracking & evidence report")
        shuttle_status=json.loads((run/"shuttle_status.json").read_text()) if (run/"shuttle_status.json").exists() else {}
        if shuttle_status.get("state")=="running":
            st.info(f"Birdie analysis: {shuttle_status.get('stage','tracking')} · {shuttle_status.get('processed_frames',0):,} source frames processed. Refresh to see progress.")
        if shuttle_status.get("state")=="failed":st.error(f"Birdie analysis stopped: {shuttle_status.get('error','Check the worker log.')}")
        active=st.session_state.get("process")
        active=active is not None and active.poll() is None
        previous_shuttle=json.loads((run/"shuttle.json").read_text()) if (run/"shuttle.json").exists() else {}
        needs_court_review=bool(previous_shuttle) and (previous_shuttle.get("detection_zone")!=court_zone(manifest) or not previous_shuttle.get('candidate_mode'))
        label="Reprocess birdie with high-shot zone" if needs_court_review else "Track birdie in court and airspace"
        if st.button(label,disabled=active or shuttle_status.get("state")=="running" or (shuttle_status.get("state")=="complete" and not needs_court_review) or not (MODELS/"TrackNet_best.pt").exists()):
            log_file=ROOT/".cache"/f"shuttle-{run.name}.log"
            command=[sys.executable,"-u","-m","shuttlecoach.cli","shuttle",str(run)]
            if needs_court_review:command.append("--new-court-review")
            command.append('--candidates')
            with log_file.open("w") as log:
                process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            st.session_state["process"],st.session_state["log"]=process,str(log_file)
            st.session_state['process_completion_refreshed'] = False
            st.rerun()
        if (run/"shuttle.json").exists():
            shuttle_manifest=json.loads((run/"shuttle.json").read_text())
            cadence=shuttle_manifest.get('sampling_fps',manifest['video']['fps'])
            st.caption(f"{shuttle_manifest['frame_count']:,} sampled source frames at {cadence:.1f} fps · {shuttle_manifest['detected_frames']:,} candidate detections. Detection frequency is not accuracy.")
            if shuttle_manifest.get("detection_zone"):
                if shuttle_manifest['detection_zone'].get('kind')=='calibrated_court_with_airspace':
                    st.info("The green zone includes the court and raised airspace. Wider candidate mode admits near-side high shots and overlapping neighboring courts; the zone alone cannot identify the match birdie. Points above the floor footprint have no invented floor coordinates.")
                else:
                    st.info("This older run uses a floor-only zone. Reprocess with the high-shot zone to include birdies above the court.")
                with st.expander("Birdie court trajectory",expanded=True):
                    st.image(str(run/"shuttle_court_projection.png"),width=420)
                    st.caption("Projected image path only. Missing detections and large jumps break the trail. This is not a landing map or a physical flight path.")
                    st.download_button("Download birdie court trajectory",(run/"shuttle_court_projection.png").read_bytes(),"birdie-court-projection.png","image/png",on_click="ignore")
                with st.expander("Review the detection zone"):
                    st.image(str(run/"shuttle_zone_preview.png"))
            st.download_button("Download timestamped birdie CSV",(run/"shuttle.csv").read_bytes(),"shuttle.csv","text/csv",on_click="ignore")
            with st.expander("Validate birdie tracking on the frozen source sample"):
                st.write("Label the match birdie on original frames before viewing predictions. Use not_visible only when absence or occlusion is clear; leave ambiguous frames unreviewed. All 100 fixed samples must be reviewed, including at least 25 visible and 25 not visible. Precision ≥90%, recall ≥85%, median error ≤8 pixels at 720p; location match tolerance 16 pixels at 720p.")
                birdie_template=pd.read_csv(run/"shuttle_reference_template.csv")
                birdie_labels_path=run/"shuttle_reference_labels.csv"
                saved_birdie=birdie_labels_path if birdie_labels_path.exists() else run/'shuttle_reviewed_references.csv'
                birdie_labels=pd.read_csv(saved_birdie) if saved_birdie.exists() else birdie_template.copy()
                birdie_frame=st.select_slider("Birdie reference frame",options=birdie_template.source_frame.tolist(),key=f"shuttle_frame_{run.name}")
                birdie_index=birdie_labels.index[birdie_labels.source_frame==birdie_frame][0]
                saved=birdie_labels.loc[birdie_index]
                point=click_image(frame_image(manifest["video"],birdie_frame),f"birdie_reference_{run.name}_{birdie_frame}")
                visibility=st.selectbox("Match birdie visibility",["unreviewed","visible","not_visible"],index=["unreviewed","visible","not_visible"].index(saved.visibility),key=f"birdie_visibility_{run.name}_{birdie_frame}")
                st.caption(f"Original source {source_seconds(manifest['video'],birdie_frame):.3f}s. Click the birdie center; neighboring courts do not count.")
                if st.button("Save birdie reference",disabled=visibility=="visible" and point is None):
                    birdie_labels.loc[birdie_index,"visibility"]=visibility
                    birdie_labels.loc[birdie_index,["image_x","image_y"]]=[point['x'],point['y']] if visibility=="visible" else [None,None]
                    birdie_labels.to_csv(birdie_labels_path,index=False)
                    st.success("Birdie reference saved.")
                st.download_button("Download birdie reference CSV",birdie_labels.to_csv(index=False),"shuttle_references.csv","text/csv",on_click="ignore")
                imported=st.file_uploader("Import completed birdie references",type="csv",key=f"shuttle_import_{run.name}")
                if st.button("Evaluate birdie validation"):
                    try:
                        from shuttlecoach.shuttle import validate_shuttle
                        submitted=run/"shuttle_submitted_references.csv"
                        submitted.write_bytes(imported.getvalue()) if imported else submitted.write_text(birdie_labels.to_csv(index=False))
                        validate_shuttle(run,submitted)
                        st.rerun()
                    except (ValueError,OSError,KeyError) as error:st.error(str(error))
            if (run/"shuttle_validation.json").exists():
                st.write("Birdie validation:",json.loads((run/"shuttle_validation.json").read_text())["status"])
            if shuttle_manifest.get('candidate_mode'):
                with st.expander('Review play and associate the match birdie'):
                    st.write('Review source-video seconds. Play intervals exclude pauses; add visible shuttle transfers during pauses below. Saving creates a separate associated run, with both accuracy gates pending on the original full-window samples.')
                    review=manifest['config'].get('rally_review',{})
                    play=st.data_editor(pd.DataFrame(review.get('reviewed_rallies',manifest['config']['rallies'])),num_rows='dynamic',hide_index=True,key=f'play_{run.name}')
                    play_pairs=[[r['start_s'],r['end_s']] for r in review.get('reviewed_rallies',[])]
                    transfers=[r for r in review.get('target_flight_intervals',[]) if r not in play_pairs]
                    extras=st.text_area('Other visible shuttle-flight intervals (JSON pairs)',json.dumps(transfers),key=f'transfers_{run.name}')
                    reviewer=st.text_input('Activity reviewer',review.get('reviewer',''),key=f'activity_reviewer_{run.name}')
                    confirmed=st.checkbox('I reviewed these activity intervals against the original video',key=f'activity_confirm_{run.name}')
                    if st.button('Associate birdie with reviewed activity',disabled=active or not confirmed or not reviewer.strip()):
                        try:
                            from shuttlecoach.shuttle_association import validate_review,merge_activity_intervals
                            play_records=json.loads(play.to_json(orient='records'))
                            activity={'source_sha256':manifest['video']['sha256'],'reviewer':reviewer.strip(),
                                      'reviewed_rallies':play_records,'boundary_precision_s':1.,
                                      'target_flight_intervals':merge_activity_intervals([[r['start_s'],r['end_s']] for r in play_records]+json.loads(extras))}
                            validate_review(activity,manifest)
                            saved=ROOT/'.cache'/f'activity-review-{run.name}.json';write_json(saved,activity)
                            log_file=ROOT/'.cache'/f'associate-{run.name}.log'
                            with log_file.open('w') as log:
                                process=subprocess.Popen([sys.executable,'-u','-m','shuttlecoach.cli','associate-shuttle',str(run),'--review',str(saved)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                            st.session_state['process'],st.session_state['log']=process,str(log_file)
                            st.session_state['process_completion_refreshed'] = False
                            st.rerun()
                        except (ValueError,TypeError,KeyError) as error:st.error(str(error))
        if st.button("Refresh evidence report"):
            from shuttlecoach.coaching import write_coaching_report
            with st.spinner("Checking saved tracking evidence…"):write_coaching_report(run)
            st.rerun()
        if (run/"coaching.json").exists():
            coaching=json.loads((run/"coaching.json").read_text())
            from shuttlecoach.coaching import evidence_signature
            try:current_report=bool(coaching.get('evidence_signature')) and coaching['evidence_signature']==evidence_signature(run)
            except (OSError,KeyError,ValueError):current_report=False
            st.subheader('Patterns and practice')
            if not current_report:st.warning('Saved coaching is out of date. Refresh the evidence report to recheck tracking and footwork review before showing practice suggestions.')
            elif coaching['gate']=='blocked':st.warning('Practice suggestions are locked: '+' '.join(coaching['blocked_reasons']))
            elif not coaching['suggestions']:st.info('Tracking passed, but there is not enough repeated, reviewed evidence for a practice suggestion yet.')
            else:
                st.success('Tracking gates passed. Each practice card below links the reviewed evidence that supports it.')
                for number,item in enumerate(coaching['suggestions'],1):
                    with st.container(border=True):
                        st.subheader(f"{item['player']} · {item.get('title','Positioning pattern')}")
                        st.markdown('**What repeated**')
                        st.write(item['pattern'])
                        st.markdown('**What to practice**')
                        st.write(item['suggestion'])
                        st.markdown('**Review the evidence**')
                        evidence=item['evidence']
                        selected_evidence=st.selectbox('Video moment',range(len(evidence)),format_func=lambda i:f"{evidence[i]['start_s']:.1f}–{evidence[i]['end_s']:.1f}s",key=f'evidence_{run.name}_{number}')
                        original=manifest['video'].get('original_source',manifest['video'])
                        st.video(original['path'],start_time=max(0,int(evidence[selected_evidence]['start_s'])-1))
                from shuttlecoach.footwork import PRACTICE_SOURCE
                st.caption('Practice topics are conditional review prompts, not technique grades or an optimal-base diagnosis.')
                st.link_button('BWF movement and recovery reference',PRACTICE_SOURCE)
            with st.expander("Read Phase 2 evidence report"):
                report_text=(run/"coaching_report.md").read_text()
                if current_report:st.markdown(re.sub(r"\[([^\]]+)\]\((?!https?://)[^)]+\)",r"\1",report_text))
                else:st.info('Refresh the evidence report before reading or downloading current advice.')
            if current_report:st.download_button("Download Phase 2 report",report_text,"shuttlecoach-phase2-report.md","text/markdown",on_click="ignore")
        st.subheader('Footwork · movement review')
        st.write('Review lunge shapes, stance openings and left/right foot movements. These are camera-plane candidates. Opponent-hit, landing and ready-stance times need source-video confirmation; there is no automatic physical balance or technique grade.')
        if st.button('Analyze footwork from saved poses',disabled=active or not (run/'tracks.duckdb').exists()):
            from shuttlecoach.footwork import write_footwork
            try:
                with st.spinner('Analyzing observed poses…'):write_footwork(run)
                st.rerun()
            except (ValueError,OSError,KeyError) as error:st.error(str(error))
        if (run/'footwork.json').exists():
            from shuttlecoach.footwork import EVENT_NAMES,reviewed_footwork
            foot=json.loads((run/'footwork.json').read_text())
            for column,person in zip(st.columns(max(1,len(foot['summaries']))),foot['summaries']):
                with column:
                    st.markdown(f"**{person['player']}**")
                    st.metric('Usable full-body poses',f"{person['pose_coverage']:.1%}")
                    st.caption(f"{person['pose_seconds']:.1f} of {person['selected_seconds']:.1f} selected play seconds. Missing poses and cuts break sequences.")
                    for kind,label in EVENT_NAMES.items():st.write(f"{label} candidates: **{person['candidate_counts'][kind]}**")
            st.info('Candidate counts are not confirmed steps or split steps. Footwork practice needs both tracking gates, reviewed rallies, and confirmed events in at least three sufficiently covered rallies.')
            review_path=run/'footwork_review.json'
            saved_review=json.loads(review_path.read_text()) if review_path.exists() else {}
            try:
                reviewed=reviewed_footwork(foot,saved_review,foot['intervals'])
                st.caption(f"Footwork review: {reviewed['status']} · {len(reviewed['rally_ids'])} candidate-bearing rallies fully reviewed.")
            except (ValueError,KeyError) as error:st.warning(str(error))
            with st.expander('Review footwork events and timing'):
                st.download_button('Download footwork candidates',(run/'footwork_events.csv').read_bytes(),'footwork-events.csv','text/csv',on_click='ignore')
                if foot['events']:
                    rally_id=st.selectbox('Footwork rally',sorted({r['rally_id'] for r in foot['events']}),key=f'foot_rally_{run.name}')
                    candidates=[r for r in foot['events'] if r['rally_id']==rally_id]
                    event_id=st.selectbox('Movement to review',[r['event_id'] for r in candidates],format_func=lambda value:next(f"{r['player']} · {EVENT_NAMES[r['kind']]} · original {r['original_source_time_s']:.2f}s" for r in candidates if r['event_id']==value),key=f'foot_event_{run.name}')
                    event=next(r for r in candidates if r['event_id']==event_id)
                    matching=saved_review.get('evidence_sha256')==foot['evidence_sha256']
                    labels={r['event_id']:r for r in saved_review.get('events',[])} if matching else {}
                    old=labels.get(event_id,{})
                    key=f"{run.name}_{foot['evidence_sha256'][:10]}_{event_id}"
                    frame=event['source_frame'];times=np.asarray(manifest['video']['times'])
                    choices=np.flatnonzero((times>=times[frame]-2)&(times<=times[frame]+3)).tolist()
                    chosen=st.select_slider('Footwork review frame',options=choices,value=frame,key=f'foot_frame_{key}')
                    st.image(frame_image(manifest['video'],chosen))
                    stamp=source_seconds(manifest['video'],chosen)
                    st.caption(f'Original source {stamp:.3f}s · frame {source_frame_number(manifest["video"],chosen)}. Review both feet and the opponent. Choose a frame, then use the relevant timestamp button.')
                    a,b,c=st.columns(3)
                    if a.button('Use frame as opponent hit',disabled=event['kind']!='stance_opening',key=f'hit_button_{key}'):st.session_state[f'hit_{key}']=stamp
                    if b.button('Use frame as landing',disabled=event['kind']!='stance_opening',key=f'land_button_{key}'):st.session_state[f'land_{key}']=stamp
                    if c.button('Use frame as ready',disabled=event['kind']!='lunge',key=f'ready_button_{key}'):st.session_state[f'ready_{key}']=stamp
                    confirmation=st.selectbox('Does the video confirm this movement?', ['Not reviewed','Confirm','Reject'],index=0 if type(old.get('confirmed')) is not bool else 1 if old['confirmed'] else 2,key=f'confirmed_{key}')
                    hit=st.number_input('Opponent hit · original seconds',min_value=0.,value=old.get('opponent_hit_s'),step=.01,disabled=event['kind']!='stance_opening',key=f'hit_{key}')
                    landing=st.number_input('Landing · original seconds',min_value=0.,value=old.get('landing_s'),step=.01,disabled=event['kind']!='stance_opening',key=f'land_{key}')
                    ready=st.number_input('Ready stance · original seconds',min_value=0.,value=old.get('ready_s'),step=.01,disabled=event['kind']!='lunge',key=f'ready_{key}')
                    readiness_options=['Not assessed','Ready stance visible','Extra adjustment visible']
                    readiness=st.selectbox('Visible readiness after the movement',readiness_options,index=readiness_options.index(old.get('readiness','Not assessed')),key=f'readiness_{key}')
                    note=st.text_input('Review note',old.get('note',''),key=f'foot_note_{key}')
                    reviewer=st.text_input('Footwork reviewer',saved_review.get('reviewer',''),key=f'foot_reviewer_{run.name}')
                    st.caption('Leave uncertain timing empty. Negative landing-minus-hit means landing before the opponent hit; there is no universal timing target. Confirm or reject every candidate in a rally before that rally supports a footwork pattern.')
                    if st.button('Save footwork event review',disabled=not reviewer.strip() or confirmation=='Not reviewed',key=f'save_foot_{key}'):
                        try:
                            labels[event_id]={'event_id':event_id,'confirmed':confirmation=='Confirm','opponent_hit_s':hit,'landing_s':landing,'ready_s':ready,'readiness':readiness,'note':note}
                            updated={'evidence_sha256':foot['evidence_sha256'],'reviewer':reviewer.strip(),'events':list(labels.values())}
                            reviewed_footwork(foot,updated,foot['intervals'])
                            if saved_review and not matching:write_json(run/f"footwork_review_previous_{saved_review.get('evidence_sha256','unknown')[:12]}.json",saved_review)
                            write_json(review_path,updated)
                            from shuttlecoach.coaching import write_coaching_report
                            with st.spinner('Rechecking evidence and practice gates…'):write_coaching_report(run)
                            st.rerun()
                        except (ValueError,OSError,KeyError) as error:st.error(str(error))
                else:st.info('No sufficiently visible pose events were found. Rejected or estimated positions are never used to fill footwork evidence.')
        st.subheader('Short occlusions · estimated motion')
        st.write('A separate preview can continue a missing position for up to 0.3 seconds using recent observed motion. It stops at cuts and rally changes. Estimates never replace detections or earn validation credit.')
        if st.button('Generate short-gap motion preview',disabled=active or not (run/'tracks.duckdb').exists()):
            log_file=ROOT/'.cache'/f'motion-{run.name}.log'
            with log_file.open('w') as log:
                process=subprocess.Popen([sys.executable,'-u','-m','shuttlecoach.cli','motion-preview',str(run)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            st.session_state['process'],st.session_state['log']=process,str(log_file)
            st.session_state['process_completion_refreshed']=False
            st.rerun()
        if motion:
            st.caption(f"{motion['estimated_positions']} estimated positions · maximum horizon {motion['maximum_horizon_s']:.1f}s. Select Short-gap motion preview under Playback above.")
            diagnostic=motion['diagnostic']
            if diagnostic['median_difference_m'] is not None:
                st.write(f"Masked-observation diagnostic: median difference **{diagnostic['median_difference_m']:.2f} m**, 90th percentile **{diagnostic['p90_difference_m']:.2f} m**, on {diagnostic['predicted_positions']} of {diagnostic['masked_positions']} withheld tracker positions.")
            st.caption(diagnostic['limitation'])
        with st.expander("Correct rally boundaries or athlete names"):
            st.write("Corrections use cached detections within the tracked interval. Saving creates a new analysis with validation pending. Court changes require a new tracking run because the court also filters detections.")
            corrected=st.data_editor(pd.DataFrame(manifest["config"]["rallies"]),num_rows="dynamic",
                                     hide_index=True,key=f"correct_{run.name}")
            if st.button("Save corrections from cached tracks"):
                try:
                    revised=revise_run(run,corrected.to_dict("records"))
                    st.session_state["selected_run"]=str(revised)
                    st.rerun()
                except (ValueError,TypeError,KeyError) as error:
                    st.error(str(error))
        st.subheader("Annotate the frozen validation sample")
        st.write("Label foot positions on the original frames before consulting predictions. Review the overlay separately to mark whether the near/far identity is correct. Complete all sampled frames; failed or missing detections stay in the evaluation.")
        template = pd.read_csv(run / "reference_template.csv")
        labels_path = run / "reference_labels.csv"
        saved_labels=labels_path if labels_path.exists() else run/'reviewed_references.csv'
        labels = pd.read_csv(saved_labels) if saved_labels.exists() else template.copy()
        labels[["visible","identity_correct"]] = labels[["visible","identity_correct"]].astype(object)
        meta = manifest["video"]
        sample_frames = template.source_frame.unique().tolist()
        selected_frame = st.select_slider("Reference frame", options=sample_frames, key=f"ref_frame_{run.name}")
        side = st.radio("Player to label", ["near","far"], horizontal=True, key=f"ref_side_{run.name}")
        label_key = f"{run.name}_{selected_frame}_{side}"
        row_index = labels.index[(labels.source_frame==selected_frame)&(labels.side==side)][0]
        image = frame_image(meta,selected_frame)
        saved=labels.loc[row_index]
        previous_point=None
        if pd.notna(saved.image_x) and pd.notna(saved.image_y):
            previous_point={"x":float(saved.image_x),"y":float(saved.image_y)}
            image=image.copy()
            cv2.circle(image,(round(previous_point["x"]),round(previous_point["y"])),5,(255,190,30),2)
        point = click_image(image, f"reference_{label_key}")
        point=point or previous_point
        saved_visible=str(saved.visible).strip().lower() in {"true","1","1.0"} if pd.notna(saved.visible) else True
        saved_identity=str(saved.identity_correct).strip().lower() in {"true","1","1.0"} if pd.notna(saved.identity_correct) else False
        visible = st.checkbox("Player's ground-contact position is visible", value=saved_visible, key=f"visible_{label_key}")
        identity = st.checkbox("Near/far selection follows the correct athlete in the overlay", value=saved_identity, key=f"identity_{label_key}")
        rally=next(r for r in manifest['config']['rallies'] if r['start_s']<=meta['times'][selected_frame]<r['end_s'])
        st.caption(f"Original source {source_seconds(meta,selected_frame):.3f}s · frame {source_frame_number(meta,selected_frame)} · {rally[f'{side}_player']} ({side}) · Click the estimated foot ground point. No prediction is drawn here.")
        if point:
            st.caption(f"Reference pixels: ({point['x']:.1f}, {point['y']:.1f})")
        if st.button("Save this player reference", disabled=visible and point is None):
            labels.loc[row_index,["visible","identity_correct"]] = [visible,identity]
            labels.loc[row_index,["image_x","image_y"]] = [point["x"],point["y"]] if visible else [None,None]
            labels.to_csv(labels_path,index=False)
            st.success("Reference saved.")
        st.caption(f"{int((labels.visible.notna() & labels.identity_correct.notna()).sum())}/{len(template)} player references saved")
        st.download_button("Download reference CSV", labels.to_csv(index=False), "references.csv", "text/csv",on_click="ignore")
        uploaded_labels = st.file_uploader("Or import a completed reference CSV",type="csv",key=f"refs_{run.name}")
        st.subheader("Check calibration on unused line intersections")
        st.write("Enter at least four distinct painted-line intersections spanning width and depth. Use source pixels and known court coordinates in metres (x: 0–6.10, y: 0–13.40). Exclude every point used to fit this calibration (corners or intersections). Maximum permitted error: 0.25 m.")
        saved_checks=json.loads((run/'reviewed_calibration_checks.json').read_text()) if (run/'reviewed_calibration_checks.json').exists() else []
        checks = st.data_editor(pd.DataFrame(saved_checks,columns=["image_x","image_y","court_x_m","court_y_m"]),
                                num_rows="dynamic", key=f"checks_{run.name}", hide_index=True)
        if st.button("Evaluate validation", type="primary"):
            try:
                review_dir = run / "submissions"
                review_dir.mkdir(exist_ok=True)
                submitted_refs = review_dir / "references.csv"
                submitted_refs.write_bytes(uploaded_labels.getvalue()) if uploaded_labels else submitted_refs.write_text(labels.to_csv(index=False))
                checks_path = review_dir / "checks.json"
                write_json(checks_path,checks.to_dict("records"))
                validate_run(run,submitted_refs,checks_path)
                st.rerun()
            except Exception as error:
                st.error(str(error))
        if validation["status"]!="pending":
            with st.expander("Detailed validation measurements"):
                st.json(validation)
        st.download_button("Download report", (run / "report.md").read_text(), "shuttlecoach-report.md", "text/markdown",on_click="ignore")
        if st.button("Prepare analysis ZIP"):
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer,"w",compression=zipfile.ZIP_DEFLATED) as archive:
                for artifact in run.rglob("*"):
                    if artifact.is_file() and artifact.suffix in {".csv",".json",".png",".md",".mp4"}:
                        archive.write(artifact,str(artifact.relative_to(run)))
            st.download_button("Download analysis ZIP",buffer.getvalue(),f"shuttlecoach-{run.name}.zip","application/zip",on_click="ignore")
