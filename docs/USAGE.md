# Using ShuttleCoach

For a fresh Docker installation, use the [illustrated setup guide](INSTALL.md).

Run commands from the repository root. Docker operations are covered in
[DOCKER.md](DOCKER.md). This guide also documents native Mac execution for MPS.

## Fresh installation

Use native ARM Python on Apple Silicon:

```sh
make setup
make run
```

`make setup` requires `uv`, creates the Python 3.12 `.venv` if needed, syncs
`requirements-lock.txt`, then downloads missing models and verifies their pinned
checksums. `make deps` and `make models` run those steps separately.

The downloaders verify pinned model checksums. FFmpeg is supplied by
`imageio-ffmpeg`; PyAV decodes and encodes source timestamps. MPS was exercised on
this Mac; `auto` falls back to CPU. This is offline processing, not a live-speed
claim. Restricted YouTube downloads use local MP4/MOV fallback; no access controls
are bypassed. A permitted public download of the supplied singles URL succeeded.

## Project commands

Run `make` or `make help` for the full command menu and required arguments.

| Command | Purpose |
|---|---|
| `make setup` | Install locked dependencies and fetch/verify both models |
| `make run` | Start the dashboard; stop with Ctrl+C |
| `make run PORT=8502` | Use a different port; `HOST` defaults to `127.0.0.1` |
| `make doctor` | Check dependencies, Python architecture, MPS availability and model presence |
| `make test` | Run the test suite |
| `make check` | Run doctor, then tests |
| `make test-calibration-ui` | Exercise cropped-court calibration in installed Chrome |
| `make test-ui VIDEO="data/clip.mp4" RUN="results/RUN_ID"` | Exercise dashboard playback and exports in installed Chrome |
| `make clean` | Remove Python/pytest caches; keep footage, models, results and `.cache` provenance |

Browser checks require a running dashboard and installed Google Chrome. Pass the
same `HOST` and `PORT` to the app and browser-check targets when overriding them.
`make doctor` reports model presence; `make models` verifies their checksums.
Tracking commands accept `DEVICE=auto`, `DEVICE=cpu` or `DEVICE=mps`.

## Analyze another clip

1. Supply an authorized YouTube URL or local path. Playlist parameters are removed.
2. Mark rally intervals and distinct athlete names, including end changes. Each
   run supports a span of up to ten minutes, including retained context or gaps.
   Warm-up, setup and retrieval need separate review. After a run finishes, edit
   the times or player names under **Prepare a clip** and run again to create a
   separate result. Calibration stays saved for the same source frame and method;
   keep that frame inside the selected interval. The earlier result is preserved.
3. Choose a calibration method. **Four outer corners** requires all four physical
   doubles-court corners to be visible: near-left, near-right, far-right, far-left.
   For a cropped court, choose **Visible line intersections**, select a numbered
   intersection from the court reference, and click its matching painted crossing
   in the video. Mark at least four distinct matches spread across width and depth
   (at least two metres in each direction); six or more is preferable. The net and
   image-edge line endpoints are not floor intersections. Near is the camera-side
   baseline, shown at the bottom of the reference. Confirm that the projected green
   lines align, then use **Download reviewed configuration** to save it. Off-screen corners are
   inferred and saved for tracking and the birdie zone. This is manual calibration;
   automatic court detection is not claimed.
4. Run players, optionally recover uncertain actual poses, then run birdie quality
   mode. Freeze and annotate the validation templates from unannotated source.
5. Supply at least four unused painted-line intersections. Evaluate both gates,
   inspect the evidence report, then download the report or analysis ZIP.

```sh
make ingest URL='https://www.youtube.com/watch?v=RZaZ7GaPJU8'
make probe VIDEO="data/RZaZ7GaPJU8.mp4"
make prepare-play REVIEW="review.json" OUT="data/selected-play"
make analyze CONFIG="data/demo-config.json" DEVICE=auto
make recover-players RUN="results/NEW_RUN" DEVICE=auto
make shuttle RUN="results/RECOVERED_RUN" DEVICE=auto
make associate-shuttle RUN="results/BIRDIE_RUN" REVIEW="activity_review.json"
make validate RUN="results/ASSOCIATED_RUN" PLAYER_REFS="player_references.csv" SHUTTLE_REFS="birdie_references.csv" CHECKS="calibration_checks.json"
make report RUN="results/ASSOCIATED_RUN"
```

Commands that produce a revised run print its new path. Reference files must
complete the exact frozen templates; selecting only successful frames is rejected.
Blank/uncertain samples cannot pass. Keep calibration checks separate from every
point used to fit the court, including fitted intersections. Fit residuals and a
green preview do not constitute independent validation.

To reproduce the **saved final gate evaluations**, without repeating inference:

```sh
make validate RUN="results/20261111T224529051246Z" \
  PLAYER_REFS="results/20261111T224529051246Z/submitted_player_references.csv" \
  SHUTTLE_REFS="results/20261111T224529051246Z/submitted_shuttle_references.csv" \
  CHECKS="data/calibration-checks.json"
make report RUN="results/20261111T224529051246Z"
make test
```

The final combined run reuses genuine inference from three child runs listed in
`analysis.json`. Every decoded 8-bit YUV frame was compared byte-for-byte before
reuse. Offsets and source maps preserve all references. Unprocessed join context
stays empty; poses, birdie coordinates and gaps are not synthesized. Child inputs
are 8-bit conversions of the 10-bit original, not exact original RGB copies.

## Footwork and short-gap previews

The app now includes **Footwork · movement review**: timestamped lunge-shape,
stance-opening and left/right foot-movement candidates from observed poses.
Review the source frames to confirm/reject candidates and mark opponent hit,
landing and ready-stance times. These support split-step timing, lunge-to-ready
intervals, step sequences and reviewed adjustment steps. They are video-assisted
measurements, not automatic biomechanics or physical balance grades.

**Patterns and practice** shows readable evidence/practice cards after player and
birdie validation pass. Footwork advice additionally needs complete candidate
reviews within at least three sufficiently covered rallies, each with at least
five accepted seconds and thirty accepted seconds combined. Partial rallies and
unreviewed candidates cannot support these suggestions. Reports are invalidated
when their source, tracking or review inputs change. Saving a review or evaluating
validation refreshes the report automatically.

**Short occlusions · estimated motion** creates a separate playback preview.
Past-only linear regression uses up to 0.6 seconds of real observations and
forecasts at most 0.3 seconds. Yellow hollow markers identify estimates. Cuts,
rally/athlete changes and unreliable fits stop prediction. Estimates never enter
the raw tracks, occupancy, validation or coaching. This is an initial regression
prototype; no neural model was trained and long occlusions remain unsupported.

```sh
make footwork RUN="results/20261112T211353004692Z"
make motion-preview RUN="results/20261112T211353004692Z"
make report RUN="results/20261112T211353004692Z"
```

Outputs include `footwork.json`, `footwork_samples.csv`, `footwork_events.csv`,
the user's `footwork_review.json`, and separate `motion_estimates.csv`,
`motion_preview.mp4` and `motion.json`. The motion diagnostic hides observed tracker
positions; agreement with those withheld coordinates is not ground-truth accuracy
on real occlusions. Original tracking quality gates remain unchanged.

## Tracking and visualization

Player tracking samples approximately 10 real frames/s; recovery selects actual
YOLO poses using shirt appearance and same-athlete temporal anchors, separated by
court half across end changes. When full-frame detection misses the small far
player, recovery also searches an enlarged source crop and maps detections back
to source pixels. A pose already assigned to the other athlete cannot be reused.
It never substitutes interpolated positions.

The review screen shows tracked/missing seconds, coverage, and front/middle/rear
position bars for each athlete. Coverage is availability, not measured accuracy;
court percentages exclude missing and off-court positions and include pauses in
the selected windows. Raw tables and CSV exports live under Detailed measurements
and CSV. Coaching stays locked until player, birdie and rally-context gates pass.
TrackNet quality mode samples approximately 30 real frames/s with overlapping
8-frame weighted predictions and retains disconnected candidate components.

The zone preserves the court floor footprint, raises the target corridor to the
image top, and allows a high upper-left strip. Temporal/wrist association and
compact white source foreground reduce neighboring-court/racket/clothing errors.
This image heuristic can still overlap another court and miss high or blurred
shots. The preview shows the exact mask. Recalibrate for a new view.

Player mini-court trails use projected feet. Birdie mini-court trails transform
image observations inside the floor footprint through the same homography.
**They are not physical birdie floor paths or 3D flight trajectories.** Airspace
points stay in the image/CSV with no floor projection. Missing detections, large
jumps and source cuts break trails. Heatmaps/raw occupancy include retained analysis
windows; the coaching report uses the stricter rally intervals.

## Research, licenses and audit

[Phase 0 comparison](FEASIBILITY.md) covers all six proposed repositories;
only BadmintonAI and TrackNetV3 adapters were executed locally. [AUDIT.md](../AUDIT.md)
maps every approved phase to its evidence and records earlier failures.

BadmintonAI revision `5c356f05cde0ed90c75316c133e0aba495e5232b` and TrackNetV3
revision `6eda442ada1740573f200f836d93edc9a541ee86` are pinned and vendored
unchanged. Their project code is MIT; YOLO/Ultralytics has separate AGPL terms.
Review upstream model terms before distribution or commercial deployment.
Official TrackNet weights are reused. No custom training, cloud GPU, automatic
BWF grading, biomechanics or Raspberry Pi integration was added.
