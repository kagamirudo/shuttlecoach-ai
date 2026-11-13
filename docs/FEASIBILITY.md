# Phase 0 — feasibility and reuse decision

Primary repositories reviewed 2026-11-11. Repository promises are not validation
of this oblique club recording. Only the selected adapters have been executed here.

| Candidate | Reusable component | License / weights | Local feasibility and decision |
|---|---|---|---|
| [BadmintonAI / COURTSIDE](https://github.com/jayclim/BadmintonAI) | YOLO pose, ByteTrack, court transform | MIT project; Ultralytics model has separate AGPL terms | Selected, pinned player subset; Apple MPS inference verified on this machine. Club footage needs manual corners and local identity recovery. |
| [BaddieVision](https://github.com/MaxLinCode/BaddieVision) | Tracking, in-play and diagnostics | MIT project; private weights absent from checkout | Research prototype; MPS fallback documented, this stack untested locally. Avoid another full pipeline. |
| [Badminton-Tracker](https://github.com/m11ahmed/Badminton-Tracker) | TrackNet, pose, Streamlit | MIT project; model dependencies need separate review | Useful reference; documented Windows/CPU setup does not establish M1 performance. Not selected. |
| [RallyLive](https://github.com/JWM0203/RallyLive) | Venue ball tracking, RTSP/OBS | Apache-2.0 project; release model availability advertised | Windows/NVIDIA-oriented venue workflow; Apple setup untested. Infrastructure exceeds this local MVP. |
| [TrackNetV3](https://github.com/qaz812345/TrackNetV3) | Temporal shuttle heatmaps and official checkpoint | MIT project and published official checkpoint | Selected, pinned source; local MPS adapter verified. Keep real components, weighted overlap; no gap inpainting. |
| [RallyLens](https://github.com/YeonSeong-Lee/rallylens) | CLI/browser analytics and report design | MIT project; YOLO has separate AGPL terms; browser model manifest incomplete in reviewed version | More browser/model/LLM pieces than needed; not verified locally. Not selected. |

## Concrete implementation

BadmintonAI revision `5c356f05cde0ed90c75316c133e0aba495e5232b` supplies
player detection, skeletons, ground-point conversion and court rendering. TrackNetV3
revision `6eda442ada1740573f200f836d93edc9a541ee86` supplies the pretrained
shuttle network and temporal preprocessing. Vendored source stays unchanged.
Local code handles timestamps, manual calibration, target association, reference
review and evidence gates. Existing weights are reused; no training or cloud service.

The player model is YOLO11m-pose at 1920 pixels, approximately 10 sampled frames/s.
The quality shuttle adapter samples 30 real frames/s and weights overlapping
8-frame predictions. All runs preserve source presentation timestamps. The
five-minute runs demonstrate local execution; they do not establish live speed.
Weights are checksum pinned by the download scripts. See README and requirements
lock for the exact reproducible environment and licensing limitations.

## Practical boundary

A camera image cannot uniquely assign a birdie to a court from its floor footprint:
lofts can appear above another court. The widened calibrated airspace therefore
needs temporal and target-player association, plus manually reviewed activity.
Image-to-floor projections are visualization only, not height-corrected trajectories.
Manual play boundaries are permitted; automatic rally/shot labels are outside MVP.
The current same-clip development audit is not a held-out generalization benchmark.
