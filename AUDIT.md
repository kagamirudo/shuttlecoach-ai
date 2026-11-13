# MVP completion audit

Final run: `results/20261111T224529051246Z`. Approved scope: the original
brief's Phases 0–2, birdie tracking, expanded court airspace, five minutes of
actual play, and coaching enabled only after both accuracy gates pass.
Acceptance thresholds were not relaxed.

| Stage | Evidence and outcome |
|---|---|
| Phase 0 | `docs/FEASIBILITY.md` compares six proposed repositories, weights, licenses and Mac feasibility. BadmintonAI and TrackNetV3 adapters executed locally on MPS; other candidates were researched, not benchmarked locally. |
| Ingestion | Complete original `data/RZaZ7GaPJU8.mp4` retained: 29:57.9 with audio. Selected analysis windows preserve the original frame/time map. |
| Calibration | Four manually reviewed outer corners; five separate painted-line checks, maximum projected error 0.233 m, all below 0.25 m. |
| Phase 1 / players | 146 fixed sample frames; 142 usable player pairs (97.3%). Median projected foot error 0.131 m, p90 0.426 m. Four far-player gaps remain; no accepted identity errors in reviewed samples. |
| Birdie | Same 146 sampled frames: 84 visible, 62 not visibly separable. 73 correct points, 8 false candidates, 11 misses; precision 90.1%, recall 86.9%, median matched error 1.41 px. Both unchanged gates pass. |
| Trajectories | Real observations and source timestamps retained. Sample cadence and cut-aware trail rendering corrected. Floor-homography projection is an image proxy; airborne image points outside the floor retain no floor coordinates. |
| Actual play | 62 manually reviewed intervals/fragments total 308.6 seconds. Setup and retrieval excluded from coaching. Partial fragments excluded from recurring-pattern support. |
| Phase 2 | Saved references and hashes re-evaluated before enabling coaching. Gary-specific middle-third pattern supported by nine complete intervals and 60.8 accepted seconds, with original-video evidence links and a conditional drill. |
| Delivery | Local dashboard, playable overlays, full original source, timestamped CSVs, report and analysis ZIP. Final playback/export checks are recorded in `audit/delivery_checks.json`. |

## Revisions that mattered

Earlier passes were insufficient: a 20-second preview did not satisfy the five-minute
request; broad activity windows included preparation/retrieval; the high-shot zone
included neighboring-court detections; far-player appearance was mixed across end
changes; quality-mode projection trails assumed consecutive source frames.

Player recovery now selects actual poses using athlete/half-specific shirt evidence
and temporal anchors. The birdie zone keeps the full floor and raised target corridor,
while restricting the lower neighboring-court wedge. Temporal/wrist association and
compact white source foreground reject some racket, clothing and background candidates.
No reference coordinates are used to manufacture detections or fill gaps.

The replacement analysis combines three inference inputs. Every one of 34,998 decoded
8-bit YUV frames was checked against its child input before genuine inference reuse;
`assembly.json` and manifest lineage record the proof. Two unprocessed join-context
frames stay empty. Inputs are 8-bit conversions, not identical RGB copies of the
10-bit original. A 2.2-microsecond assembled-clock boundary mismatch initially blocked
report generation; endpoint mapping was corrected rather than loosening the gate.

## What this pass establishes

This is a same-recording development audit on one fixed singles camera, with manual
court calibration and approximate rally boundaries reviewed at 0.5-second cadence.
It is not an untouched held-out benchmark. The original 100 samples remain; 30 early
and 16 tail samples were frozen before their additional inference. Source-only feet
and birdie points were recorded first; identities were reviewed separately. Later
model-guided source-pixel corrections to birdie labels are explicit, and historical
label versions are retained in `audit/`. In particular, tail sample B12 was corrected
following dense temporal review of a projectile crossing fixed net hardware.

Eight false birdie candidates, eleven visible misses and four unusable player pairs
remain in evaluation. Passing does not prove accuracy on another recording. Neighboring
court overlap, contact blur, occlusion and uncertain tiny targets remain limitations.
Manual labels and boundaries are fallible. Overlays contain selected discontinuous
windows, approximately 9:41 with context, not every minute of the original match.

Coaching is descriptive and conditional. Middle-third occupancy alone does not establish
a weakness, slow recovery or an optimal base. No automatic shot labels, landing points,
physical birdie floor trajectory, 3D flight, speed or biomechanics are claimed.

## 2026-11-13: footwork review and separate forecasts

Added observed-pose movement candidates, explicit hit/landing/readiness review,
and evidence-linked practice cards. Footwork candidates remain unvalidated
heuristics; confirmed event reviews and repeated sufficiently covered rallies are
required in addition to the unchanged player/birdie gates. Stale report inputs
hide advice until the report is refreshed. No real event confirmations were
invented to demonstrate these new paths.

A separate past-only regression preview predicts at most 0.3 s. Current source
`-mxdSIBuGiU` has 230 estimated positions; the original observed tracks and failed
84/100 player-pair gate are unchanged. On 426 artificially withheld tracker
coordinates, 366 estimates had median/p90 differences 0.292/0.860 m. This is a
diagnostic against tracker output, not independent truth for hidden feet.
Preview timing matches all 3,750 source samples exactly. Long gaps stay empty.

149 tests passed, including gated/stale AppTest practice-card paths, reviewed
footwork timing, and temporal leakage/cut/athlete-change regression checks.
See `docs/HANDOFF.md` for current runs, commands and remaining evidence work.

Optional Groq wording integration was added on 2026-11-13 with the user-authorized
Free plan use case. It sends anonymized summaries, not video/names; local evidence
is rechecked and failed gates permit only data-quality narration. Keys are never
written to reports. 158 tests passed, including mocked request/cache/quota paths.
No key/account was provisioned and no live provider call was claimed.

### Inline AI analysis, superseding the initial wording-only draft

The initial request supplied too little movement evidence and the saved report
returned one sentence. The new main-page AI section shows per-player findings,
coverage/dominant-zone/missing-time metrics, and source-measurement/video expanders.
AI connection settings are separate; no report download is required or offered.

The provider receives anonymous aggregate coverage, accepted-zone percentages,
thirty-second windows, footwork candidates and reviewed timing, rather than just
validation verdicts. Failed gates permit descriptive measurement analysis while
practice remains bound to approved evidence. Same-player citations and copied
numeric measurements are checked. Candidate-count interpretations are replaced
with an explicit video-review requirement, never a claim of true movement quality.

Groq GPT-OSS 120B with strict JSON Schema successfully generated six findings for
the current recording, using the existing saved key without exposing/changing it.
No video or player names were sent. Local tracking artifacts/gates are unchanged.
The old one-line report remains in `audit/before-ai-analysis-v2/`; report cache
version is now 3. Actual inline rendering was exercised with AppTest. 178 tests
passed; manual browser appearance and automatic video-based technique grading
remain outside this verification.

## Docker distribution

Docker/Compose package uses a CPU-only, non-root Python 3.12 image and persists
only the four local runtime folders. The host working path is preserved so saved
absolute source references remain valid without rewriting validation inputs.
Groq credentials are supplied as runtime secrets; media, weights, results, local
keys and upstream Git metadata are excluded from build context and publication.
Native and Linux-container suites each passed 184 tests. Container health,
non-root execution and absence of the API key in its environment were verified.
Docker packaging does not constitute a new tracking-quality or biomechanics pass.
