# Current project handoff

Updated 2026-11-13. Implemented and tested: video-assisted footwork candidate
analysis/review, clear gated practice cards, and a separate short-gap regression
preview. No neural model was trained. Footwork candidate accuracy and long
occlusion recovery have not been independently validated.

## Current footage and result

- Source: `data/-mxdSIBuGiU.mp4`, YouTube `-mxdSIBuGiU`, 1280×720, ~60 fps.
- Selected interval: 0–375 s (6:15). Names: Shi Yu Qi near, Random China far;
  these are user labels, not independently verified identities.
- Current result: `results/20261112T211353004692Z`.
- Baseline: `results/20261112T200933831294Z`, preserved.
- Far availability improved 49.4% → 77.9% (292.2/375 s), near 99.1%.
- Frozen 100-frame agent visual audit: 84% usable pairs, gate requires 95%.
  14 far-foot references occluded, two visible far-player misses. No retained
  reviewed identity errors. Median foot error 0.304 m; p90 0.806 m. Four unused
  near-half court checks max 0.089 m. Far-half projection lacks independent checks.
- Gate FAILED. Birdie/rally validation not completed for this source; no coaching.
- `audit/validation_summary.md`, `audit/provenance.json`, reviewed CSVs and source
  references preserve the method. This is same-recording development review,
  not an untouched held-out benchmark or physical motion-capture truth.
- Actual YOLO pose recovery uses appearance, real temporal anchors and enlarged
  far crops. Assigned/trusted poses are reserved so one athlete cannot fill both
  slots. No synthetic positions enter the raw database or observations.
- Overlay has 3,750 samples, zero timestamp mismatch. Full source with audio is
  separate from the silent analyzed overlay. App offers both playback choices.

## Previous passing demonstration

`results/20261111T224529051246Z` is the unrelated Leon/Gary source
`data/RZaZ7GaPJU8.mp4`. It assembles 308.6 s of manually reviewed play with context.
Both gates passed on 146 frozen samples. Do not mix these labels/calibration with
the current source. README documents its measured scope and audit limitations.

## Entry points and checks

- `app.py`: Streamlit prepare/review, validation, playback, readable coverage bars.
- `shuttlecoach/player_recovery.py`: real-pose reacquisition; CLI recover-players.
- `shuttlecoach/coaching.py`: rechecks source/artifact hashes and saved references,
  then gates patterns/drills. `make report RUN=...` regenerates the report.
- `shuttlecoach/validation.py`: fixed 95% player-pair gate; no invented labels.
- `shuttlecoach/timeline.py`: original clocks and cuts for assembled source.
- `make run`, `make test`, `make help`. Python `.venv/bin/python`, Python 3.12.
- Last completed suite: **184 tests passed natively and in Docker** (`.cache/docker-native-tests.log`, `.cache/docker-tests.log`).
  Includes AppTest checks that practice cards appear for current passing reports
  and disappear for blocked/stale reports, footwork timing/review boundaries,
  regression past-only behavior, cuts/end changes, and VFR preview timestamps.
- Mac GPU can require execution outside the sandbox; MPS is unavailable inside.
  Actual prior MPS recovery took about six minutes. No GPU needed for cached-pose
  analysis, report generation or regression.

## Footwork and regression implementation

- `shuttlecoach/footwork.py`: confidence-filtered 2D pose features and event
  candidates. Missing poses, cuts and rally/athlete boundaries break sequences.
  No physical balance, optimal-base or 3D joint-angle grades are claimed.
- Split-step landing-minus-hit and lunge-to-ready timing require explicit source
  annotations in the app. Ready-stance/extra-adjustment labels are reviewer
  observations. Candidate event thresholds are heuristics, not validated labels.
- `footwork_review.json` binds to an evidence hash. Every candidate in a rally
  must be confirmed/rejected before that rally supports footwork patterns. At
  least three eligible rallies, each >=5 accepted seconds and >=95% coverage,
  with >=30 accepted seconds combined, are needed. Partial rallies are excluded.
- `shuttlecoach/coaching.py`: footwork suggestions only after both tracking gates;
  readable app cards show repeated pattern, practice and original-video evidence.
  Evidence signatures prevent displaying stale advice after inputs change.
- `shuttlecoach/motion.py`: past-only least-squares linear regression, 0.6 s
  history, >=4 real samples, maximum 0.3 s horizon, fit RMSE <=0.25 m and fitted
  speed <=8 m/s. No recursive prediction or future detections are used.
- Current clip preview has **230 estimated positions**. Masked diagnostic:
  366/426 withheld tracker coordinates predicted; median difference 0.292 m,
  p90 0.860 m. This is tracker agreement, not manual truth or occlusion accuracy.
- `motion_preview.mp4`: 3,750 frames, zero timestamp mismatch. Raw observations,
  database and validation credit are unchanged. Yellow hollow markers are
  estimates; longer gaps remain empty. `motion.json` records limitations/hashes.
- `make footwork RUN=...`, `make motion-preview RUN=...`, `make report RUN=...`.
  New controls also appear in Review & validate. Refresh/rerun the app to load
  code changes; avoid restarting an active calibration session unnecessarily.
- Both the current failing recording and earlier passing demonstration had their
  reports regenerated. Historical demo report/manifest copies are retained under
  `audit/before-footwork-ui/`. No real footwork candidate reviews were fabricated.

## Remaining evidence and future work

### Optional LLM wording (2026-11-13)

`shuttlecoach/llm_report.py` and `report_ui.py` now provide inline **AI analysis**
above birdie/review controls. Click **Analyze tracking and footwork**; credentials
live under **AI connection settings**. There is no AI-download button or required
file workflow. Source report JSON/Markdown remain internal/local CLI artifacts.

Current default is Groq `openai/gpt-oss-120b`, listed in the provider's Free plan
limits. The smaller model's unsupported summaries were rejected during development;
the larger model uses strict JSON Schema output plus local evidence validation.
Get the user's own key from https://console.groq.com/keys; never create an account,
print a key, enable paid billing, or ask for secrets in chat. Quotas/access can change.

The old writer saw only validation verdicts and approved drill text. New input
contains anonymized per-player weighted coverage/occupancy, thirty-second source
windows, visible gaps, unconfirmed footwork counts, and reviewed timing when present.
Current recording supplies 30 evidence facts for two players. No video, raw player
names, local paths, reviewer names, free-form notes, or forecast coordinates are sent.
Local labels are restored in UI only. Local raw tracks and validation are unchanged.

Failed tracking gates still permit DESCRIPTIVE analysis of accepted measurements,
not automatic technique diagnoses or personalized practice. Practice sections must
exactly match locally approved suggestions and remain gated. Every AI finding must
cite same-player fact IDs; numeric values must occur in those cited measurements.
Wrong-player citations, invented values, links, and whole-match denominator claims
are rejected. Unconfirmed footwork findings are replaced with a deterministic video-
review requirement: detector counts cannot prove step rates, quality, or activity
between athletes. Local code renders trusted coverage/dominant-zone/missing-time
metrics and reliability context; semantic AI interpretations can still need review.

Report cache version is **3**; update it when input/output/policy changes. The old
one-line cache is preserved under `audit/before-ai-analysis-v2/`. Current report was
successfully generated using the existing Keychain key through Groq GPT-OSS 120B:
**two named player cards, six findings, 4,956 provider tokens**. The API key was
never printed, uploaded as evidence, written to reports, or changed by the agent.
No new account/key was provisioned. Earlier malformed/unsupported drafts were
rejected, without automatic retries; only the valid result was saved.

Actual saved-result UI rendering was checked with Streamlit AppTest:
`.cache/verify_ai_ui.py`, log `.cache/ai-ui-actual.log`. Six findings appear directly
on the main page, with measurement/video expanders; settings remain collapsed and
no AI download control exists. **178 tests passed** including grounded numbers,
wrong-player citations, stale/legacy caches, candidate-count interpretation, inline
UI display, credentials isolation and API/quota behavior. UI tests now also block
native Keychain access to prevent accidental real credential changes.

Groq 403 diagnosis: the default `Python-urllib/3.12` User-Agent reproduced a
Cloudflare/gateway 403 on an unauthenticated GET `/openai/v1/models`; the honest
`ShuttleCoach/0.2` header returned the expected 401 (no key supplied). Both checks
ran on this Mac with network access, without credentials or video. Requests now
set User-Agent and Accept explicitly. Error bodies are classified using known
org/project permission codes versus gateway blocks; raw bodies are never echoed.
Unknown 403 responses no longer assert model denial. This initial diagnostic
used no key; the later live inline-analysis test used the saved Keychain credential.
An unauthenticated POST to the actual completion endpoint with the corrected
headers also returned expected HTTP 401, confirming it reaches authentication.
163 tests passed, including gateway/org/project/unknown 403 classification and
credential-redaction cases. In the app, use Main menu → Rerun to retain session
inputs while applying changed code, then retry Generate AI wording draft.

### Remembered API keys (2026-11-13)

The user requested persistence after losing a session-only key. `credentials.py`
uses macOS Security/CoreFoundation through stdlib ctypes, with service
`ShuttleCoach Groq API`, account `groq-api-key`. No dependency, plaintext secret
file, shell argument containing a key, or report embedding is used. Resolve
precedence: explicit entry → GROQ_API_KEY → this app's Keychain item.

In the app, **Save API key on this Mac** saves without making an API request.
**Remember API key on this Mac** defaults on and saves entered keys before
generation, even if Groq rejects the subsequent request. A saved key is reused
automatically, without preloading it into the browser's password input. Native
load queries suppress authentication popups; locked/denied access is reported.
Non-Mac systems use session/environment credentials with no file fallback.

Native save/load/update/delete was verified using an isolated random-service
dummy credential, which was then removed. Later API analysis used the existing
user credential internally without exposing or changing it. Existing lost browser sessions cannot be recovered; the user
must enter their key once if no session/environment/Keychain copy remains.
169 tests passed, including explicit/env/Keychain precedence, invalid-token
rejection, no plaintext fallback, and AppTest saving/reusing a key in a fresh
session. UI tests use mocked Keychain storage; native testing used only dummy data.

Current recording still has a failed player gate, no validated birdie tracking,
and a continuous selected interval needing actual-rally review. Forecasts do not
solve that evidence deficit. Footwork candidate precision/recall and confirmed
timing need source-video annotation and evaluation on additional recordings.
Longer occlusions, learned motion models and automatic contact/landing detection
remain future work. Browser appearance was not manually rechecked in this pass;
the UI paths were exercised with Streamlit AppTest. Preserve these limits when
continuing elsewhere; do not describe this prototype as full automatic coaching.

## Docker packaging and GitHub (2026-11-13)

- Preferred packaged command: `make docker-up`; dashboard http://127.0.0.1:8502.
  Native `make run` at 8501 remains available for Mac MPS and existing sessions.
- Multi-stage Debian/Python 3.12 image, pinned CPU Torch/Torchvision, Linux
  `watchdog` lock marker, non-root runtime matching host UID/GID, loopback-only
  port, healthcheck and model-checksum startup verification.
- `data/`, `models/`, `results/`, `.cache/` are persistent host bind mounts, never
  build context. Matching the native absolute application path preserves current
  source/config/model/map references without modifying frozen evidence.
- The host Make launcher supplies a saved Keychain/environment key through a
  Compose runtime secret. Containers read `GROQ_API_KEY_FILE`; the key is not a
  build argument or container environment value. Linux hides native Keychain
  saving controls; session input remains usable. Tests run without API secrets.
- `make docker-down`, `docker-logs`, `docker-status`, `docker-build`, `docker-test`.
  Host prerequisites: Docker/Compose, Make, Python 3. No native venv is needed for
  the launcher. Standard CPU Docker is slower than native Mac MPS.
- Actual Docker image built, service healthy, HTTP health endpoint verified,
  non-root execution and secret availability checked without revealing the key.
  Container tests: **184 passed**. Native tests: **184 passed**. CPU checkpoint
  smoke test: YOLO inferred one real frame; TrackNet loaded pinned weights and
  completed a forward pass of shape `(1, 8, 32, 64)`. Existing source paths were
  readable and the Groq runtime secret was available. These are compatibility
  checks, not a new tracking-accuracy pass. Log: `.cache/docker-smoke.log`.
- Documentation reorganized: root README is quick start; `docs/DOCKER.md` covers
  operations, `USAGE.md` preserves native/calibration/CLI details,
  `DEMONSTRATION.md` scopes historical metrics, `DEVELOPMENT.md` maps code/tests.
- The user explicitly requested pushing to GitHub under `kagamirudo`. CLI auth
  matches that account; the connected GitHub plugin uses another account, so use
  `gh` CLI. Repository: public `https://github.com/kagamirudo/shuttlecoach-ai`. Do not
  upload media, models, results, caches, credentials or upstream `.git` folders.
- Root Git repository initialized with source files tracked as regular files,
  including pinned vendored TrackNet source (no submodule gitlink). Nested vendor
  metadata remains untouched and excluded. A source allowlist/credential-pattern
  check found 123 files, approximately 3.53 MiB, with no runtime private artifacts.
  Initial source commit `5b8b2c0b6fbf82636bb4495a7f81c81e51dba356` was pushed to
  `origin/main` and verified through GitHub's API. The repository was initially
  private and was made public at the user's explicit request on 2026-11-13.
  Working-tree runtime files remain ignored. Future source changes should use
  this root repository; do not initialize inside a vendored upstream directory.

## Public installation guide (2026-11-13)

- User explicitly authorized public GitHub visibility. Remote HEAD matched the
  reviewed local history before the visibility change; GitHub confirmed PUBLIC.
- A source/history review checked 123 tracked files and 122 historical blobs.
  No private media, model weights, generated results or API-key/private-key
  patterns were found. No runtime folders were added by this documentation change.
- `docs/INSTALL.md` now covers prerequisites, Mac/Linux/WSL setup, clone/build,
  health checks, video import, calibration, inline AI, key persistence, updates,
  shutdown and troubleshooting. README links it prominently.
- Five original schematic SVG illustrations live under `docs/assets/`. Their
  text/controls match the workflow and are explicitly labelled as illustrations,
  not screenshots or real tracking results. They contain no real footage/keys.
  All five were rasterized and visually checked; local Markdown links and SVG
  XML were checked. SVGs are editable directly; previews are not committed.
- Tested installation scope remains Apple-silicon Mac + Linux ARM64 container.
  Intel/AMD64 and Windows/WSL instructions are documented but not exercised here.
  No application code/dependency changed in this documentation-only update;
  the latest functional suites remain 184 passing native and container tests.

## Dashboard redesign (2026-11-13)

- Follow-up: tab underlines sit 0.45rem below the selected pill, with extra
  tab-strip bottom padding so the line no longer touches its edge (CSS only).

- `app.py` now has three preparation tabs (Footage, Players & time, Court &
  tracking) and five review tabs (Overview, AI & practice, Birdie, Footwork,
  Validation). All tab contents remain mounted to preserve widget/calibration
  state. Tracking workers and evidence/quality-gate logic are unchanged.
- Overview pairs playback with a smaller court map, followed by bordered player
  summaries. Shared cards show player/birdie validation and selected duration.
  Run choices use UTC dates; local paths/raw measurements sit in expanders.
- Upload is the primary local input; **Use a file path instead** exposes the path.
  The load button remains enabled so an unapplied text edit cannot trap a user
  behind a disabled control. Time columns have readable labels and seconds help.
- Completed tracking can open its selected analysis directly in Review & validate;
  the worker progress fragment is shared across both workspaces.
- `.streamlit/config.toml` provides the light green theme; `shuttlecoach/ui.css`
  styles native Streamlit controls without new dependencies or changing click
  coordinates. Docker explicitly includes only the config file, never secrets.
- Native **184 passed** (`.cache/ui-tests.log`); rebuilt Docker **184 passed**
  (`.cache/ui-docker-tests.log`). AppTest now checks tab placement, empty start,
  current/blocked/stale practice cards, rerun state and opening completed jobs.
- Browser review checked the current recording's overview/inline AI, file-path
  loading and separate timing/calibration steps at actual 614px and 816px widths.
  No overflow of page bounds was observed. Chrome's existing Dark Reader
  extension recolors the page; neutral appearance was checked in the in-app browser.
  Requested viewport overrides did not change actual widths, so 390/1440px layouts
  are not claimed tested. Full calibration/export browser scripts were updated
  for the new navigation but were not rerun in this pass.
- Docker is rebuilt and healthy on 8502. No tracking rerun, accuracy improvement,
  fresh AI API request or new coaching validation is claimed. Current source
  remains failed/pending as documented above. Usage/install docs and two schematic
  illustrations were updated; both illustrations were rendered and checked.
