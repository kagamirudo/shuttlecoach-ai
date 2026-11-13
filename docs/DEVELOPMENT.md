# Development

## Entry points

| Path | Responsibility |
|---|---|
| `app.py`, `shuttlecoach/report_ui.py` | Prepare/review workflow and inline AI findings |
| `shuttlecoach/ui.css`, `.streamlit/config.toml` | Responsive workspace styling and light court-green theme |
| `shuttlecoach/runner.py`, `player_recovery.py` | Player pipeline, real-pose recovery and overlays |
| `shuttlecoach/shuttle.py`, `shuttle_association.py` | Birdie candidates, raised court zone and reviewed association |
| `shuttlecoach/calibration.py`, `validation.py` | Visible-intersection homography and frozen quality gates |
| `shuttlecoach/footwork.py`, `motion.py` | Reviewed pose events and separate short-gap regression |
| `shuttlecoach/coaching.py`, `llm_report.py` | Evidence rechecks and source-linked reporting |
| `shuttlecoach/credentials.py` | Mac Keychain and container runtime-secret loading |
| `scripts/docker_compose.py` | Docker path/ownership and secret bridge |

Read [HANDOFF.md](HANDOFF.md) and the root [AGENTS.md](../AGENTS.md) before changes.
New installation instructions and schematic SVG pictures live in
[INSTALL.md](INSTALL.md) and `docs/assets/`. Edit the SVGs directly; they contain
no scripts, private screenshots, sample footage or API keys. Keep UI labels in
the pictures aligned with the app and identify them as illustrations.

## Tests

```sh
make test                 # native environment
make docker-test          # already-built image, no API secrets
make docker-build         # build and dependency consistency check
```

Native installation uses `make setup` (uv and Python 3.12). Container dependencies
are locked in `requirements-lock.txt`; Linux also needs Streamlit's `watchdog`,
which is not installed by the Mac platform marker. CPU Torch/Torchvision come
from PyTorch's CPU index. Keep Docker's Torch pins synchronized with the lockfile.
No models, private footage or generated results belong in build context or Git.

Browser checks under `scripts/` require native Chrome and an already-running app;
the container test target runs unit/integration/AppTest checks, not Chrome UI tests.

## Packaging and evidence rules

The root Makefile keeps native targets and adds `docker-*` targets. Compose mounts
only runtime data folders; code is baked into the image, so rebuild after edits.
`.dockerignore` is an explicit code/docs/vendor allowlist. Do not copy `.env`,
Keychain material, model weights or results into an image. The container runs
non-root and publishes only a loopback port by default.

Keep vendored code unchanged. The root repository stores TrackNet files as normal
vendored source, not a git submodule; nested upstream `.git` metadata is excluded.
Preserve vendor licenses and provenance. Do not lower tracking gates or turn
regression predictions/unreviewed pose candidates into validation/coaching evidence.
Bump the AI report version when changing its input/output policy to invalidate
stale caches; keep local display labels separate from anonymous provider input.
