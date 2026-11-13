# Docker operations

## Start, stop and update

From the repository root, with Docker running and Make/Python 3 available:

```sh
make docker-up
make docker-status
make docker-logs
make docker-down
```

The dashboard is at http://127.0.0.1:8502. Change the published port with
`make docker-up DOCKER_PORT=8503`. The port binds to loopback only. No registry
push or public service is performed. To rebuild after code/dependency changes,
run `make docker-up` again; dependency layers are cached.

The image is Python 3.12 on Debian Bookworm with pinned CPU PyTorch and application
dependencies. Build tools stay in a separate build stage. The runtime is non-root;
the launcher matches the host UID/GID for writable mounted folders. It runs one
Streamlit server; tracking jobs remain separate child processes. Health is checked
through Streamlit's local health endpoint. Initial model download can take time.

## Persistent data and saved absolute paths

These four folders are mounted; none is copied into the image or committed:

| Folder | Contents |
|---|---|
| `data/` | Footage, calibration/configuration and source-frame maps |
| `models/` | Downloaded, checksum-verified weights |
| `results/` | Run databases, overlays, reviews and reports |
| `.cache/` | Model/application caches and local processing logs |

`make docker-down` removes containers/network, not these folders. Do not delete
or edit existing run artifacts to make validation pass.
Finish active tracking before stopping: interrupted workers do not resume
automatically after a container restart.

The launcher sets the image's application path to the native repository's
absolute path. This lets existing manifests continue to resolve source/model/map
paths without changing signed/frozen evidence. Run from the same project folder
when using existing results. Moving old footage/results to a different machine
or absolute folder requires reimport/reanalysis or a separately reviewed path
migration; copying a manifest does not rewrite its source locations.

Direct Compose defaults work for a fresh checkout when invoked from its root,
but `make docker-up` also handles ownership and saved API credentials for you.

## Groq credentials

The base Compose service has no API secret. For `up`/runtime commands, the host
launcher reads `GROQ_API_KEY` or this app's Mac Keychain item. When available it
adds `compose.groq.yaml`, which mounts a runtime secret at
`/run/secrets/groq_api_key`. Only its path (`GROQ_API_KEY_FILE`) is placed in the
container environment. The key is never a build argument, image layer, Git file,
CLI argument or application log.

The container cannot access macOS Keychain directly. Save a key through the
native Mac app if needed, or provide the environment variable through your normal
secret manager, then restart with `make docker-up`. If none is available, the
app still starts and its AI settings accept a session-only key. Keychain-save
controls are hidden in Linux containers. Model tracking requires no API key.

`make docker-build`, `make docker-test`, logs/status and shutdown do not read the
Mac key. Tests run without a Groq secret. Compose manages secret delivery; do not
publish container diagnostics that expose secret contents or mount credentials
into unrelated services.

## GPU and performance

This image intentionally uses CPU PyTorch. Docker on this Mac is a Linux runtime
and does not use the app's native Apple MPS backend. Full-match tracking will be
slower than native MPS. Use `make run` and `DEVICE=mps` for native Mac inference.
The CPU container can still review existing results, calibrate, generate reports,
run footwork analysis and process new footage. No CUDA/NVIDIA profile is claimed.

## Checks and troubleshooting

```sh
make docker-test
python3 scripts/docker_compose.py exec app python -m shuttlecoach.cli probe data/YOUR_VIDEO.mp4
```

- **Docker unavailable:** start Docker Desktop/Engine and retry. The helper does not start another daemon.
- **Port busy:** choose `DOCKER_PORT=8503`; it does not kill a native app or another service.
- **Model startup failed:** inspect `make docker-logs`; downloads/checksums must succeed before the server starts.
- **Permission denied in mounted folders:** use the Make launcher to match your host UID/GID.
- **Saved source unavailable:** keep videos inside this checkout's `data/` folder, and preserve existing absolute paths.
- **Old image after edits:** rerun `make docker-up` to rebuild and recreate the service.
- **API key missing:** review the secret setup above; never paste a key into chat or a Dockerfile.
