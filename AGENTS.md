# ShuttleCoach working instructions

Read `docs/HANDOFF.md` before changing the project, then the relevant code/tests.
Keep that handoff current after meaningful changes: current source/run, actual
validation results, commands, limitations and unfinished work. Never claim a
historical recording's pass applies to a new recording.

- Prefer `make docker-up` for the packaged app at port 8502; native `make run`
  remains at 8501 for Mac MPS. Read `docs/DOCKER.md`; use `make help` for targets.
- Keep BadmintonAI and TrackNet vendored sources unchanged. Use local adapters.
- Preserve source clocks, frame maps, manual calibration and old run artifacts.
- Never lower tracking quality gates to hide occlusions. Forecasts/interpolation
  are separate, visibly labelled estimates, excluded from validation, occupancy,
  footwork evidence and coaching. They must not overwrite observed tracks.
- Footwork pose heuristics are review candidates, not validated technique grades.
  Split-step contact/landing and readiness require explicit video annotations.
- Coaching needs rechecked player + birdie validation and reviewed play intervals.
  Show evidence, limitations and practice suggestions in plain language.
- Check callers before edits. Use existing NumPy/pandas/OpenCV/Streamlit helpers;
  no new dependency for simple geometry or short-window regression.
- Run `make test` after substantive changes. Test gap/cut/end-change boundaries,
  stale review rejection and blocked coaching paths, not only successful output.
- Do not change user names, calibration or selected intervals to make a gate pass.
- Optional Groq report wording is user-authorized. Keep keys out of code, logs
  and reports; send anonymized summary evidence only, never video or player names.
  Key persistence is user-authorized through this app's macOS Keychain item;
  use `credentials.py`, never plaintext files or secrets in subprocess arguments.
  Docker gets the saved key through a runtime secret, never an image/build arg.
  Do not create accounts, enable paid billing, or claim live API verification
  without an actual authorized successful call. Free-plan quotas can change.
- AI analysis is shown inline via `report_ui.py`. Send aggregate observations
  and time windows, not just the validation verdict. Descriptive analysis can
  discuss measured accepted positions while quality gates fail; personalized
  practice remains gated. Require same-player evidence IDs and grounded numbers.
  Candidate footwork counts cannot establish movement quality or true step rates.

This file does not authorize delegation, new cloud services or external messages.
