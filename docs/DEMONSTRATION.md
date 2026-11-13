# Reviewed demonstration and measured limits

Media, results and private review artifacts are local-only and are not included
in the GitHub repository or Docker image. These measurements refer to the saved
recording identified below; they do not validate a newly imported video.

## Open the completed demonstration

```sh
make run
```

Open http://127.0.0.1:8501, choose **Review & validate**, then analysis
`20261111T224529051246Z`. Playback offers the player overlay, birdie overlay,
and **Full source video** with the complete original audio. Press **Ctrl+C** in
the terminal to stop the dashboard. Complete the installation below first if
`.venv` is not set up yet.

| Artifact | Location / scope |
|---|---|
| Complete original match | `data/RZaZ7GaPJU8.mp4` — 29:57.9, with audio |
| Final reviewed result | `results/20261111T224529051246Z/` |
| Player movement overlay | `overlay.mp4` — approximately 9:41, silent |
| Birdie overlay and mini-court | `shuttle_overlay.mp4` — approximately 9:41, silent |
| Coaching report | `coaching_report.md` — Gary first, original-video evidence links |
| Raw measurements | `observations.csv`, `shuttle.csv`, `raw_tracks.csv` |
| Validation | `validation.json`, `shuttle_validation.json`, reviewed reference CSVs |
| Provenance and historical labels | `audit/`, `assembly.json`, `source_frame_map.csv` |

This result contains **308.6 seconds (5:08.6) in manually reviewed rally intervals**.
The longer tracking video retains setup/retrieval context. It assembles selected
moments from original 2:51.4–27:21; it is not a continuous five-minute match window.
Boundaries were inspected at 0.5-second cadence and remain approximate. Partial
rally fragments do not support recurring-pattern coaching. The user confirmed
scored play starts no later than original 2:00.

The full source is separate from the analysis overlays. Overlays do not contain
original audio or every minute of the match. Their watermarks and
`original_source_time_s`/`original_source_frame` retain the original clock;
`source_time_s` and `source_frame` describe the assembled input.

## Measured result and limits

The original 100 frozen samples remain. Thirty early and sixteen later samples
were frozen before their additional inference, making **146 samples per gate**.
Manual feet were marked on source frames before inspecting predictions; identities
were reviewed separately. Birdie labels were checked against temporal source
frames, with historical versions and subsequent model-guided source corrections
preserved. This is a **same-recording development audit**, not an untouched
held-out benchmark or physical motion-capture ground truth.

| Gate | Final result | Unchanged requirement |
|---|---:|---:|
| Both player identities/positions usable | 142/146 = 97.3% | ≥95%, no retained reviewed identity errors |
| Median / 90th-percentile projected foot error | 0.131 / 0.426 m | ≤0.75 / ≤1.5 m |
| Unused court-line checks | 5; maximum 0.233 m | ≥4; each ≤0.25 m |
| Birdie precision / recall | 90.1% / 86.9% | ≥90% / ≥85% |
| Median birdie image error | 1.41 px | ≤8 px at 720p |

Birdie evaluation has **73 correct points, 8 false candidates and 11 visible
misses**, using a 16-pixel match tolerance. Four sampled player pairs have no
reliable far-player position. Those failures remain in the result. A sample pass
is not a claim of perfect tracking or reliable behavior on other recordings.

Gary’s supported observation is middle-third occupancy of at least 60% in nine
reviewed complete intervals, totaling 60.8 accepted seconds. Eight of those have
no accepted position in the front third. The report links each interval and asks
for review of short replies before recommending a conditional deep/short feed
drill. Occupancy alone does not diagnose slow recovery, bad tactics or a weakness.
No automatic shot labels, speed, landing locations or technique grades are claimed.
