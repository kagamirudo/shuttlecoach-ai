"""Held-out manual references. Fail closed when labels or calibration checks are incomplete."""
import numpy as np
import pandas as pd

from .upstream import court
from .calibration import fitted_court_points


def screen_coverage(observations,template):
    observations=observations.copy()
    if 'tracking_observed' in observations:observations['observed']=observations.tracking_observed
    sample=template[["source_frame","side"]].merge(observations[["source_frame","side","observed"]],
                                                     on=["source_frame","side"],how="left",validate="one_to_one")
    sample["observed"]=sample["observed"].eq(True)
    complete=sample.groupby("source_frame")["observed"].all()
    upper_bound=float(complete.mean()) if len(complete) else 0.
    return {"status":"failed" if upper_bound<0.95 else "manual_review_required",
            "sample_frames":len(complete),"both_slots_observed_frames":int(complete.sum()),
            "upper_bound_on_both_players_correct_rate":upper_bound,"required_rate":0.95,
            "identity_accuracy":"not assessed","position_accuracy":"not assessed",
            "explanation":"A missing reliable athlete position cannot be a correct tracking result. Verified off-court tracking counts here, but never enters court occupancy. This is a coverage upper bound, not measured identity or position accuracy.",
            "coaching_enabled":False}


def sample_template(observations, count=100):
    # Spread samples throughout every selected rally, without consulting predictions.
    groups = [g.source_frame.unique() for _, g in observations.groupby("rally_id")]
    count = min(count, sum(map(len, groups)))
    allocations = [0] * len(groups)
    for _ in range(count):
        available = [i for i, frames in enumerate(groups) if allocations[i] < len(frames)]
        chosen = min(available, key=lambda i: allocations[i])
        allocations[chosen] += 1
    selected = set()
    for frames, amount in zip(groups, allocations):
        if amount:
            selected.update(int(frames[index]) for index in np.linspace(0, len(frames)-1, amount).round().astype(int))
    return pd.DataFrame([{"source_frame": frame, "side": side, "visible": None,
                          "image_x": None, "image_y": None, "identity_correct": None}
                         for frame in sorted(selected) for side in ("near", "far")])


def evaluate(observations, labels, calibration_checks, homography, template, calibration=None):
    observations=observations.copy()
    # Off-court athlete detections can be valid tracking; occupancy remains court-only.
    if 'tracking_observed' in observations:observations['observed']=observations.tracking_observed
    required = {"source_frame", "side", "visible", "image_x", "image_y", "identity_correct"}
    if not required <= set(labels.columns):
        raise ValueError("Reference CSV is missing required columns.")
    labels = labels.copy()
    if labels.duplicated(["source_frame", "side"]).any():
        raise ValueError("Duplicate frame/side labels are not allowed.")
    if not labels.side.isin(["near", "far"]).all():
        raise ValueError("Reference side must be near or far.")
    frames = pd.to_numeric(labels.source_frame, errors="coerce")
    if frames.isna().any() or not np.isfinite(frames).all() or (frames % 1 != 0).any():
        raise ValueError("Reference frame numbers must be finite integers.")
    labels["source_frame"] = frames.astype(int)
    expected = set(map(tuple, template[["source_frame", "side"]].to_numpy()))
    actual = set(map(tuple, labels[["source_frame", "side"]].to_numpy()))
    if actual != expected:
        raise ValueError("Complete the exact frozen reference template; do not select only successful detections.")
    for field in ("visible", "identity_correct"):
        mapping = {"true": True, "false": False, "1": True, "0": False, "1.0": True, "0.0": False}
        labels[field] = labels[field].map(lambda v: mapping.get(str(v).lower().strip()))
        if labels[field].isna().any():
            raise ValueError(f"Every {field} cell must explicitly be true or false.")
    refs = labels[labels.visible].copy()
    for field in ("image_x", "image_y"):
        refs[field] = pd.to_numeric(refs[field], errors="coerce")
        if not np.isfinite(refs[field].to_numpy(dtype=float)).all():
            raise ValueError("Every visible player needs a finite manual foot position.")
    joined = observations.merge(refs, on=["source_frame", "side"], suffixes=("", "_reference"), validate="one_to_one")
    if len(joined) != len(refs):
        raise ValueError("Reference frames fall outside the analyzed rallies.")
    matched = joined.observed & joined.identity_correct
    errors = []
    if matched.any():
        part = joined[matched]
        reference = court.image_to_court(part[["image_x_reference", "image_y_reference"]].to_numpy(), homography)
        prediction = part[["court_x_m", "court_y_m"]].to_numpy(dtype=float)
        errors = np.linalg.norm(reference - prediction, axis=1).tolist()
    # Both athletes must be visible and correctly assigned at the same sample frame.
    success = labels.merge(observations[["source_frame", "side", "observed"]], on=["source_frame", "side"], validate="one_to_one")
    success["correct"] = success.visible & success.identity_correct & success.observed
    per_frame = success.groupby("source_frame").correct.all()
    both_rate = float(per_frame.mean()) if len(per_frame) else 0
    checks = pd.DataFrame(calibration_checks)
    check_errors = []
    if len(checks):
        fields = ["image_x", "image_y", "court_x_m", "court_y_m"]
        if not set(fields) <= set(checks.columns) or not np.isfinite(checks[fields].to_numpy(dtype=float)).all():
            raise ValueError("Calibration checks need finite image and court coordinates.")
        if checks.duplicated(["image_x", "image_y"]).any():
            raise ValueError("Use distinct, unused court-line intersections for calibration checks.")
        expected_points = checks[["court_x_m", "court_y_m"]].to_numpy(dtype=float)
        if any(not court.in_court(p) for p in expected_points):
            raise ValueError("Calibration check points must be on the court.")
        fitted = fitted_court_points(calibration or {})
        if len(fitted) and any(np.min(np.linalg.norm(fitted - p, axis=1)) < 0.05 for p in expected_points):
            raise ValueError("Calibration checks must use points other than the fitted calibration landmarks.")
        if len(checks) >= 4 and np.linalg.matrix_rank(expected_points - expected_points.mean(axis=0)) < 2:
            raise ValueError("Calibration checks must span both court width and depth.")
        actual_points = court.image_to_court(checks[["image_x", "image_y"]].to_numpy(), homography)
        check_errors = np.linalg.norm(actual_points - expected_points, axis=1).tolist()
    median = float(np.median(errors)) if errors else None
    p90 = float(np.percentile(errors, 90)) if errors else None
    criteria = {
        "100_sample_frames": len(per_frame) >= 100,
        "both_players_correct_95pct": both_rate >= 0.95,
        "no_reviewed_identity_errors": bool(success.loc[success.observed,"identity_correct"].all()),
        "visible_reference_positions_100": len(refs) >= 100,
        "median_position_error_le_0_75m": median is not None and median <= 0.75,
        "p90_position_error_le_1_5m": p90 is not None and p90 <= 1.5,
        "four_unused_calibration_points_le_0_25m": len(check_errors) >= 4 and max(check_errors) <= 0.25,
    }
    return {"status": "passed" if all(criteria.values()) else "failed", "criteria": criteria,
            "sample_frames": len(per_frame), "visible_references": len(refs), "matched_positions": len(errors),
            "both_players_correct_rate": both_rate, "median_projected_error_m": median,
            "p90_projected_error_m": p90, "calibration_check_errors_m": check_errors,
            "limitations": "Manual image agreement under the reviewed homography; not independent physical motion-capture ground truth. A player-only pass does not unlock coaching; shuttle and rally-context gates are separate."}
