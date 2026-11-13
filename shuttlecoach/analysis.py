"""Reviewed, time-weighted occupancy; no shot, speed, or technique claims."""
import math

import numpy as np
import pandas as pd

from .upstream import court
from .calibration import fit_calibration

MAX_ANALYSIS_SPAN_S = 600


def validate_config(config, meta):
    if config.get("singles_confirmed") is not True:
        raise ValueError("Confirm exactly two singles players. Doubles is not supported.")
    if config.get("camera_stable_confirmed") is not True:
        raise ValueError("Confirm a fixed camera view without cuts throughout the selected interval.")
    calibration = config.get("calibration", {})
    if calibration.get("reviewed") is not True:
        raise ValueError("Review and confirm the projected court lines.")
    homography, corners = fit_calibration(calibration, meta)
    if calibration.get('mode') == 'intersections':
        calibration['corners_px'] = corners.tolist()
    frame = calibration.get("frame")
    if type(frame) is not int or not 0 <= frame < meta["frame_count"]:
        raise ValueError("Calibration frame is outside the source video.")
    rallies = config.get("rallies", [])
    if not rallies:
        raise ValueError("Add at least one manually reviewed rally interval.")
    previous = -1.0
    for rally in rallies:
        start, end = float(rally["start_s"]), float(rally["end_s"])
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= meta["duration_s"]):
            raise ValueError("Rally start/end must be within the video, with start < end.")
        if np.count_nonzero((np.asarray(meta["times"])>=start)&(np.asarray(meta["times"])<end))<2:
            raise ValueError("Every rally needs at least two decoded source frames.")
        if start < previous:
            raise ValueError("Rallies must be sorted and must not overlap.")
        names = [str(rally.get(key, "")).strip() for key in ("near_player", "far_player")]
        if not all(names) or names[0] == names[1]:
            raise ValueError("Give the near and far players different names or labels.")
        rally["start_s"],rally["end_s"]=start,end
        rally["near_player"],rally["far_player"]=names
        previous = end
    span = rallies[-1]["end_s"] - rallies[0]["start_s"]
    if span > MAX_ANALYSIS_SPAN_S:
        raise ValueError(f"Selected tracking span is {span:.1f}s; the maximum per run is {MAX_ANALYSIS_SPAN_S}s (10 minutes). Split longer selections into separate runs.")
    if not (rallies[0]["start_s"] <= meta["times"][frame] < rallies[-1]["end_s"]):
        raise ValueError("Select a calibration frame inside the analysis interval.")
    return homography


def track_reason(item):
    points=np.asarray(item['keypoints']).reshape(17,3)
    if not all(math.isfinite(float(item[k])) for k in ('court_x','court_y','img_x','img_y')):
        return 'nonfinite'
    if max(points[15,2],points[16,2])<.3 or item['pose_conf']<.3:
        return 'low_confidence'
    return 'accepted' if court.in_court((item['court_x'],item['court_y'])) else 'outside_court'


def build_observations(raw, meta, rallies, sample_frames=None):
    from .timeline import source_seconds,source_frame_number
    indexed = {(int(row["frame_num"]), row["player_id"]): row for row in raw}
    times = np.asarray(meta["times"])
    sampled=np.arange(len(times)) if sample_frames is None else np.asarray(sample_frames,dtype=int)
    ends=dict(zip(sampled,np.r_[times[sampled[1:]],meta["duration_s"]]))
    rows = []
    for rally_id, rally in enumerate(rallies, 1):
        for frame in sampled[(times[sampled] >= rally["start_s"]) & (times[sampled] < rally["end_s"])]:
            seconds = max(0.0, min(ends[frame], rally["end_s"]) - max(times[frame], rally["start_s"]))
            for side in ("near", "far"):
                item = indexed.get((int(frame), side))
                row = {"rally_id": rally_id, "source_frame": int(frame), "source_time_s": float(times[frame]),
                       "youtube_time_s": float(source_seconds(meta,frame)), "original_source_time_s":float(source_seconds(meta,frame)),
                       'original_source_frame':int(source_frame_number(meta,frame)), "sample_duration_s": seconds,
                       "side": side, "player": rally[f"{side}_player"], "observed": False,
                       "tracking_observed":False,
                       "reason": "missing", "image_x": None, "image_y": None,
                       "court_x_m": None, "court_y_m": None, "pose_confidence": None, "zone": None}
                if item:
                    x, y = float(item["court_x"]), float(item["court_y"])
                    reason=track_reason(item);finite=reason!='nonfinite'
                    row.update(reason=reason, observed=reason == "accepted",
                               tracking_observed=reason=="accepted" or (reason=="outside_court" and bool(item.get("identity_verified",False))),
                               image_x=float(item["img_x"]) if finite else None,
                               image_y=float(item["img_y"]) if finite else None,
                               court_x_m=x if finite else None, court_y_m=y if finite else None,
                               pose_confidence=float(item["pose_conf"]))
                    if row["observed"]:
                        depth = abs(y - court.NET_Y_M)
                        row["zone"] = "front" if depth < court.NET_Y_M / 3 else "middle" if depth < 2 * court.NET_Y_M / 3 else "rear"
                rows.append(row)
    return pd.DataFrame(rows)


def summarize(observations):
    summaries = []
    for (rally_id, player, side), group in observations.groupby(["rally_id", "player", "side"], sort=False):
        valid = group[group.observed]
        total = float(group.sample_duration_s.sum())
        usable = float(valid.sample_duration_s.sum())
        summaries.append({"rally_id": int(rally_id), "player": player, "side": side,
                          "sampled_time_s": total, "observed_time_s": usable,
                          "coverage": usable / total if total else 0,
                          "occupancy": {zone: float(valid.loc[valid.zone == zone, "sample_duration_s"].sum()) / usable if usable else None
                                        for zone in ("front", "middle", "rear")}})
    return summaries


def player_overview(observations):
    """Time-weighted player totals; off-court tracking stays outside occupancy."""
    overview=[]
    for player, group in observations.groupby('player',sort=False):
        selected=float(group.sample_duration_s.sum())
        available=group['tracking_observed'] if 'tracking_observed' in group else group.observed
        tracked=float(group.loc[available.eq(True),'sample_duration_s'].sum())
        on_court=group[group.observed.eq(True)]
        accepted=float(on_court.sample_duration_s.sum())
        overview.append({'player':player,'sides':group.side.drop_duplicates().tolist(),
                         'selected_s':selected,'tracked_s':tracked,'unavailable_s':max(0.,selected-tracked),
                         'coverage':tracked/selected if selected else 0.,'in_court_s':accepted,
                         'occupancy':{zone:float(on_court.loc[on_court.zone.eq(zone),'sample_duration_s'].sum())/accepted if accepted else None
                                      for zone in ['front','middle','rear']}})
    return overview
