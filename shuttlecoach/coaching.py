"""Small evidence-linked reporting layer. Re-evaluate saved references before coaching."""
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import summarize
from .media import sha256, write_json
from .validation import evaluate


def evidence_signature(run):
    run=Path(run);manifest=json.loads((run/'analysis.json').read_text())
    files=set(manifest['artifacts_sha256'])|{'reviewed_references.csv','reviewed_calibration_checks.json','shuttle.json','shuttle_reviewed_references.csv','shuttle_reference_template.csv','shuttle.csv','footwork_review.json'}
    if (run/'shuttle.json').exists():files.update(json.loads((run/'shuttle.json').read_text()).get('artifacts_sha256',{}))
    inputs={'source':sha256(manifest['video']['path']),'video':manifest['video'],'expected_artifacts':manifest['artifacts_sha256'],'config':manifest['config'],'homography':manifest['homography'],
            'files':{name:sha256(run/name) if (run/name).exists() else None for name in sorted(files)}}
    if manifest['config'].get('original_source'):inputs['original_source']=sha256(manifest['config']['original_source']['path'])
    return hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest()


def reviewed_play(observations, config):
    intervals=config.get('reviewed_rallies',config['rallies'])
    if not config.get('reviewed_rallies'):return observations,intervals
    previous=config['rallies'][0]['start_s'];end=config['rallies'][-1]['end_s'];parts=[]
    for i,rally in enumerate(intervals,1):
        a,b=rally['start_s'],rally['end_s']
        if not np.isfinite([a,b]).all() or not previous<=a<b<=end:
            raise ValueError('Reviewed play intervals must be ordered, non-overlapping and inside the tracked window.')
        previous=b
        t=observations.source_time_s;finish=t+observations.sample_duration_s
        part=observations[(t<b)&(finish>a)].copy()
        part['sample_duration_s']=np.minimum(finish.loc[part.index],b)-np.maximum(t.loc[part.index],a)
        part['rally_id']=i
        for side in ('near','far'):part.loc[part.side==side,'player']=rally[f'{side}_player']
        parts.append(part)
    return pd.concat(parts,ignore_index=True),intervals


def evidence_patterns(observations, intervals):
    patterns=[]
    for summary in summarize(observations):
        occupancy=summary["occupancy"]
        if summary["coverage"]<.95 or summary["observed_time_s"]<5:continue
        zone=max(occupancy,key=lambda z:occupancy[z] or 0)
        if (occupancy[zone] or 0)<.6:continue
        interval=intervals[summary["rally_id"]-1]
        if interval.get('partial') is True:continue
        patterns.append({"player":summary["player"],"side":summary["side"],"zone":zone,
            "fraction":occupancy[zone],"coverage":summary["coverage"],"accepted_s":summary['observed_time_s'],
            'front_fraction':occupancy['front'],
            "start_s":interval.get('original_start_s',interval["start_s"]),"end_s":interval.get('original_end_s',interval["end_s"])})
    return patterns


def coaching_payload(observations, config, player_validation, shuttle_validation):
    play,intervals=reviewed_play(observations,config)
    patterns=evidence_patterns(play,intervals)
    reasons=[]
    for label,result in [("Player tracking",player_validation),("Shuttle tracking",shuttle_validation)]:
        if result.get("status")!="passed" or not result.get("criteria") or not all(result["criteria"].values()):
            reasons.append(f"{label}: {result.get('status','pending')} validation.")
    if config.get("includes_between_point_pauses") and not config.get('reviewed_rallies'):
        reasons.append("Intervals include between-point pauses. Mark actual rallies before interpreting positioning as play behavior.")
    suggestions=[]
    if not reasons:
        groups={}
        for item in patterns:groups.setdefault((item["player"],item["zone"]),[]).append(item)
        for (player,zone),items in groups.items():
            if len(items)<3 or sum(r['accepted_s'] for r in items)<30:continue
            drill={'rear':"Use a controlled rear-corner feed, play an overhead, then recover to a balanced base before the next feed. Add a short feed to test the transition forward.",
                   'front':"Use a controlled net feed, play the reply, then recover to a balanced base before the next feed. Add a deep feed to test the transition back.",
                   'middle':"Use alternating front and rear feeds and check whether the first movement from the base matches the feed."}[zone]
            zero_front=sum(r['front_fraction']==0 for r in items)
            detail=''
            if zone=='middle' and zero_front>=3:
                detail=f" In {zero_front} of these intervals, no accepted position entered the front third."
                drill="Review the linked intervals for any short reply and check your first movement forward. If the video shows a late transition, practice two sets of eight alternating deep and short feeds: play the reply and regain a balanced ready position before the next controlled feed. Record the drill to compare readiness; do not assume middle-court occupancy itself is a fault."
            suggestions.append({"player":player,"pattern":f"At least 60% of accepted time in the {zone} third in {len(items)} reviewed rallies ({sum(r['accepted_s'] for r in items):.1f} accepted seconds)."+detail,
                "suggestion":f"{drill} Compare the linked rallies before choosing a recovery change: the opponent's placement and your tactical choice can explain this occupancy. Refilm the drill and review balance and readiness; these tracks do not measure an optimal base or diagnose a weakness.",
                "evidence":items})
    focus=config.get('focus_player',config.get('rally_review',{}).get('focus_player'))
    suggestions.sort(key=lambda r:r['player']!=focus)
    summaries=summarize(play)
    return {"phase":2,"coaching_enabled":bool(suggestions),"gate":"blocked" if reasons else "passed",
            "blocked_reasons":reasons,"patterns":patterns,"suggestions":suggestions,
            'reviewed_play_intervals':intervals,'reviewed_play_s':sum(r['end_s']-r['start_s'] for r in intervals),
            'rally_summaries':summaries,
            "scope":"Descriptive occupancy and conditional review prompts. No automatic shot labels, shuttle speed, landing locations or technique grades."}


def write_coaching_report(run):
    run=Path(run);manifest=json.loads((run/"analysis.json").read_text());source=manifest["video"]
    observations=pd.read_csv(run/"observations.csv")
    if manifest['config'].get('source_frame_map'):
        from .timeline import attach_timeline
        attach_timeline(source,manifest['config'])
    if manifest['config'].get('reviewed_rallies'):
        from .shuttle_association import validate_review
        review=manifest['config'].get('rally_review',{})
        validate_review(review,manifest)
        if review.get('reviewed_rallies')!=manifest['config']['reviewed_rallies']:
            raise ValueError('Coaching intervals differ from the original-video activity review.')
    player={"status":"pending"};shuttle={"status":"pending"}
    try:
        if sha256(source["path"])!=source["sha256"]:raise ValueError("Source changed")
        for name,digest in manifest["artifacts_sha256"].items():
            if sha256(run/name)!=digest:raise ValueError(f"Player artifact changed: {name}")
        if (run/"reviewed_references.csv").exists() and (run/"reviewed_calibration_checks.json").exists():
            player=evaluate(observations,pd.read_csv(run/"reviewed_references.csv"),json.loads((run/"reviewed_calibration_checks.json").read_text()),np.asarray(manifest["homography"]),pd.read_csv(run/"reference_template.csv"),manifest['config']['calibration'])
    except (ValueError,OSError,KeyError) as error:player={"status":"failed","reason":str(error)}
    if (run/"shuttle.json").exists() and (run/"shuttle_reviewed_references.csv").exists():
        from .shuttle import evaluate_shuttle
        try:
            m=json.loads((run/"shuttle.json").read_text())
            if m["source_sha256"]!=source["sha256"]:raise ValueError("Shuttle source changed")
            for name,digest in m["artifacts_sha256"].items():
                if sha256(run/name)!=digest:raise ValueError(f"Shuttle artifact changed: {name}")
            shuttle=evaluate_shuttle(pd.read_csv(run/"shuttle.csv"),pd.read_csv(run/"shuttle_reviewed_references.csv"),pd.read_csv(run/"shuttle_reference_template.csv"),source["width"],source["height"])
        except (ValueError,OSError,KeyError) as error:shuttle={"status":"failed","reason":str(error)}
    payload=coaching_payload(observations,manifest["config"],player,shuttle)
    if (run/'tracks.duckdb').exists():
        from .footwork import write_footwork,reviewed_footwork,footwork_suggestions
        try:
            footwork=write_footwork(run)
            review=json.loads((run/'footwork_review.json').read_text()) if (run/'footwork_review.json').exists() else {}
            reviewed=reviewed_footwork(footwork,review,footwork['intervals'])
            payload['footwork']={'summaries':footwork['summaries'],'review':reviewed,'limitations':footwork['limitations']}
            if payload['gate']=='passed':
                play,intervals=reviewed_play(observations,manifest['config'])
                payload['suggestions'].extend(footwork_suggestions(reviewed,play,intervals))
                payload['coaching_enabled']=bool(payload['suggestions'])
        except (ValueError,OSError,KeyError) as error:
            payload['footwork']={'review':{'status':'blocked','reason':str(error)},'summaries':[]}
    payload["validation"]={"players":player,"shuttle":shuttle}
    try:payload['evidence_signature']=evidence_signature(run)
    except (OSError,KeyError):payload['evidence_signature']=None
    write_json(run/"coaching.json",payload)
    manifest.update(phase=2,coaching_enabled=payload['coaching_enabled'],coaching_gate=payload['gate'])
    write_json(run/'analysis.json',manifest)
    lines=["# ShuttleCoach — Phase 2 evidence report","",f"Coaching gate: **{payload['gate'].upper()}**","",
           f"Analyzed source: {manifest['overlay_source_start_s']:.2f}–{source['times'][manifest['end_frame_exclusive']-1]:.2f} seconds.","",
           "## Validation","",f"- Players: **{player['status']}**",f"- Shuttle: **{shuttle['status']}**"]
    if source.get('original_source'):
        from .timeline import source_seconds
        lines[4]=f"Reviewed play is assembled from original source {source_seconds(source,manifest['start_frame']):.2f}–{source_seconds(source,manifest['end_frame_exclusive']-1):.2f}s. Evidence links and original_source_time_s use that original clock; source_time_s is the assembled clip clock. Context padding is excluded from play measurements."
    lines += [f"- {reason}" for reason in payload["blocked_reasons"]]
    if (run/'shuttle_audit.json').exists():
        audit=json.loads((run/'shuttle_audit.json').read_text())
        lines += ['',f"Birdie source review: {audit['resolved_samples']}/100 labels resolved; {audit['unresolved_samples']} remain uncertain. Partial-review diagnostics do not establish full-window accuracy.",
                  '[Birdie audit details](shuttle_audit.json) · [Source review labels](shuttle_reference_labels.csv)']
    if (run/"shuttle.json").exists():
        m=json.loads((run/"shuttle.json").read_text())
        lines += ["",f"The tracking output contains {m['detected_frames']:,} observed candidate points across {m['frame_count']:,} source frames. Detection frequency is not measured accuracy. No gaps were inpainted.","",
                  "[Birdie tracking video](shuttle_overlay.mp4) · [Timestamped shuttle CSV](shuttle.csv) · [Shuttle reference sample](shuttle_reference_template.csv)"]
        if m.get("detection_zone"):
            zone_text="Detection is masked to the court plus raised airspace, with a narrow left boundary excluding neighboring courts. Overlapping airspace still requires identity review." if m['detection_zone'].get('kind')=='calibrated_court_with_airspace' else "This older run masks detection to the floor footprint only."
            if m.get('candidate_mode'):zone_text='Detection uses the wider calibrated court airspace so near-side lofts are retained. Neighboring courts overlap this image zone; reviewed activity, target wrists and temporal component association are required. Foreground recovery uses actual white source pixels; it can still confuse hands, clothing or floor markings.'
            lines += ["",zone_text,
                      "High-shot detections outside the floor footprint stay in the video and CSV, with projection_available=false and empty floor-projection coordinates.",
                      "The mini-court and exported projection coordinates transform image points through the floor homography. They are not true birdie floor positions, landing points or physical flight trajectories.","",
                      "[Projected birdie trail](shuttle_court_projection.png) · [Detection-zone preview](shuttle_zone_preview.png)"]
    lines += ["","## Measured positioning patterns","","These are observations to inspect, not diagnosed weaknesses. Percentages exclude rejected positions. A suggestion needs at least three reviewed intervals, each with ≥5 accepted seconds and ≥95% coverage, and ≥30 accepted seconds combined. A zone must account for ≥60% of accepted time in each interval. These descriptive support rules are separate from the unchanged tracking accuracy gates.",""]
    if manifest['config'].get('reviewed_rallies'):
        lines += [f"Positioning below uses {len(payload['reviewed_play_intervals'])} visually reviewed play intervals totaling {payload['reviewed_play_s']:.1f} seconds; pauses are excluded. Boundaries are approximate, not an independently validated official rally count. The full-window tracking gates still use the original frozen samples.",'',
                  '| Play interval | Player | Accepted / sampled seconds | Front / middle / rear |','|---|---|---:|---|']
        for s in payload['rally_summaries']:
            z=' / '.join(f'{s["occupancy"][k]:.1%}' if s['occupancy'][k] is not None else 'unavailable' for k in ('front','middle','rear'))
            lines.append(f'| {s["rally_id"]} | {s["player"]} | {s["observed_time_s"]:.1f} / {s["sampled_time_s"]:.1f} | {z} |')
        lines.append('')
    elif manifest["config"].get("includes_between_point_pauses"):
        lines += ["This continuous section includes between-point pauses; its occupancy cannot establish rally tactics.",""]
    for item in payload["patterns"]:
        label=str(item["player"]).replace("<","&lt;").replace("\n"," ")
        url=manifest["config"].get("youtube_url")
        stamp=f"{item['start_s']:.1f}–{item['end_s']:.1f}s"
        evidence=f"[{stamp}]({url}&t={int(item['start_s'])}s)" if url else stamp
        lines.append(f"- {label}: {item['zone']} third for {item['fraction']:.1%} of accepted time, coverage {item['coverage']:.1%}; {evidence}.")
    if not payload["patterns"]:lines.append("No repeated-zone candidates met the minimum coverage and duration rules.")
    lines += ["","## Coaching","" ]
    if payload["suggestions"]:
        for item in payload["suggestions"]:
            lines += [f"- {item['player']}: {item['pattern']} {item['suggestion']}"]
            url=manifest['config'].get('youtube_url')
            if url:
                lines += ['  Evidence: '+', '.join(f'[{r["start_s"]:.1f}–{r["end_s"]:.1f}s]({url}&t={int(r["start_s"])}s)' for r in item['evidence'])]
    elif payload["blocked_reasons"]:lines.append("Suggestions withheld until the validation and rally-context gates pass.")
    else:lines.append("Tracking gates pass, but no repeated pattern meets the evidence-duration and coverage rules. No suggestion is justified.")
    if payload['suggestions']:
        lines += ['', 'Drill principles: [BWF Shuttle Time teachers’ manual](https://www.badminton.org.au/wp-content/uploads/2022/11/1.-BWF-ST-Teachers-Manual_Inclusivity_ENGLISH.pdf), preparation and recovery in the hitting cycle. The linked video measurements select a review topic; they do not establish a technique fault.']
    if payload.get('footwork'):
        foot=payload['footwork'];review=foot['review']
        lines += ['', '## Footwork review', '', f"Event review: **{review['status'].upper()}**. {review.get('reason','')}",
                  foot.get('limitations',''), '', 'Candidate counts are not confirmed steps or split steps. Timing comes from explicit source-video annotations. No estimated positions contribute.',
                  '', '[Footwork candidates](footwork_events.csv) · [Pose measurements](footwork_samples.csv)']
        for person in foot['summaries']:
            lines += [f"- {person['player']}: usable full-body poses in {person['pose_seconds']:.1f}/{person['selected_seconds']:.1f} seconds ({person['pose_coverage']:.1%})."]
    lines += ["",payload["scope"],"","[Player occupancy report](report.md) · [Machine-readable evidence](coaching.json)",""]
    (run/"coaching_report.md").write_text("\n".join(lines))
    return payload
