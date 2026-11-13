"""Camera-plane movement candidates and explicit video review, never technique grades."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .media import sha256, write_json
from .timeline import crosses_cut, source_seconds

PRACTICE_SOURCE = 'https://www.badminton.org.au/wp-content/uploads/2022/11/1.-BWF-ST-Teachers-Manual_Inclusivity_ENGLISH.pdf'
EVENT_NAMES = {'lunge':'Lunge shape', 'stance_opening':'Stance opening',
               'left_step':'Left-foot movement', 'right_step':'Right-foot movement'}


def joint_angle(a, b, c):
    u, v = a-b, c-b
    length = np.linalg.norm(u)*np.linalg.norm(v)
    return float(np.degrees(np.arccos(np.clip(np.dot(u,v)/length,-1,1)))) if length > 1e-6 else None


def pose_features(item):
    points = np.asarray(item['keypoints'],dtype=float).reshape(17,3)
    needed = [5,6,11,12,13,14,15,16]
    height = float(item['bbox'][3])
    if not np.isfinite(height) or height <= 0 or not np.isfinite(points[needed]).all() or (points[needed,2] < .5).any():return None
    xy=points[:,:2];hip=xy[11:13].mean(axis=0);shoulder=xy[5:7].mean(axis=0)
    left=joint_angle(xy[11],xy[13],xy[15]);right=joint_angle(xy[12],xy[14],xy[16])
    if left is None or right is None or np.linalg.norm(shoulder-hip)<1:return None
    return {'left_knee_image_deg':left,'right_knee_image_deg':right,
            'stance_body_heights':float(np.linalg.norm(xy[15]-xy[16])/height),
            'torso_lean_image_deg':float(np.degrees(np.arctan2(abs(shoulder[0]-hip[0]),abs(shoulder[1]-hip[1])))),
            'ankles':xy[15:17].copy(), 'height':height}


def analyze_footwork(raw, observations, meta):
    indexed={(int(r['frame_num']),r['player_id']):r for r in raw}
    samples=[];events=[]
    for (rally_id,side,player),group in observations.groupby(['rally_id','side','player'],sort=False):
        previous=None;last_event={};active=set();segment=0
        for row in group.sort_values('source_frame').itertuples():
            item=indexed.get((row.source_frame,side))
            feature=pose_features(item) if row.observed and item is not None else None
            if feature is None:
                previous=None;active=set();segment+=1
                continue
            contiguous=previous is not None and 0 < row.source_time_s-previous[0].source_time_s <= .25 and not crosses_cut(meta,previous[0].source_frame,row.source_frame)
            if not contiguous:previous=None;active=set();segment+=1
            record={'rally_id':int(rally_id),'player':player,'side':side,'source_frame':int(row.source_frame),
                    'source_time_s':float(row.source_time_s),'original_source_time_s':float(source_seconds(meta,row.source_frame)),
                    'sample_duration_s':float(row.sample_duration_s),'segment':segment,
                    **{k:v for k,v in feature.items() if k not in {'ankles','height'}}}
            samples.append(record);conditions=set()
            # shortcut: 2D pose thresholds propose review events; upgrade only with labelled event validation.
            if feature['stance_body_heights'] >= .35 and min(feature['left_knee_image_deg'],feature['right_knee_image_deg'])<=135 and max(feature['left_knee_image_deg'],feature['right_knee_image_deg'])>=145:conditions.add('lunge')
            if previous is not None:
                old=previous[1]
                if feature['stance_body_heights']-old['stance_body_heights']>=.08:conditions.add('stance_opening')
                movement=np.linalg.norm(feature['ankles']-old['ankles'],axis=1)/max(feature['height'],old['height'])
                if movement.max()>=.10 and movement.max()>1.5*movement.min():conditions.add('left_step' if movement[0]>movement[1] else 'right_step')
            for kind in sorted(conditions-active):
                if record['source_time_s']-last_event.get(kind,-1e9)<.5:continue
                last_event[kind]=record['source_time_s']
                events.append({**record,'kind':kind,'event_id':f'{int(rally_id)}:{side}:{row.source_frame}:{kind}'})
            active=conditions;previous=(row,feature)
    summaries=[]
    for player,group in observations.groupby('player',sort=False):
        usable=[r for r in samples if r['player']==player];found=[r for r in events if r['player']==player]
        selected=float(group.sample_duration_s.sum());seconds=sum(r['sample_duration_s'] for r in usable)
        summaries.append({'player':player,'pose_seconds':seconds,'selected_seconds':selected,
                          'pose_coverage':seconds/selected if selected else 0.,'candidate_counts':{kind:sum(r['kind']==kind for r in found) for kind in EVENT_NAMES},
                          'median_torso_lean_image_deg':float(np.median([r['torso_lean_image_deg'] for r in usable])) if usable else None})
    return {'summaries':summaries,'events':events,'samples':samples,
            'limitations':'Camera-plane pose candidates, not confirmed steps, split steps, physical balance, 3D joint angles or technique grades. Missing poses and cuts break sequences. Split-step hit/landing and lunge readiness times require source-video review. Estimated positions are never used.'}


def write_footwork(run):
    from .runner import raw_tracks
    run=Path(run);manifest=json.loads((run/'analysis.json').read_text())
    if sha256(manifest['video']['path'])!=manifest['video']['sha256']:raise ValueError('Source changed; footwork cannot be aligned.')
    for name,digest in manifest['artifacts_sha256'].items():
        if sha256(run/name)!=digest:raise ValueError(f'Player artifact changed: {name}')
    from .coaching import reviewed_play
    play,intervals=reviewed_play(pd.read_csv(run/'observations.csv'),manifest['config'])
    payload=analyze_footwork(raw_tracks(run/'tracks.duckdb'),play,manifest['video'])
    payload['intervals']=intervals
    identity={'source':manifest['video']['sha256'],'artifacts':manifest['artifacts_sha256'],
              'config':manifest['config'],'homography':manifest['homography'],'events':payload['events']}
    payload.update(schema_version=1,source_sha256=manifest['video']['sha256'],
                   evidence_sha256=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest())
    for field in ['samples','events']:
        pd.DataFrame(payload[field]).to_csv(run/f'footwork_{field}.csv',index=False)
    payload.pop('samples')
    write_json(run/'footwork.json',payload)
    return payload


def reviewed_footwork(footwork, review, intervals):
    """Only fully reviewed candidate sets within a rally can support a pattern."""
    empty={'status':'pending','rally_ids':[],'events':[],'reason':'Confirm or reject every candidate in each rally you want to use.'}
    if not review:return empty
    if review.get('evidence_sha256')!=footwork['evidence_sha256']:return {**empty,'status':'stale','reason':'Tracking or footwork evidence changed; review again.'}
    if not str(review.get('reviewer','')).strip():raise ValueError('A footwork reviewer is required.')
    events={r['event_id']:r for r in footwork['events']};labels=review.get('events',[])
    if len({r['event_id'] for r in labels})!=len(labels) or any(r['event_id'] not in events for r in labels):raise ValueError('Unknown or duplicate footwork event.')
    indexed={r['event_id']:r for r in labels};complete=[];confirmed=[]
    checked={}
    for label in labels:
        event=events[label['event_id']]
        if type(label.get('confirmed')) is not bool:raise ValueError('Every submitted footwork decision must be confirm or reject.')
        record={**event,'readiness':label.get('readiness','Not assessed'),'note':str(label.get('note',''))}
        if record['readiness'] not in {'Not assessed','Ready stance visible','Extra adjustment visible'}:raise ValueError('Choose an explicit readiness review label.')
        interval=intervals[event['rally_id']-1];start=interval.get('original_start_s',interval['start_s']);end=interval.get('original_end_s',interval['end_s'])
        for field in ['opponent_hit_s','landing_s','ready_s']:
            value=label.get(field)
            if value is not None and (not np.isfinite(float(value)) or not start<=float(value)<=end):raise ValueError('Footwork review timestamps must lie inside the same original-video rally.')
            record[field]=float(value) if value is not None else None
        t=event['original_source_time_s'];hit,landing,ready=(record[k] for k in ['opponent_hit_s','landing_s','ready_s'])
        if (hit is not None or landing is not None) and (event['kind']!='stance_opening' or any(abs(v-t)>2 for v in [hit,landing] if v is not None) or (hit is not None and landing is not None and abs(landing-hit)>1)):raise ValueError('Hit/landing timing needs a nearby reviewed stance-opening event (within two seconds; difference at most one second).')
        if ready is not None and (event['kind']!='lunge' or not t<=ready<=t+3):raise ValueError('Ready time must be within three seconds after this reviewed lunge.')
        record['split_step_offset_s']=landing-hit if hit is not None and landing is not None else None
        record['lunge_to_ready_s']=ready-t if ready is not None else None
        checked[event['event_id']]=record
    for rally_id in sorted({r['rally_id'] for r in events.values()}):
        part=[r for r in events.values() if r['rally_id']==rally_id]
        if not all(r['event_id'] in indexed for r in part):continue
        complete.append(rally_id)
        confirmed.extend(checked[r['event_id']] for r in part if indexed[r['event_id']]['confirmed'])
    return {'status':'reviewed' if len(complete)==len({r['rally_id'] for r in events.values()}) and complete else 'partial' if complete else 'pending',
            'rally_ids':complete,'events':confirmed,'reason':'Review applies to selected candidate-bearing rallies only; detector recall is not established.'}


def footwork_suggestions(footwork_review, observations, intervals):
    from .analysis import summarize
    summaries={(r['rally_id'],r['player']):r for r in summarize(observations)}
    suggestions=[]
    for player in observations.player.unique():
        for kind,field,title,practice in [
            ('stance_opening','split_step_offset_s','Reviewed split-step timing','Use controlled direction feeds. Review your landing relative to the feeder’s hit, then refilm the same sequence to compare timing.'),
            ('lunge','lunge_to_ready_s','Reviewed lunge recovery','Use controlled front-court feeds, play the reply, and recover to a ready stance before the next feed. Refilm and compare the lunge-to-ready interval.'),
            ('left_step',None,'Reviewed step sequence','Rehearse the reviewed approach and recovery slowly, then repeat with controlled feeds. Compare the sequence on video before changing it.'),
            ('readiness',None,'Video-reviewed readiness','Use controlled feeds and pause after each reply to review the ready stance and any adjustment steps. Refilm the same movement to compare readiness.')]:
            items=[r for r in footwork_review['events'] if r['player']==player and (r['kind']==kind or (kind=='left_step' and r['kind']=='right_step') or (kind=='readiness' and r['readiness']=='Extra adjustment visible')) and (field is None or r[field] is not None) and not intervals[r['rally_id']-1].get('partial')]
            items=[r for r in items if (s:=summaries.get((r['rally_id'],player))) and s['coverage']>=.95 and s['observed_time_s']>=5]
            rally_ids={r['rally_id'] for r in items};seconds=sum(summaries[(r,player)]['observed_time_s'] for r in rally_ids)
            if len(rally_ids)<3 or seconds<30:continue
            values=[r[field] for r in items] if field else []
            sequences={}
            for r in sorted(items,key=lambda r:r['source_time_s']):sequences.setdefault((r['rally_id'],r['segment']),[]).append('L' if r['kind']=='left_step' else 'R')
            detail=f'Median {np.median(values):+.2f} s relative to opponent hit (negative = before).' if kind=='stance_opening' else f'Median {np.median(values):.2f} s from reviewed lunge candidate to manually marked ready stance.' if field else ' | '.join(' → '.join(s[:12]) for s in list(sequences.values())[:3])+' (separate visible segments; not an optimal sequence).'
            if kind=='readiness':detail='The reviewer marked extra adjustment steps. This is a video observation, not a physical balance measurement.'
            suggestions.append({'player':player,'title':title,'pattern':f'{len(items)} confirmed events across {len(rally_ids)} reviewed rallies. '+detail,
                                'suggestion':practice+' These observations do not establish a technique fault or a universal timing target.',
                                'evidence':[{'start_s':r['original_source_time_s'],'end_s':r['original_source_time_s'],'rally_id':r['rally_id']} for r in items], 'kind':'footwork'})
    return suggestions
