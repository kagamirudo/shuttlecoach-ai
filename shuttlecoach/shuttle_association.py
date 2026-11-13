"""Associate observed shuttle components with reviewed target-court activity.

Temporal estimates guide association only. Every emitted point is a TrackNet
component or a white foreground component in the actual source image.
"""
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .analysis import track_reason
from .media import read_frame, sha256, write_json
from .player_recovery import source_images
from .runner import raw_tracks, revise_run
from .shuttle import court_zone, project_shuttle, projection_plot, render_shuttle


def validate_review(review, manifest):
    if review.get('source_sha256')!=manifest['video']['sha256']:
        raise ValueError('Activity review belongs to a different source.')
    start=manifest['video']['times'][manifest['start_frame']]
    end=manifest['config']['rallies'][-1]['end_s']
    for field in ['reviewed_rallies','target_flight_intervals']:
        if not review.get(field):raise ValueError(f'Missing {field}.')
        previous=start
        intervals=[(r['start_s'],r['end_s']) for r in review[field]] if field=='reviewed_rallies' else sorted(review[field])
        for a,b in intervals:
            if not np.isfinite([a,b]).all() or not start-1e-6<=a<b<=end+1e-6 or a<previous-1e-6:
                raise ValueError('Reviewed intervals must be finite, non-overlapping and inside the tracked span.')
            previous=b
    if not review.get('reviewer'):raise ValueError('Record who reviewed the original source.')
    for rally in review['reviewed_rallies']:
        names=[str(rally.get(f'{side}_player','')).strip() for side in ('near','far')]
        if not all(names) or names[0]==names[1]:raise ValueError('Give reviewed players distinct names.')


def merge_activity_intervals(intervals):
    merged=[]
    for a,b in sorted(intervals):
        if not np.isfinite([a,b]).all() or a>=b:raise ValueError('Activity intervals need finite start < end.')
        if merged and a<=merged[-1][1]:merged[-1][1]=max(b,merged[-1][1])
        else:merged.append([a,b])
    return merged


def player_context(raw, frames, stride):
    index={}
    for row in raw:
        reason=track_reason(row)
        if reason!='accepted' and not (reason=='outside_court' and row.get('identity_verified',False)):continue
        points=np.asarray(row['keypoints']).reshape(17,3)
        refs=[p[:2] for p in points[[9,10]] if p[2]>=.3]
        if refs:index.setdefault(int(row['frame_num']),[]).extend((p,float(row['bbox'][3])) for p in refs)
    known=np.array(sorted(index));contexts=[]
    for f in frames:
        at=int(np.searchsorted(known,f));options=known[max(0,at-1):min(len(known),at+1)]
        nearest=int(min(options,key=lambda a:abs(a-f))) if len(options) else None
        contexts.append(index[nearest] if nearest is not None and abs(nearest-f)<=stride/2 else [])
    return contexts


def contact_distance(point, wrists):
    return min((np.linalg.norm(np.subtract(point,p))/max(25.,.3*h) for p,h in wrists),default=20.)


def white_components(image, background, floor, wrists):
    hsv=cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    difference=image.astype(float).mean(2)-background.astype(float).mean(2)
    mask=((hsv[:,:,1]<45)&(hsv[:,:,2]>150)&(difference>15)).astype(np.uint8)
    count,labels,stats,centers=cv2.connectedComponentsWithStats(mask,8)
    points=[]
    for component,((x,y,w,h,area),center) in enumerate(zip(stats[1:],centers[1:]),1):
        if not 3<=area<=100 or w>22 or h>22:continue
        distance=min((np.linalg.norm(center-p)/max(8.,min(18.,.12*height)) for p,height in wrists),default=100.)
        on_floor=cv2.pointPolygonTest(floor,tuple(map(float,center)),False)>=0
        values=hsv[y:y+h,x:x+w,2][labels[y:y+h,x:x+w]==component]
        points.append({'image_x':float(center[0]),'image_y':float(center[1]),'confidence':float(values.mean()/255),
                       'method':'held_foreground' if distance<=1. else 'floor_foreground' if on_floor else 'air_foreground','wrist_distance':distance,'area':int(area)})
    return points


def flight_path(candidates, contexts, times):
    """Short beam search: prefer smooth real observations and target-wrist contacts."""
    # score, last point, previous point, last time, previous time, actual choices
    beam=[(0.,None,None,None,None,[])]
    for options,wrists,t in zip(candidates,contexts,times):
        next_beam=[]
        for k,p in enumerate(options):
            xy=np.array([p['image_x'],p['image_y']]);contact=contact_distance(xy,wrists)
            best=None
            for score,last,previous,last_t,previous_t,path in beam:
                gap=t-last_t if last_t is not None else 1.
                if last is None or gap>.25:
                    if contact>3.:continue
                    motion=.15*contact
                else:
                    if np.linalg.norm(xy-last)>160*gap/(1/30):continue
                    velocity=(last-previous)/(last_t-previous_t) if previous is not None and last_t>previous_t else np.zeros(2)
                    residual=np.linalg.norm(xy-(last+velocity*gap))
                    motion=.003*residual if contact>2. else 0.
                cost=score+motion+.08*min(contact,10.)-.7*p['confidence']
                node=(cost,xy,last,t,last_t,path+[k])
                if best is None or cost<best[0]:best=node
            if best is not None:next_beam.append(best)
        next_beam.extend((s+.6,last,prev,lt,pt,path+[None]) for s,last,prev,lt,pt,path in beam[:3])
        beam=sorted(next_beam,key=lambda n:n[0])[:5]
    return min(beam,key=lambda n:n[0])[-1]


def bidirectional_flight(candidates, contexts, times):
    forward=flight_path(candidates,contexts,times)
    backward=flight_path(candidates[::-1],contexts[::-1],-np.asarray(times)[::-1])[::-1]
    selected=[]
    for options,wrists,a,b in zip(candidates,contexts,forward,backward):
        choices={k for k in (a,b) if k is not None}
        selected.append(min(choices,key=lambda k:contact_distance([options[k]['image_x'],options[k]['image_y']],wrists)*.05-options[k]['confidence']) if choices else None)
    return selected


def target_flight(candidates, contexts, times, core_left):
    """Retain the strongest in-core response; require contact-linked motion in wider airspace."""
    temporal=bidirectional_flight(candidates,contexts,times)
    selected=[]
    for options,k in zip(candidates,temporal):
        core=[i for i,p in enumerate(options) if p.get('method','tracknet')=='tracknet' and p['image_x']>=core_left]
        # Network component area is its localization support, not calibrated probability.
        selected.append(max(core,key=lambda i:(options[i]['area'],options[i]['confidence'])) if core else k)
    return selected


def on_player(point, row):
    """Reject white clothing without masking the empty space inside a pose box."""
    p=np.asarray(point,float);j=np.asarray(row['keypoints']).reshape(17,3)
    bx,by,w,h=row['bbox'];margin=max(3.,.02*h)
    torso=j[[5,6,12,11],:2].astype(np.float32)
    if (j[[5,6,12,11],2]<.3).any() or cv2.contourArea(torso)<10:
        return bx-w/2-3<=p[0]<=bx+w/2+3 and by-h/2-3<=p[1]<=by+h/2+2
    if cv2.pointPolygonTest(torso,tuple(p),True)>=-margin:return True
    for a,b in [(5,7),(7,9),(6,8),(8,10),(11,13),(13,15),(12,14),(14,16)]:
        if min(j[a,2],j[b,2])<.3:continue
        delta=j[b,:2]-j[a,:2]
        fraction=np.clip(np.dot(p-j[a,:2],delta)/max(1.,np.dot(delta,delta)),0.,1.)
        if np.linalg.norm(p-j[a,:2]-fraction*delta)<=margin:return True
    if any(j[k,2]>=.3 and abs(p[0]-j[k,0])<=.14*h and abs(p[1]-j[k,1])<=.04*h for k in [15,16]):return True
    return bool(j[0,2]>=.3 and np.linalg.norm(p-j[0,:2])<=.05*h)


def associate_candidates(table, foreground, raw, manifest, review):
    from .timeline import crosses_cut
    frames=table.source_frame.to_numpy(int);times=table.source_time_s.to_numpy(float)
    contexts=player_context(raw,frames,manifest['tracking_stride'])
    corners=np.asarray(manifest['config']['calibration']['corners_px'])
    core_left=float((corners[0]+.65*(corners[3]-corners[0]))[0])
    upper_y=.35*float(corners[2:,1].min())
    zone=np.asarray(court_zone(manifest)['polygon_px'],np.float32)
    options=[]
    for i,(row,t) in enumerate(zip(table.itertuples(),times)):
        live=any(r['start_s']<=t<r['end_s'] for r in review['reviewed_rallies'])
        points=[]
        for p in json.loads(row.candidates):
            x,y=p['image_x'],p['image_y']
            if cv2.pointPolygonTest(zone,(float(x),float(y)),False)<0:continue
            on_floor=cv2.pointPolygonTest(corners.astype(np.float32),(float(x),float(y)),False)>=0
            if x<core_left and not (on_floor or live and y<=upper_y):continue
            after_play=any(r['end_s']<=t<r['end_s']+.75 for r in review['reviewed_rallies'])
            if not live and not after_play and not on_floor:continue
            if p['confidence']<.75 and contact_distance([x,y],contexts[i])<2. and not any(np.linalg.norm(np.array([q['image_x'],q['image_y']])-[x,y])<8 for q in foreground[i]):continue
            points.append(p|{'method':'tracknet'})
        options.append(points)
    selected=[None]*len(table)
    for a,b in sorted(review['target_flight_intervals']):
        ids=np.flatnonzero((times>=a)&(times<b))
        path=target_flight([options[i] for i in ids],[contexts[i] for i in ids],times[ids],core_left)
        for i,k in zip(ids,path):
            if k is not None:selected[i]=options[i][k]
    boxes={}
    for row in raw:
        if track_reason(row)=='accepted' or row.get('identity_verified',False):
            boxes.setdefault(int(row['frame_num']),[]).append(row)
    known=np.asarray(sorted(boxes));last=previous=None;last_t=previous_t=model_t=None
    for i,t in enumerate(times):
        if i and crosses_cut(manifest['video'],frames[i-1],frames[i]):last=previous=None;last_t=previous_t=model_t=None
        bodies=boxes[known[np.argmin(abs(known-frames[i]))]] if len(known) else []
        stationary=previous is not None and np.linalg.norm(last-previous)<3.
        max_age=2. if stationary else .25
        floor_support=stationary and any(p['method']=='floor_foreground' and np.linalg.norm(np.array([p['image_x'],p['image_y']])-last)<6 and not any(on_player([p['image_x'],p['image_y']],body) for body in bodies) for p in foreground[i])
        if selected[i] is not None and floor_support and t-last_t<=.25 and t-model_t<=max_age and selected[i]['confidence']<.7 and np.linalg.norm(np.array([selected[i]['image_x'],selected[i]['image_y']])-last)>120:
            selected[i]=None
        if selected[i] is not None:model_t=t
        elif last is not None and t-last_t<=.25 and t-model_t<=max_age and len(known):
            velocity=(last-previous)/(last_t-previous_t) if previous is not None and last_t>previous_t else np.zeros(2)
            proposal=last+velocity*(t-last_t)
            candidates=[]
            for p in foreground[i]:
                if p['method']!='floor_foreground':continue
                x,y=p['image_x'],p['image_y']
                if any(on_player([x,y],body) for body in bodies):continue
                distance=min(np.linalg.norm(np.array([x,y])-proposal),np.linalg.norm(np.array([x,y])-last))
                if distance<30:candidates.append((distance,p))
            if candidates:selected[i]=min(candidates,key=lambda q:q[0])[1]
        if selected[i] is not None:
            previous,previous_t=last,last_t
            last=np.array([selected[i]['image_x'],selected[i]['image_y']]);last_t=t
    return selected


def associate_shuttle(source, review_path):
    source=Path(source).resolve();manifest=json.loads((source/'analysis.json').read_text())
    review=json.loads(Path(review_path).read_text());validate_review(review,manifest)
    if sha256(manifest['video']['path'])!=manifest['video']['sha256']:raise ValueError('Source changed.')
    shuttle=json.loads((source/'shuttle.json').read_text())
    for name,digest in shuttle['artifacts_sha256'].items():
        if sha256(source/name)!=digest:raise ValueError(f'Shuttle artifact changed: {name}')
    table=pd.read_csv(source/('shuttle_candidates.csv' if (source/'shuttle_candidates.csv').exists() else 'shuttle.csv'))
    if 'candidates' not in table:raise ValueError('Run --new-court-review --candidates first.')
    target=revise_run(source,manifest['config']['rallies'],reuse_birdie=False);shutil.copy2(source/'shuttle_reference_template.csv',target/'shuttle_reference_template.csv');manifest=json.loads((target/'analysis.json').read_text());meta=manifest['video']
    manifest['config']['shuttle_association']=True;manifest['config']['reviewed_rallies']=review['reviewed_rallies'];manifest['config']['rally_review']=review
    write_json(target/'analysis.json',manifest);write_json(target/'activity_review.json',review)
    write_json(target/'shuttle_status.json',{'state':'running','stage':'source foreground and association'})
    table.to_csv(target/'shuttle_candidates.csv',index=False)
    frames=table.source_frame.astype(int).tolist();contexts=player_context(raw_tracks(target/'tracks.duckdb'),frames,manifest['tracking_stride'])
    cache_path=source/'foreground_candidates.json'
    cache_key={'source':meta['sha256'],'tracks':sha256(target/'raw_tracks.csv'),
               'frames':frames,'algorithm':'white-components-v2'}
    cache=json.loads(cache_path.read_text()) if cache_path.exists() else {}
    if cache.get('key')==cache_key and len(cache.get('points',[]))==len(frames):
        foreground=cache['points']
    else:
        samples=np.linspace(manifest['start_frame'],manifest['end_frame_exclusive']-1,64).round().astype(int)
        background=np.median(np.stack([read_frame(meta['path'],int(f),meta['times'],meta['pts_origin_s']) for f in samples]),axis=0).astype(np.uint8)
        floor=np.array(manifest['config']['calibration']['corners_px'],np.float32);foreground=[]
        for n,(f,bgr) in enumerate(source_images(meta,frames)):
            foreground.append(white_components(bgr[:,:,::-1],background,floor,contexts[n]))
            if (n+1)%900==0:print(f'Foreground {n+1}/{len(frames)} source frames',flush=True)
    write_json(target/'foreground_candidates.json',{'key':cache_key,'points':foreground})
    selected=associate_candidates(table,foreground,raw_tracks(target/'tracks.duckdb'),manifest,review)
    table['raw_detected']=table.detected;table['raw_image_x']=table.image_x;table['raw_image_y']=table.image_y
    table['detected']=[p is not None for p in selected]
    for axis in ['x','y']:table[f'image_{axis}']=[p[f'image_{axis}'] if p else np.nan for p in selected]
    table['association_method']=[p['method'] if p else 'unresolved' for p in selected]
    table=project_shuttle(table,manifest);table.to_csv(target/'shuttle.csv',index=False)
    segments=projection_plot(target,table)
    write_json(target/'shuttle_status.json',{'state':'running','stage':'rendering associated trajectories','processed_frames':len(table)})
    timing=render_shuttle(target,meta,table,manifest['start_frame'],manifest['end_frame_exclusive'],shuttle['tracking_stride'])
    shuttle.update(detected_frames=int(table.detected.sum()),overlay_timing=timing,projection_segments=segments,
                   raw_detection_zone=shuttle['detection_zone'],detection_zone=court_zone(manifest),
                   association={'method':'Actual TrackNet components, target-wrist temporal association in raised upper airspace, and short source-foreground recovery outside player boxes','manual_activity_review':True,'reference_points_used':False,'interpolated_positions':False},
                   artifacts_sha256={name:sha256(target/name) for name in ['shuttle.csv','shuttle_candidates.csv','shuttle_reference_template.csv','activity_review.json','foreground_candidates.json','shuttle_overlay.mp4']})
    write_json(target/'shuttle.json',shuttle);write_json(target/'shuttle_status.json',{'state':'complete','processed_frames':len(table),'validation':'pending'})
    from .coaching import write_coaching_report
    write_coaching_report(target)
    print(f'RESULT={target}',flush=True);return target
