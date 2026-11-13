"""Reacquire uncertain athletes from real YOLO detections; never fill positions."""
import json
import math
from pathlib import Path
import time

import av
import cv2
import numpy as np
import pandas as pd

from .analysis import build_observations, summarize, track_reason
from .media import read_frame, sha256, write_json
from .runner import choose_device, raw_tracks, revise_run, overlay, heatmaps, render_report
from .upstream import court, detect, db
from .validation import sample_template, screen_coverage


def shirt_histogram(image, box, torso_only=False):
    x,y,w,h=map(float,box)
    x0,x1=max(0,round(x-.3*w)),min(image.shape[1],round(x+.3*w))
    y0,y1=max(0,round(y-.3*h)),min(image.shape[0],round(y+(.05 if torso_only else .3)*h))
    if x1<=x0 or y1<=y0:return None
    hsv=cv2.cvtColor(image[y0:y1,x0:x1],cv2.COLOR_BGR2HSV)
    # Keep shorts separate: adjacent-court players can wear the same black shirt.
    split=max(1,round(hsv.shape[0]*.58))
    histograms=[]
    for region in ((hsv,) if torso_only else (hsv[:split],hsv[split:])):
        if not region.size:return None
        histogram=cv2.calcHist([region],[0,1,2],None,[12,4,4],[0,180,0,256,0,256])
        histograms.append(cv2.normalize(histogram,histogram,norm_type=cv2.NORM_L1).flatten())
    return np.concatenate(histograms)


def appearance_distance(histogram, gallery):
    if histogram is None or not gallery:return 1.
    return min(float(cv2.compareHist(histogram,reference,cv2.HISTCMP_BHATTACHARYYA)) for reference in gallery)


def select_candidate(candidates, expected, expected_height, gallery, radius, used=()):
    scored=[]
    for item in candidates:
        if any(np.linalg.norm(np.asarray(item['bbox'][:2])-other['bbox'][:2])<=.25*min(item['bbox'][3],other['bbox'][3]) for other in used):continue
        distance=float(np.linalg.norm(np.asarray([item['img_x'],item['img_y']])-expected))
        ratio=item['bbox'][3]/max(expected_height,1.)
        appearance=appearance_distance(item['shirt'],gallery)
        points=np.asarray(item['keypoints']).reshape(17,3)
        if distance>radius or not .45<=ratio<=2.25 or appearance>.65 or item['pose_conf']<.3 or points[15:17,2].max()<.3:continue
        cost=distance/radius+.6*appearance+.3*abs(math.log(ratio))
        if cost<1.3:scored.append((cost,item))
    return min(scored,key=lambda pair:pair[0])[1] if scored else None


def source_images(meta, frames):
    wanted=iter(frames);target=next(wanted,None)
    if target is None:return
    with av.open(meta['path']) as source:
        source.seek(max(0,int((meta['times'][target]+meta['pts_origin_s'])*av.time_base)),backward=True)
        for decoded in source.decode(video=0):
            stamp=float(decoded.pts*decoded.time_base)-meta['pts_origin_s']
            if stamp<meta['times'][target]-.0001:continue
            if abs(stamp-meta['times'][target])>.0001:raise ValueError('Recovery decoder disagrees with source timestamps.')
            yield target,decoded.to_ndarray(format='bgr24')
            target=next(wanted,None)
            if target is None:return
    raise ValueError('Source ended during identity recovery.')


def rally_slot(seconds, rallies):
    """Assign short context padding to its nearest rally, including after an end change."""
    return min(range(len(rallies)),key=lambda i:max(rallies[i]['start_s']-seconds,seconds-rallies[i]['end_s'],0.))


def identity_anchors(frame, side, anchors, rallies, times):
    slot=rally_slot(times[frame],rallies)
    available=np.asarray(anchors[(slot,side)],dtype=int)
    if not len(available):
        name=rallies[slot][f'{side}_player']
        same_half=[f for (i,s),values in anchors.items() if s==side and rallies[i][f'{side}_player']==name for f in values]
        # A whole rally can be uncertain; use one real same-athlete anchor, never interpolate across a cut.
        return np.asarray([min(same_half,key=lambda f:abs(times[f]-times[frame]))],dtype=int) if same_half else available
    at=int(np.searchsorted(available,frame))
    return available[max(0,at-1):min(len(available),at+1)]


def recovery_crop(image, expected, height):
    x,y=map(float,expected);height=max(40.,float(height))
    return [max(0,int(x-3*height)),max(0,int(y-2*height)),
            min(image.shape[1],int(x+3*height)),min(image.shape[0],int(y+.5*height))]


def pose_candidates(model, image, homography, frame, device, crop=None):
    x0,y0,x1,y1=crop if crop is not None else [0,0,image.shape[1],image.shape[0]]
    view=image[y0:y1,x0:x1]
    if not view.size:return []
    prediction=model.predict(view,imgsz=1920,device=device,verbose=False)[0]
    candidates=[]
    if prediction.keypoints is not None and prediction.boxes is not None:
        for box,xy,confidence in zip(prediction.boxes.xywh.cpu().numpy(),prediction.keypoints.xy.cpu().numpy(),prediction.keypoints.conf.cpu().numpy()):
            box=box.copy();xy=xy.copy();box[:2]+=[x0,y0];xy+=[x0,y0]
            point=detect.ground_point(box,xy,confidence);mapped=court.image_to_court(point.reshape(1,2),homography)[0]
            if not np.isfinite(np.r_[point,mapped]).all():continue
            candidates.append({'frame_num':frame,'bbox':box.tolist(),'keypoints':np.c_[xy,confidence].ravel().tolist(),
                'pose_conf':float(confidence.mean()),'img_x':float(point[0]),'img_y':float(point[1]),
                'court_x':float(mapped[0]),'court_y':float(mapped[1]),'shirt':shirt_histogram(image,box)})
    return candidates


def recover_players(source, device='auto'):
    from ultralytics import YOLO
    source=Path(source).resolve();manifest=json.loads((source/'analysis.json').read_text());meta=manifest['video']
    if sha256(meta['path'])!=meta['sha256']:raise ValueError('Source video changed.')
    device=choose_device(device)
    target=revise_run(source,manifest['config']['rallies']);manifest=json.loads((target/'analysis.json').read_text())
    write_json(target/'status.json',{'state':'running','stage':'identity recovery','validation':'pending'})
    raw=raw_tracks(target/'tracks.duckdb');indexed={(int(r['frame_num']),r['player_id']):r for r in raw}
    frames=manifest['processed_source_frames'];rallies=manifest['config']['rallies']
    observations=build_observations(raw,meta,rallies,frames)
    galleries={(r[f'{side}_player'],side):[] for r in rallies for side in ['near','far']};torso_galleries={key:[] for key in galleries}
    def athlete(f,side):return rallies[rally_slot(meta['times'][f],rallies)][f'{side}_player']
    seed_frames={key:[] for key in galleries}
    for f in frames:
        for side in ('near','far'):
            key=(athlete(f,side),side);item=indexed.get((f,side))
            if len(seed_frames[key])<5 and item is not None and track_reason(item)=='accepted':seed_frames[key].append(f)
    # The far view has different scale and exposes the opposite shirt face after an end change.
    for f,image in source_images(meta,sorted({f for values in seed_frames.values() for f in values})):
        for side in ('near','far'):
            key=(athlete(f,side),side)
            if f not in seed_frames[key]:continue
            item=indexed.get((f,side))
            if item is not None:
                h=shirt_histogram(image,item['bbox'])
                if h is not None:galleries[key].append(h)
                h=shirt_histogram(image,item['bbox'],torso_only=True)
                if h is not None:torso_galleries[key].append(h)
    if not all(galleries.values()):raise ValueError('Appearance seeds are required for each athlete and court half.')
    assert all(1<=len(values)<=5 for values in galleries.values())
    trusted={(int(r.source_frame),r.side) for r in observations.itertuples() if r.observed}
    for f,image in source_images(meta,frames):
        for side in ('near','far'):
            item=indexed.get((f,side))
            # A bending pose moves shorts out of their usual box region. Use the
            # torso alone to check existing tracks; shorts disambiguate new ones.
            if item is not None and appearance_distance(shirt_histogram(image,item['bbox'],torso_only=True),torso_galleries[(athlete(f,side),side)])>.65:
                trusted.discard((f,side))
    anchors={(i,side):sorted(f for f,s in trusted if s==side and rally_slot(meta['times'][f],rallies)==i) for i in range(len(rallies)) for side in ('near','far')}
    uncertain={f for f in frames if any((f,side) not in trusted for side in ('near','far'))}
    model=YOLO(manifest['model']['path']);H=np.asarray(manifest['homography']);records=[];started=time.perf_counter()
    for n,(f,image) in enumerate(source_images(meta,sorted(uncertain)),1):
        candidates=pose_candidates(model,image,H,f,device)
        # Reserve trusted poses too: recovery must not assign the near athlete to both slots.
        chosen=[indexed[(f,side)] for side in ('near','far') if (f,side) in trusted]
        for side in ('near','far'):
            if (f,side) in trusted:continue
            near=identity_anchors(f,side,anchors,rallies,meta['times'])
            if not len(near):
                indexed.pop((f,side),None)
                records.append({'source_frame':f,'side':side,'player':athlete(f,side),'reacquired':False,'anchor_frames':[], 'method':'No trusted same-athlete anchor; position withheld.'})
                continue
            expected=np.array([np.interp(f,near,[indexed[(int(a),side)][key] for a in near]) for key in ['img_x','img_y']])
            expected_height=float(np.median([indexed[(int(a),side)]['bbox'][3] for a in near]))
            gap=(meta['times'][int(near[-1])]-meta['times'][int(near[0])]) if len(near)>1 else 2.
            radius=min(180.,50.+35.*gap)
            selected=select_candidate(candidates,expected,expected_height,galleries[(athlete(f,side),side)],radius,chosen)
            view='full_frame'
            if selected is None and side=='far':
                # Small far-side athletes need more source detail than a full-frame resize preserves.
                enlarged=pose_candidates(model,image,H,f,device,recovery_crop(image,expected,expected_height))
                selected=select_candidate(enlarged,expected,expected_height,galleries[(athlete(f,side),side)],radius,chosen)
                view='far_crop'
            indexed.pop((f,side),None)
            if selected is not None:
                chosen.append(selected);row={k:v for k,v in selected.items() if k!='shirt'};row.update(player_id=side,identity_verified=True,match_id='clip');indexed[(f,side)]=row
            records.append({'source_frame':f,'side':side,'player':athlete(f,side),'reacquired':selected is not None,'anchor_frames':near.tolist(),'detection_view':view,
                            'method':'Appearance and temporal anchors select a real pose; no position interpolation.'})
        if n%50==0:print(f'Player recovery {n}/{len(uncertain)} frames; {time.perf_counter()-started:.1f}s',flush=True)
    repaired=list(indexed.values())
    with db.transaction(target/'tracks.duckdb') as connection:
        connection.execute('ALTER TABLE tracks ADD COLUMN IF NOT EXISTS identity_verified BOOLEAN DEFAULT FALSE')
        connection.execute("DELETE FROM tracks WHERE match_id='clip'")
        rows=[['clip',r['frame_num'],r['player_id'],r['court_x'],r['court_y'],r['img_x'],r['img_y'],list(r['bbox']),list(r['keypoints']),r['pose_conf'],bool(r.get('identity_verified',False))] for r in repaired]
        connection.executemany('INSERT INTO tracks(match_id,frame_num,player_id,court_x,court_y,img_x,img_y,bbox,keypoints,pose_conf,identity_verified) VALUES (?,?,?,?,?,?,?,?,?,?,?)',rows)
    raw=raw_tracks(target/'tracks.duckdb');observations=build_observations(raw,meta,rallies,frames)
    observations.to_csv(target/'observations.csv',index=False)
    pd.DataFrame([{k:v for k,v in r.items() if k not in ['bbox','keypoints']} for r in raw]).to_csv(target/'raw_tracks.csv',index=False)
    assert (sample_template(observations)[['source_frame','side']].to_numpy()==pd.read_csv(source/'reference_template.csv')[['source_frame','side']].to_numpy()).all()
    heatmaps(observations,target/'heatmap.png')
    manifest['overlay_timing']=overlay(Path(meta['path']),target/'overlay.mp4',raw,meta,manifest['start_frame'],manifest['end_frame_exclusive'],H,manifest['tracking_stride'])
    manifest['player_recovery']={'method':'Appearance and bidirectional temporal reacquisition of actual YOLO poses; enlarged far-player crop when full-frame detection misses','uncertain_frames':len(uncertain),'reacquired_positions':sum(r['reacquired'] for r in records),'far_crop_reacquired':sum(r['reacquired'] and r.get('detection_view')=='far_crop' for r in records),'references_used_for_selection':False}
    manifest['summaries']=summarize(observations);manifest['artifacts_sha256']={name:sha256(target/name) for name in manifest['artifacts_sha256']}
    write_json(target/'analysis.json',manifest);write_json(target/'player_recovery.json',records)
    write_json(target/'screening.json',screen_coverage(observations,pd.read_csv(target/'reference_template.csv')))
    render_report(target);write_json(target/'status.json',{'state':'complete','validation':'pending','inference_reused':True});print(f'RESULT={target}',flush=True);return target
