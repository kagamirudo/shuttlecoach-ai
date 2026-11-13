"""Bounded past-only regression previews. Forecasts never become observations."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .media import sha256, write_json
from .timeline import crosses_cut

HORIZON_S = .3
HISTORY_S = .6


def fit_motion(history, seconds):
    if len(history)<4:return None
    data=np.asarray(history,dtype=float);t=data[:,0]-data[-1,0]
    if data[-1,0]-data[0,0]<.25 or seconds-data[-1,0]>HORIZON_S+1e-8:return None
    design=np.c_[np.ones(len(t)),t];coef=np.linalg.lstsq(design,data[:,1:3],rcond=None)[0]
    residual=float(np.sqrt(np.mean(np.sum((design@coef-data[:,1:3])**2,axis=1))))
    if residual>.25 or np.linalg.norm(coef[1])>8:return None
    age=seconds-data[-1,0];point=np.array([1,age])@coef
    if not (-1.5<=point[0]<=7.6 and -1.5<=point[1]<=14.9):return None
    return point,age,residual


def motion_estimates(observations, meta):
    columns=['rally_id','player','side','source_frame','source_time_s','original_source_time_s',
             'court_x_m','court_y_m','prediction_age_s','fit_rmse_m','position_source']
    rows=[]
    for (rally,player,side),part in observations.groupby(['rally_id','player','side'],sort=False):
        history=[];previous=None
        for row in part.sort_values('source_frame').itertuples():
            if previous is not None and (row.source_time_s-previous.source_time_s>.25 or crosses_cut(meta,previous.source_frame,row.source_frame)):history=[]
            available=bool(getattr(row,'tracking_observed',row.observed))
            if available and np.isfinite([row.court_x_m,row.court_y_m]).all():
                if history and row.source_time_s-history[-1][0]>.25:history=[]
                history.append([row.source_time_s,row.court_x_m,row.court_y_m])
                history=[r for r in history if row.source_time_s-r[0]<=HISTORY_S+1e-8]
            else:
                fitted=fit_motion(history,row.source_time_s)
                if fitted is not None:
                    point,age,error=fitted
                    rows.append({'rally_id':int(rally),'player':player,'side':side,'source_frame':int(row.source_frame),
                        'source_time_s':float(row.source_time_s),'original_source_time_s':float(getattr(row,'original_source_time_s',row.source_time_s)),
                        'court_x_m':float(point[0]),'court_y_m':float(point[1]),'prediction_age_s':float(age),
                        'fit_rmse_m':error,'position_source':'estimated_past_only_regression'})
            previous=row
    return pd.DataFrame(rows,columns=columns)


def masked_diagnostic(observations,meta):
    """Hide three consecutive real positions at fixed time intervals, never use their future."""
    masked=observations.copy(deep=True);targets=[]
    for _,part in masked.groupby(['rally_id','player','side'],sort=False):
        last=-1e9;indices=part.sort_values('source_frame').index.to_list()
        for i in range(6,len(indices)-3):
            group=masked.loc[indices[i-6:i+3]]
            times=group.source_time_s.to_numpy();frames=group.source_frame.to_numpy()
            if times[-1]-last<5 or not group.observed.eq(True).all() or (np.diff(times)>.15).any() or any(crosses_cut(meta,int(a),int(b)) for a,b in zip(frames,frames[1:])):continue
            selected=indices[i:i+3];targets.extend(selected);masked.loc[selected,['observed','tracking_observed']]=False;last=times[-1]
    predictions=motion_estimates(masked,meta)
    keys=['source_frame','side','player','rally_id']
    reference=observations.loc[targets]
    joined=predictions.merge(reference,on=keys,suffixes=('_estimate','_observed'),validate='one_to_one')
    errors=np.hypot(joined.court_x_m_estimate-joined.court_x_m_observed,joined.court_y_m_estimate-joined.court_y_m_observed)
    return {'masked_positions':len(targets),'predicted_positions':len(joined),
            'median_difference_m':float(errors.median()) if len(errors) else None,
            'p90_difference_m':float(errors.quantile(.9)) if len(errors) else None,
            'limitation':'Diagnostic against withheld tracker coordinates, not manual truth or validation of real occlusions. Parameters fixed before this run. No tracking gate changes.'}


def write_motion_preview(run):
    from .runner import overlay,raw_tracks
    run=Path(run);manifest=json.loads((run/'analysis.json').read_text());meta=manifest['video']
    if sha256(meta['path'])!=meta['sha256']:raise ValueError('Source video changed.')
    for name,digest in manifest['artifacts_sha256'].items():
        if sha256(run/name)!=digest:raise ValueError(f'Player artifact changed: {name}')
    observations=pd.read_csv(run/'observations.csv');estimates=motion_estimates(observations,meta)
    estimates.to_csv(run/'motion_estimates.csv',index=False)
    timing=overlay(Path(meta['path']),run/'motion_preview.mp4',raw_tracks(run/'tracks.duckdb'),meta,
                   manifest['start_frame'],manifest['end_frame_exclusive'],np.asarray(manifest['homography']),
                   manifest.get('tracking_stride',1),estimates.to_dict('records'))
    payload={'method':'Past-only local linear least-squares regression; no future frames or neural retraining.',
             'history_s':HISTORY_S,'maximum_horizon_s':HORIZON_S,'estimated_positions':len(estimates),
             'source_sha256':meta['sha256'],'observations_sha256':sha256(run/'observations.csv'),
             'overlay_timing':timing,'diagnostic':masked_diagnostic(observations,meta),
             'artifacts_sha256':{name:sha256(run/name) for name in ['motion_estimates.csv','motion_preview.mp4']},
             'validation_credit':False,'coaching_evidence':False,
             'limitations':'Visual continuity only. Yellow hollow markers are estimated positions, not detected feet. Stops after 0.3 s, at cuts, athlete/rally changes, high residuals or implausible speed. Fit residual is not a confidence interval. Long occlusions stay empty.'}
    write_json(run/'motion.json',payload)
    print(f'RESULT={run.resolve()}',flush=True)
    return payload
