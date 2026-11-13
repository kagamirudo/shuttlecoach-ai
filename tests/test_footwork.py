import copy

import numpy as np
import pandas as pd
import pytest

from shuttlecoach.footwork import analyze_footwork, pose_features, reviewed_footwork, footwork_suggestions


def pose(frame=0):
    p=np.ones((17,3));p[:,2]=.9
    p[5,:2]=[45,20];p[6,:2]=[55,20];p[11,:2]=[45,60];p[12,:2]=[55,60]
    p[13,:2]=[20,80];p[14,:2]=[55,80];p[15,:2]=[20,110];p[16,:2]=[55,100]
    return {'frame_num':frame,'player_id':'near','bbox':[40,60,80,100],'keypoints':p.ravel().tolist()}


def test_pose_candidates_require_visible_legs_and_are_camera_plane_geometry():
    row=pose();features=pose_features(row)
    assert features['left_knee_image_deg']==pytest.approx(128.66,abs=.01)
    assert features['right_knee_image_deg']==pytest.approx(180)
    assert features['stance_body_heights']>.35
    points=np.asarray(row['keypoints']).reshape(17,3);points[15,2]=.1;row['keypoints']=points.ravel().tolist()
    assert pose_features(row) is None


def test_events_do_not_bridge_missing_poses_or_cuts_and_keep_original_time():
    raw=[pose(i) for i in range(5)]
    # A large foot shift on the first frame after an occlusion/cut is not a step event.
    for i in [2,4]:
        points=np.asarray(raw[i]['keypoints']).reshape(17,3);points[15,0]+=20;raw[i]['keypoints']=points.ravel().tolist()
    observations=pd.DataFrame([{'rally_id':1,'side':'near','player':'A','source_frame':i,
        'source_time_s':i/10,'sample_duration_s':.1,'observed':i!=1} for i in range(5)])
    original=observations.copy(deep=True)
    result=analyze_footwork(raw,observations,{'times':[i/10 for i in range(5)],'original_times':[500+i/10 for i in range(5)],'cut_frames':[4]})
    assert all(r['source_frame'] not in [2,4] or r['kind']=='lunge' for r in result['events'])
    assert result['events'][0]['original_source_time_s']==500.
    assert result['summaries'][0]['pose_seconds']==pytest.approx(.4)
    assert 'physical balance' in result['limitations']
    pd.testing.assert_frame_equal(observations,original)


def review_fixture():
    events=[];intervals=[];labels=[];rows=[]
    for i in range(3):
        start=10*i
        intervals.append({'start_s':start,'end_s':start+10,'original_start_s':100+start,'original_end_s':110+start})
        for kind in ['stance_opening','lunge']:
            event={'rally_id':i+1,'player':'A','side':'near','source_frame':100*i,'source_time_s':start+1,
                   'original_source_time_s':101+start,'kind':kind,'event_id':f'{i}:{kind}','segment':1}
            events.append(event)
            labels.append({'event_id':event['event_id'],'confirmed':True,'opponent_hit_s':100.95+start if kind=='stance_opening' else None,
                'landing_s':101.05+start if kind=='stance_opening' else None,'ready_s':102+start if kind=='lunge' else None,
                'readiness':'Extra adjustment visible'})
        rows.append({'rally_id':i+1,'player':'A','side':'near','observed':True,'sample_duration_s':10.,'zone':'front'})
    return {'evidence_sha256':'frozen','events':events}, {'evidence_sha256':'frozen','reviewer':'Tester','events':labels}, intervals,pd.DataFrame(rows)


def test_only_complete_rally_reviews_can_support_timing_and_practice():
    foot,review,intervals,observations=review_fixture()
    partial=copy.deepcopy(review);partial['events'].pop()
    result=reviewed_footwork(foot,partial,intervals)
    assert result['status']=='partial' and result['rally_ids']==[1,2]
    assert not footwork_suggestions(result,observations,intervals)
    result=reviewed_footwork(foot,review,intervals)
    assert result['status']=='reviewed'
    assert result['events'][0]['split_step_offset_s']==pytest.approx(.1)
    suggestions=footwork_suggestions(result,observations,intervals)
    assert {r['title'] for r in suggestions}=={'Reviewed split-step timing','Reviewed lunge recovery','Video-reviewed readiness'}
    assert suggestions[0]['evidence'][0]['start_s']==101.
    observations.loc[0,'observed']=False
    assert not footwork_suggestions(result,observations,intervals)


@pytest.mark.parametrize('problem',['stale','duplicate','unknown','nonfinite','outside','wrong_kind','before_lunge','string_bool','no_reviewer'])
def test_review_rejects_stale_or_invented_timing(problem):
    foot,review,intervals,_=review_fixture()
    if problem=='stale':
        review['evidence_sha256']='old'
        assert reviewed_footwork(foot,review,intervals)['status']=='stale'
        return
    if problem=='duplicate':review['events'].append(review['events'][0].copy())
    if problem=='unknown':review['events'][0]['event_id']='fabricated'
    if problem=='nonfinite':review['events'][0]['landing_s']=float('nan')
    if problem=='outside':review['events'][0]['opponent_hit_s']=99.
    if problem=='wrong_kind':review['events'][1]['landing_s']=101.1
    if problem=='before_lunge':review['events'][1]['ready_s']=100.5
    if problem=='string_bool':review['events'][0]['confirmed']='true'
    if problem=='no_reviewer':review['reviewer']=''
    with pytest.raises(ValueError):reviewed_footwork(foot,review,intervals)


def test_partial_rallies_and_rejected_candidates_never_support_footwork_advice():
    foot,review,intervals,observations=review_fixture()
    for label in review['events']:label['confirmed']=False
    result=reviewed_footwork(foot,review,intervals)
    assert result['status']=='reviewed' and not result['events']
    assert not footwork_suggestions(result,observations,intervals)
    for label in review['events']:label['confirmed']=True
    intervals[0]['partial']=True
    assert not footwork_suggestions(reviewed_footwork(foot,review,intervals),observations,intervals)


@pytest.mark.parametrize('gate',['passed','failed'])
def test_coaching_writer_requires_tracking_gates_even_with_confirmed_footwork(tmp_path,monkeypatch,gate):
    import json
    from shuttlecoach.coaching import write_coaching_report,evidence_signature
    from shuttlecoach.media import sha256
    foot,review,intervals,observations=review_fixture()
    source=tmp_path/'source.mp4';source.write_bytes(b'source')
    observations.to_csv(tmp_path/'observations.csv',index=False)
    (tmp_path/'tracks.duckdb').write_bytes(b'cached poses')
    (tmp_path/'reviewed_references.csv').write_text('source_frame,side\n0,near\n')
    (tmp_path/'reviewed_calibration_checks.json').write_text('[]')
    (tmp_path/'reference_template.csv').write_text('source_frame,side\n0,near\n')
    (tmp_path/'shuttle.json').write_text(json.dumps({'source_sha256':sha256(source),'artifacts_sha256':{},'detected_frames':1,'frame_count':1}))
    (tmp_path/'shuttle.csv').write_text('source_frame\n0\n')
    (tmp_path/'shuttle_reviewed_references.csv').write_text('source_frame\n0\n')
    (tmp_path/'shuttle_reference_template.csv').write_text('source_frame\n0\n')
    (tmp_path/'footwork_review.json').write_text(json.dumps(review))
    manifest={'video':{'path':str(source),'sha256':sha256(source),'times':[0.,30.],'width':640,'height':480},'config':{'rallies':intervals,'calibration':{}},
              'homography':np.eye(3).tolist(),'artifacts_sha256':{'observations.csv':sha256(tmp_path/'observations.csv')},
              'overlay_source_start_s':0.,'end_frame_exclusive':2}
    (tmp_path/'analysis.json').write_text(json.dumps(manifest))
    foot.update(summaries=[],intervals=intervals,limitations='Camera-plane review only.')
    monkeypatch.setattr('shuttlecoach.footwork.write_footwork',lambda run:foot)
    monkeypatch.setattr('shuttlecoach.coaching.evaluate',lambda *args:{'status':gate,'criteria':{'reviewed':gate=='passed'}})
    monkeypatch.setattr('shuttlecoach.shuttle.evaluate_shuttle',lambda *args:{'status':'passed','criteria':{'reviewed':True}})
    payload=write_coaching_report(tmp_path)
    practices=[r for r in payload['suggestions'] if r.get('kind')=='footwork']
    assert bool(practices)==(gate=='passed')
    assert payload['evidence_signature']==evidence_signature(tmp_path)
    (tmp_path/'footwork_review.json').write_text('{}')
    assert payload['evidence_signature']!=evidence_signature(tmp_path)
