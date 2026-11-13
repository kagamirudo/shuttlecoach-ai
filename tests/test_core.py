from fractions import Fraction
import copy
import json

import av
import numpy as np
import pandas as pd
import pytest

from shuttlecoach.analysis import build_observations, summarize, validate_config
from shuttlecoach.media import probe, read_frame, verify_overlay, verify_tracker_frames, youtube_id
from shuttlecoach.runner import overlay
from shuttlecoach.upstream import court
from shuttlecoach.validation import evaluate, sample_template, screen_coverage


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "variable.mp4"
    with av.open(str(path), "w") as output:
        stream = output.add_stream("libx264", rate=30)
        stream.width, stream.height, stream.pix_fmt = 640, 480, "yuv420p"
        stream.time_base = stream.codec_context.time_base = Fraction(1,90000)
        stream.options = {"bf":"0"}
        for index, stamp in enumerate([0,3000,7500,10500,18000,21000]):
            image = np.full((480,640,3),index*30,np.uint8)
            frame = av.VideoFrame.from_ndarray(image,format="rgb24")
            frame.pts, frame.time_base = stamp,Fraction(1,90000)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    return path


def config(meta):
    return {"singles_confirmed":True,"camera_stable_confirmed":True,
            "calibration":{"frame":0,"reviewed":True,"corners_px":[[80,450],[560,450],[490,40],[150,40]]},
            "rallies":[{"start_s":0.,"end_s":meta["duration_s"],"near_player":"A","far_player":"B"}]}


def raw(frame,side,x=3.,y=3.):
    points=np.zeros((17,3))
    points[:,2]=0.9
    return {"frame_num":frame,"player_id":side,"court_x":x,"court_y":y,"img_x":300.,"img_y":200.,
            "pose_conf":0.9,"keypoints":points.flatten().tolist(),"bbox":[300,200,60,100]}


def test_vfr_overlay_and_exact_seek(video,tmp_path):
    meta=probe(video)
    assert not np.allclose(np.diff(meta["times"]),1/meta["fps"])
    assert meta["frame_count"]==6
    for index in [0,2,5]:
        a=read_frame(video,index)
        b=read_frame(video,index,meta["times"],meta["pts_origin_s"])
        np.testing.assert_array_equal(a,b)
    h=validate_config(config(meta),meta)
    result=overlay(video,tmp_path/"overlay.mp4",[],meta,1,6,h)
    assert result["frames"]==5
    assert result["max_timestamp_error_s"]<0.0001
    with pytest.raises(ValueError,match="frame count"):
        verify_overlay(tmp_path/"overlay.mp4",meta["times"])


def test_tracker_decoder_is_checked_against_source_pts(video):
    meta=probe(video)
    result=verify_tracker_frames(meta,0,meta["frame_count"])
    assert result["status"]=="passed"
    assert max(p["timestamp_error_s"] for p in result["probes"])<0.002
    meta["times"]=np.asarray(meta["times"])+0.2
    with pytest.raises(ValueError):
        verify_tracker_frames(meta,0,meta["frame_count"])


def test_sampled_overlay_retains_source_times_without_filling_frames(video,tmp_path):
    meta=probe(video)
    h=validate_config(config(meta),meta)
    result=overlay(video,tmp_path/"sampled.mp4",[],meta,0,6,h,stride=2)
    assert result["frames"]==3
    assert result["max_timestamp_error_s"]<0.0001
    assert verify_tracker_frames(meta,0,6,stride=2)["status"]=="passed"


def test_overlay_uses_the_same_rejection_rules_as_measurements(video,tmp_path,monkeypatch):
    meta=probe(video);h=validate_config(config(meta),meta)
    rows=[raw(0,'near'),raw(1,'far',x=-1.),raw(2,'near'),raw(3,'far',x=-1.)]
    rows[0]['keypoints']=[0.]*51;rows[3]['identity_verified']=True
    drawn=[]
    monkeypatch.setattr('shuttlecoach.runner.draw_skeleton',lambda image,points,color:drawn.append(points))
    overlay(video,tmp_path/'accepted.mp4',rows,meta,0,6,h)
    assert len(drawn)==2  # Accepted position and explicitly verified off-court player only.


def test_estimate_preview_marks_forecasts_without_overriding_real_poses_or_source_pts(video,tmp_path,monkeypatch):
    meta=probe(video);h=validate_config(config(meta),meta);labels=[]
    import cv2
    original=cv2.putText
    def capture(image,text,*args,**kwargs):
        labels.append(text)
        return original(image,text,*args,**kwargs)
    monkeypatch.setattr(cv2,'putText',capture)
    estimates=[{'source_frame':f,'side':'near','court_x_m':3.,'court_y_m':3.,'prediction_age_s':.1} for f in [0,1]]
    result=overlay(video,tmp_path/'preview.mp4',[raw(0,'near')],meta,0,6,h,estimates=estimates)
    assert result['max_timestamp_error_s']<.0001 and result['frames']==6
    assert sum(t.startswith('ESTIMATED near') for t in labels)==1
    assert any(t.startswith('ESTIMATE PREVIEW') for t in labels)


def test_sampled_occupancy_counts_time_but_never_invents_detections():
    meta={"times":[0.,.1,.2,.3,.4,.5],"duration_s":.6}
    intervals=[{"start_s":0.,"end_s":.6,"near_player":"A","far_player":"B"}]
    rows=build_observations([raw(0,"near"),raw(4,"near")],meta,intervals,[0,2,4])
    assert set(rows.source_frame)=={0,2,4}
    near=next(s for s in summarize(rows) if s["side"]=="near")
    assert near["sampled_time_s"]==pytest.approx(.6)
    assert near["observed_time_s"]==pytest.approx(.4)
    assert near["coverage"]==pytest.approx(2/3)


def test_missing_frames_reduce_time_coverage_without_becoming_stationary():
    meta={"times":[0.,0.1,0.4,0.5],"duration_s":0.6}
    rallies=[{"start_s":0.,"end_s":0.6,"near_player":"A","far_player":"B"}]
    table=build_observations([raw(0,"near"),raw(2,"near")],meta,rallies)
    summary=next(s for s in summarize(table) if s["side"]=="near")
    assert summary["observed_time_s"]==pytest.approx(0.2)
    assert summary["coverage"]==pytest.approx(1/3)
    assert len(table)==8
    assert sum(summary["occupancy"].values())==pytest.approx(1.)
    assert table.loc[table.reason=="missing","court_x_m"].isna().all()


def test_low_ankle_confidence_and_outside_court_are_excluded():
    rows=[raw(0,"near"),raw(1,"near",x=-0.1)]
    rows[0]["keypoints"]=[0.]*51
    table=build_observations(rows,{"times":[0.,0.1],"duration_s":0.2},
                             [{"start_s":0.,"end_s":0.2,"near_player":"A","far_player":"B"}])
    assert not table.observed.any()
    assert {"low_confidence","outside_court","missing"}<=set(table.reason)


def test_rallies_do_not_double_count_or_include_breaks():
    meta={"times":list(np.arange(0,10,0.1)),"duration_s":10.}
    rallies=[{"start_s":1.,"end_s":2.,"near_player":"A","far_player":"B"},
             {"start_s":8.,"end_s":9.,"near_player":"B","far_player":"A"}]
    table=build_observations([],meta,rallies)
    assert set(table.source_frame)==set(range(10,20))|set(range(80,90))
    assert table.loc[table.rally_id==2].iloc[0].player=="B"
    template=sample_template(table,count=16)
    assert len(template)==32
    assert {10,19,80,89}<=set(template.source_frame)


@pytest.mark.parametrize("change,match",[
    (lambda c:c.update(singles_confirmed=False),"singles"),
    (lambda c:c.update(camera_stable_confirmed=False),"fixed camera"),
    (lambda c:c["calibration"].update(reviewed=False),"confirm"),
    (lambda c:c["calibration"].update(corners_px=[[0,0]]*4),"quadrilateral"),
    (lambda c:c["rallies"][0].update(end_s=float("nan")),"start/end"),
    (lambda c:c["rallies"][0].update(far_player="A"),"different names"),
])
def test_bad_config_is_rejected(video,change,match):
    meta=probe(video)
    value=config(meta)
    change(value)
    with pytest.raises(ValueError,match=match):
        validate_config(value,meta)


@pytest.fixture
def references():
    h=np.eye(3)
    rows=[]
    for frame in range(100):
        for side in ["near","far"]:
            rows.append({"source_frame":frame,"side":side,"observed":True,"image_x":3.,"image_y":3.,"court_x_m":3.,"court_y_m":3.})
    observations=pd.DataFrame(rows)
    observations["rally_id"]=1
    template=sample_template(observations)
    labels=template.copy()
    labels[["visible","identity_correct"]]=True
    labels[["image_x","image_y"]]=3.
    checks=[{"image_x":x,"image_y":y,"court_x_m":x,"court_y_m":y} for x,y in [(1,2),(5,2),(1,11),(5,11)]]
    return observations,labels,checks,h,template


def test_complete_validation_can_pass(references):
    result=evaluate(*references)
    assert result["status"]=="passed"
    assert result["matched_positions"]==200


def test_verified_tracking_still_requires_position_and_identity_accuracy(references):
    observations,labels,checks,h,template=references
    observations=observations.copy();observations['tracking_observed']=True;observations['observed']=False
    assert evaluate(observations,labels,checks,h,template)['status']=='passed'
    observations['court_x_m']+=5
    result=evaluate(observations,labels,checks,h,template)
    assert result['matched_positions']==200 and not result['criteria']['median_position_error_le_0_75m']
    labels=labels.copy();labels.loc[0,'identity_correct']=False
    assert not evaluate(observations,labels,checks,h,template)['criteria']['no_reviewed_identity_errors']


def test_coverage_screen_never_claims_accuracy_pass(references):
    observations,labels,checks,h,template=references
    assert screen_coverage(observations,template)["status"]=="manual_review_required"
    observations.loc[observations.source_frame<18,"observed"]=False
    result=screen_coverage(observations,template)
    assert result["status"]=="failed"
    assert result["upper_bound_on_both_players_correct_rate"]==pytest.approx(.82)


def test_missing_predictions_are_counted_in_validation(references):
    observations,labels,checks,h,template=references
    observations.loc[observations.source_frame<10,"observed"]=False
    result=evaluate(observations,labels,checks,h,template)
    assert result["status"]=="failed"
    assert result["both_players_correct_rate"]==pytest.approx(.9)
    assert result["matched_positions"]==180


def test_identity_error_fails_even_with_perfect_position(references):
    observations,labels,checks,h,template=references
    labels.loc[0,"identity_correct"]=False
    result=evaluate(observations,labels,checks,h,template)
    assert not result["criteria"]["no_reviewed_identity_errors"]


def test_missing_detections_reduce_coverage_without_becoming_identity_switches(references):
    observations,labels,checks,h,template=references
    observations.loc[observations.source_frame<2,"observed"]=False
    labels.loc[labels.source_frame<2,"identity_correct"]=False
    result=evaluate(observations,labels,checks,h,template)
    assert result["both_players_correct_rate"]==pytest.approx(.98)
    assert result["criteria"]["no_reviewed_identity_errors"]
    assert result["status"]=="passed"


def test_wrong_person_during_occlusion_is_still_an_identity_error(references):
    observations,labels,checks,h,template=references
    labels.loc[0,"visible"]=False
    labels.loc[0,"identity_correct"]=False
    result=evaluate(observations,labels,checks,h,template)
    assert not result["criteria"]["no_reviewed_identity_errors"]


@pytest.mark.parametrize("kind",["partial","duplicate","blank","nonfinite","fractional"])
def test_invalid_reference_sets_fail_closed(references,kind):
    observations,labels,checks,h,template=references
    if kind=="partial": labels=labels.iloc[:-1]
    elif kind=="duplicate": labels=pd.concat([labels,labels.iloc[:1]])
    elif kind=="blank":
        labels["visible"]=labels.visible.astype(object)
        labels.loc[0,"visible"]=None
    elif kind=="nonfinite": labels.loc[0,"image_x"]=float("inf")
    elif kind=="fractional":
        labels["source_frame"]=labels.source_frame.astype(float)
        labels.loc[0,"source_frame"]=0.5
    with pytest.raises(ValueError):
        evaluate(observations,labels,checks,h,template)


def test_fitted_corners_cannot_validate_their_own_calibration(references):
    observations,labels,checks,h,template=references
    checks=[{"image_x":x,"image_y":y,"court_x_m":x,"court_y_m":y} for x,y in court.COURT_CORNERS_M]
    with pytest.raises(ValueError,match="other than"):
        evaluate(observations,labels,checks,h,template)


@pytest.mark.parametrize("url",["http://youtube.com/watch?v=RZaZ7GaPJU8","https://youtube.com.evil.test/watch?v=RZaZ7GaPJU8", "https://youtube.com/playlist?list=x"])
def test_downloader_rejects_non_video_urls(url):
    with pytest.raises(ValueError): youtube_id(url)


def test_downloader_extracts_single_video_from_playlist_link():
    assert youtube_id("https://www.youtube.com/watch?v=RZaZ7GaPJU8&list=abc&index=5")=="RZaZ7GaPJU8"


@pytest.mark.parametrize('end',[300.,375.,600.])
def test_edited_tracking_duration_and_names_are_valid_up_to_ten_minutes(end):
    meta={'width':640,'height':480,'times':list(np.arange(0,701,.1)),
          'frame_count':7010,'duration_s':701.}
    value=config(meta)
    value['rallies'][0].update(end_s=end,near_player='Shi Yu Qi',far_player='Random China')
    assert validate_config(value,meta).shape==(3,3)
    assert value['rallies'][0]['near_player']=='Shi Yu Qi'


@pytest.mark.parametrize('mapped',[False,True])
def test_tracking_duration_limit_is_explicit_and_shared_by_both_sources(mapped):
    meta={'width':640,'height':480,'times':list(np.arange(0,701,.1)),
          'frame_count':7010,'duration_s':701.}
    value=config(meta);value['rallies'][0]['end_s']=601.
    if mapped:value['source_frame_map']={'path':'map.csv'}
    with pytest.raises(ValueError,match=r'601.0s.*600s'):
        validate_config(value,meta)


def test_player_overview_weights_time_and_separates_offcourt_tracking():
    from shuttlecoach.analysis import player_overview
    rows=pd.DataFrame({'player':['A','A','A','A','B'],'side':['near','far','far','far','far'],
        'sample_duration_s':[1.,3.,2.,4.,5.],'observed':[True,True,False,False,False],
        'tracking_observed':[True,True,True,False,False],'zone':['front','rear',None,None,None]})
    a,b=player_overview(rows)
    assert a['selected_s']==10 and a['tracked_s']==6 and a['coverage']==.6 and a['unavailable_s']==4
    assert a['in_court_s']==4 and a['occupancy']=={'front':.25,'middle':0.,'rear':.75}
    assert a['sides']==['near','far']
    assert b['coverage']==0 and all(v is None for v in b['occupancy'].values())
    assert player_overview(rows.drop(columns='tracking_observed'))[0]['tracked_s']==4
