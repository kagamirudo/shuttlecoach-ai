import json
from fractions import Fraction

import av
import numpy as np
import pandas as pd
import pytest

from shuttlecoach.coaching import coaching_payload, write_coaching_report, reviewed_play
from shuttlecoach.media import probe, sha256
from shuttlecoach.shuttle import evaluate_shuttle, reference_template, render_shuttle, reuse_shuttle, court_zone, court_mask, project_shuttle
from shuttlecoach.upstream import court
from shuttlecoach.shuttle import weighted_heatmaps, projection_plot, heatmap_candidates


def test_projection_plot_uses_sample_cadence_and_breaks_real_gaps(tmp_path):
    table=pd.DataFrame({'source_frame':[10,12,14,20], 'source_time_s':[0.,1/30,2/30,5/30],
        'detected':[True]*4,'image_x':[100.,101.,102.,103.],'image_y':[200.]*4,
        'projection_x_m':[2.,2.1,2.2,2.3],'projection_y_m':[3.]*4})
    assert projection_plot(tmp_path,table)==2
    table.loc[1,'detected']=False
    assert projection_plot(tmp_path,table)==0


def test_all_components_retain_smaller_target_beside_larger_neighbor():
    heat=np.zeros((288,512),np.float32);heat[50:54,50:54]=.9;heat[100:102,300:302]=.8
    points=heatmap_candidates(heat,1280,720)
    assert len(points)==2 and points[1]['image_x']==752 and points[1]['image_y']==252


def test_reviewed_play_clips_sample_durations_and_excludes_pauses():
    rows=pd.DataFrame({'source_time_s':[0.,1.,2.,3.], 'sample_duration_s':[1.]*4,
                       'side':['far']*4,'player':['old']*4,'rally_id':[1]*4})
    original=rows.copy(deep=True)
    cfg={'rallies':[{'start_s':0.,'end_s':4.}],
         'reviewed_rallies':[{'start_s':.5,'end_s':1.5,'near_player':'Leon','far_player':'Gary'},
                             {'start_s':3.5,'end_s':4.,'near_player':'Leon','far_player':'Gary'}]}
    play,_=reviewed_play(rows,cfg)
    assert play.source_time_s.tolist()==[0.,1.,3.]
    assert play.sample_duration_s.tolist()==[.5,.5,.5]
    assert play.player.tolist()==['Gary']*3
    pd.testing.assert_frame_equal(rows,original)


def court_parent():
    corners=[[100,400],[500,400],[450,100],[150,100]]
    return {'video':{'width':640,'height':480,'sha256':'source-hash'},
            'config':{'calibration':{'corners_px':corners}},'homography':court.compute_homography(np.asarray(corners)).tolist(),
            'start_frame':0,'end_frame_exclusive':10}


def test_weighted_overlap_normalizes_edges_and_does_not_duplicate_timestamps():
    import torch
    indices=torch.tensor([[[0,0],[0,1]],[[0,1],[0,2]],[[0,2],[0,3]]])
    heat=torch.tensor([[[[1.]],[[2.]]],[[[4.]],[[6.]]],[[[8.]],[[10.]]]])
    result=weighted_heatmaps(lambda x:x,[(indices,heat)],4,'cpu',torch.tensor([1.,2.]))
    assert result[:,0,0].tolist()==pytest.approx([1.,8/3,20/3,10.])
    with pytest.raises(ValueError,match='uncovered'):
        weighted_heatmaps(lambda x:x,[(indices,heat)],5,'cpu',torch.tensor([1.,2.]))


def test_quality_request_preserves_completed_run_config(tmp_path,monkeypatch):
    from shuttlecoach.cli import main
    path=tmp_path/'analysis.json';original=json.dumps({'config':{'shuttle_quality':False}});path.write_text(original)
    (tmp_path/'shuttle.json').write_text('{}')
    monkeypatch.setattr('sys.argv',['shuttlecoach','shuttle',str(tmp_path),'--quality'])
    with pytest.raises(SystemExit):main()
    assert path.read_text()==original


@pytest.fixture
def shuttle_references():
    template=reference_template(range(100))
    labels=template.copy();labels['visibility']=['visible']*50+['not_visible']*50
    labels.loc[:49,['image_x','image_y']]=[320.,200.]
    predictions=pd.DataFrame({'source_frame':range(100),'detected':[True]*50+[False]*50,
                             'image_x':[320.]*50+[np.nan]*50,'image_y':[200.]*50+[np.nan]*50})
    return predictions,labels,template


def test_shuttle_gate_measures_visibility_and_location(shuttle_references):
    predictions,labels,template=shuttle_references
    assert evaluate_shuttle(predictions,labels,template,1280,720)['status']=='passed'
    predictions.loc[:9,'image_x']=500.
    result=evaluate_shuttle(predictions,labels,template,1280,720)
    assert result['status']=='failed'
    assert result['false_positive']==10 and result['false_negative']==10
    assert result['precision']==pytest.approx(.8)


def test_shuttle_all_invisible_cannot_pass(shuttle_references):
    p,l,t=shuttle_references;l['visibility']='not_visible';p['detected']=False
    assert evaluate_shuttle(p,l,t,1280,720)['status']=='failed'


@pytest.mark.parametrize('problem',['unknown','cherry_pick','duplicate','outside','fractional'])
def test_invalid_shuttle_references_fail_closed(shuttle_references,problem):
    p,l,t=shuttle_references
    if problem=='unknown':l.loc[0,'visibility']='unreviewed'
    if problem=='cherry_pick':l=l.iloc[1:]
    if problem=='duplicate':l.loc[1,'source_frame']=0
    if problem=='outside':l.loc[0,'image_x']=1280
    if problem=='fractional':l['source_frame']=l.source_frame.astype(float);l.loc[0,'source_frame']=.5
    with pytest.raises(ValueError):evaluate_shuttle(p,l,t,1280,720)


def pattern_data():
    rows=[];intervals=[]
    for i in range(3):
        intervals.append({'start_s':i*20.,'end_s':(i+1)*20.})
        for side,player in [('near','A'),('far','B')]:
            rows.append({'rally_id':i+1,'side':side,'player':player,'observed':True,'sample_duration_s':20.,'zone':'rear'})
    return pd.DataFrame(rows),{'rallies':intervals}, {'status':'passed','criteria':{'reviewed':True}}


def test_failed_or_pending_gate_never_emits_coaching():
    table,config,passed=pattern_data()
    for other in [{'status':'failed'},{'status':'pending'},{'status':'passed','criteria':{'x':False}},{'status':'passed'}]:
        for players,shuttle in [(passed,other),(other,passed)]:
            result=coaching_payload(table,config,players,shuttle)
            assert result['gate']=='blocked' and not result['suggestions'] and not result['coaching_enabled']


def test_both_gates_and_repeated_rallies_required_for_suggestions():
    table,config,passed=pattern_data()
    result=coaching_payload(table,config,passed,passed)
    assert result['coaching_enabled'] and len(result['suggestions'])==2
    assert len(result['suggestions'][0]['evidence'])==3
    config['includes_between_point_pauses']=True
    assert not coaching_payload(table,config,passed,passed)['coaching_enabled']
    del config['includes_between_point_pauses']
    assert not coaching_payload(table[table.rally_id<3],config,passed,passed)['coaching_enabled']


def test_short_rallies_need_pooled_duration_before_any_suggestion():
    table,config,passed=pattern_data()
    table['sample_duration_s']=6.
    assert not coaching_payload(table,config,passed,passed)['coaching_enabled']
    table=pd.concat([table.assign(rally_id=i+1) for i in range(5)],ignore_index=True)
    # One row per player per reviewed interval, six seconds each.
    table=table.drop_duplicates(['rally_id','player'])
    config['rallies']=[{'start_s':i*10.,'end_s':i*10.+6.} for i in range(5)]
    assert coaching_payload(table,config,passed,passed)['coaching_enabled']


def test_report_rejects_tampering_even_with_saved_pass(tmp_path):
    table,config,passed=pattern_data();table.to_csv(tmp_path/'observations.csv',index=False)
    source=tmp_path/'source.mp4';source.write_bytes(b'changed')
    (tmp_path/'analysis.json').write_text(json.dumps({'video':{'path':str(source),'sha256':'old','times':[0,1]},
        'config':config,'artifacts_sha256':{},'overlay_source_start_s':0.,'end_frame_exclusive':2}))
    (tmp_path/'validation.json').write_text(json.dumps(passed))
    result=write_coaching_report(tmp_path)
    assert not result['coaching_enabled']
    assert result['validation']['players']['status']=='failed'


def test_birdie_overlay_preserves_vfr_source_timestamps_and_missing_rows(tmp_path):
    source=tmp_path/'source.mp4'
    with av.open(str(source),'w') as output:
        stream=output.add_stream('libx264',rate=30);stream.width=640;stream.height=480;stream.pix_fmt='yuv420p'
        stream.time_base=stream.codec_context.time_base=Fraction(1,90000);stream.options={'bf':'0'}
        for t in [0,3000,9000,15000]:
            frame=av.VideoFrame.from_ndarray(np.zeros((480,640,3),dtype=np.uint8),format='rgb24')
            frame.pts=t;frame.time_base=Fraction(1,90000)
            for packet in stream.encode(frame):output.mux(packet)
        for packet in stream.encode():output.mux(packet)
    meta=probe(source)
    rows=pd.DataFrame({'source_frame':range(4),'detected':[True,False,False,True],
                       'image_x':[200,np.nan,np.nan,400],'image_y':[100,np.nan,np.nan,200]})
    result=render_shuttle(tmp_path,meta,rows,0,4)
    assert result['frames']==4 and result['max_timestamp_error_s']<.0001
    with av.open(str(tmp_path/'shuttle_overlay.mp4')) as output:
        frames=[f.to_ndarray(format='rgb24') for f in output.decode(video=0)]
    assert frames[0][85:115,185:215].max()>100
    assert frames[1][85:115,185:215].max()<20
    (tmp_path/'analysis.json').write_text(json.dumps(court_parent()))
    projected=project_shuttle(rows,court_parent())
    result=render_shuttle(tmp_path,meta,projected,0,4)
    assert result['frames']==4 and result['max_timestamp_error_s']<.0001
    with av.open(str(tmp_path/'shuttle_overlay.mp4')) as output:
        assert output.streams.video[0].width>640
    assert (tmp_path/'shuttle_zone_preview.png').is_file()
    sampled=projected[projected.source_frame.isin([0,2])]
    result=render_shuttle(tmp_path,meta,sampled,0,4,stride=2)
    assert result['frames']==2 and result['max_timestamp_error_s']<.0001


def test_shuttle_cache_reuse_requires_same_source_span_and_original_artifacts(tmp_path):
    source=tmp_path/'source';target=tmp_path/'renamed';source.mkdir();target.mkdir()
    (source/'shuttle.csv').write_text('source_frame,detected\n0,False\n')
    for name in ['shuttle_reference_template.csv','shuttle_overlay.mp4']:(source/name).write_bytes(b'original')
    (source/'shuttle_status.json').write_text(json.dumps({'state':'complete','validation':'pending'}))
    parent=court_parent()
    m={'source_sha256':'source-hash','start_frame':0,'end_frame_exclusive':10,'detection_zone':court_zone(parent),
       'artifacts_sha256':{'shuttle.csv':sha256(source/'shuttle.csv')}}
    (source/'shuttle.json').write_text(json.dumps(m))
    (target/'analysis.json').write_text(json.dumps(parent))
    assert reuse_shuttle(source,target)
    assert (target/'shuttle.csv').read_bytes()==(source/'shuttle.csv').read_bytes()
    parent['end_frame_exclusive']=11;(target/'analysis.json').write_text(json.dumps(parent))
    assert not reuse_shuttle(source,target)
    parent['end_frame_exclusive']=10;(target/'analysis.json').write_text(json.dumps(parent))
    (source/'shuttle.csv').write_text('altered')
    with pytest.raises(ValueError,match='artifact changed'):reuse_shuttle(source,target)


def test_court_mask_excludes_larger_neighbor_response_before_selection():
    parent=court_parent();mask=court_mask(parent,width=640,height=480)
    heat=np.zeros((480,640));heat[180:220,20:60]=1.;heat[249:252,299:302]=.8
    restricted=heat*mask
    assert heat[:,:100].sum()>heat[:,100:].sum()
    assert not restricted[:,:100].any() and restricted[250,300]==.8
    assert mask[50,300] and mask[100,150] and mask[400,500]
    assert not mask[50,25]


def test_expanded_zone_accepts_high_shots_but_keeps_left_neighbor_excluded():
    parent=court_parent();parent['video'].update(width=1280,height=720)
    corners=[[115,384],[677,701],[1125,299],[796,278]]
    parent['config']['calibration']['corners_px']=corners
    parent['homography']=court.compute_homography(np.asarray(corners)).tolist()
    mask=court_mask(parent,1280,720)
    assert mask[182,600] and mask[85,1042] and mask[0,800]
    assert not mask[190,430] and not mask[205,300] and not mask[100,1200]
    rows=pd.DataFrame({'detected':[True,True,True], 'image_x':[600.,430.,700.], 'image_y':[182.,190.,400.]})
    result=project_shuttle(rows,parent)
    assert result.detected.tolist()==[True,False,True]
    assert not result.loc[0,'projection_available']
    assert result.loc[0,['projection_x_m','projection_y_m']].isna().all()
    assert result.loc[2,'projection_available']


def test_association_zone_expands_upper_airspace_without_opening_low_neighbor_region():
    parent=court_parent();parent['video'].update(width=1280,height=720)
    parent['config']['calibration']['corners_px']=[[115,384],[677,701],[1125,299],[796,278]]
    parent['config'].update(shuttle_candidates=True,shuttle_association=True)
    mask=court_mask(parent,1280,720)
    assert mask[60,412] and mask[0,200] and mask[389,209]
    assert not mask[135,412] and not mask[242,297] and not mask[272,381]
    assert not mask[197,540] and not mask[147,547]


def test_projection_rejects_outside_and_never_clamps_or_fills_gaps():
    parent=court_parent()
    rows=pd.DataFrame({'source_frame':range(4),'detected':[True,True,False,True],
                       'image_x':[100.,25.,np.nan,500.],'image_y':[400.,250.,np.nan,400.]})
    result=project_shuttle(rows,parent)
    assert result.detected.tolist()==[True,False,False,True]
    assert np.allclose(result.loc[[0,3],['projection_x_m','projection_y_m']],[[0.,0.],[6.1,0.]],atol=1e-5)
    assert result.loc[[1,2],['projection_x_m','projection_y_m']].isna().all().all()
    assert result.loc[1,'zone_reason']=='outside_court_zone'
    assert not project_shuttle(rows.iloc[[1,2]],parent).detected.any()


def test_legacy_or_differently_calibrated_shuttle_cache_is_not_reused(tmp_path):
    source=tmp_path/'source';target=tmp_path/'new';source.mkdir();target.mkdir()
    (source/'shuttle_status.json').write_text(json.dumps({'state':'complete'}))
    (target/'analysis.json').write_text(json.dumps(court_parent()))
    m={'source_sha256':'source-hash','start_frame':0,'end_frame_exclusive':10,'artifacts_sha256':{}}
    (source/'shuttle.json').write_text(json.dumps(m))
    assert not reuse_shuttle(source,target)
    other=court_parent();other['config']['calibration']['corners_px'][0][0]=101
    m['detection_zone']=court_zone(other);(source/'shuttle.json').write_text(json.dumps(m))
    assert not reuse_shuttle(source,target)
