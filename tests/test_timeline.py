import json

import pandas as pd
import pytest

from shuttlecoach.analysis import build_observations
from shuttlecoach.media import sha256
from shuttlecoach.timeline import attach_timeline,source_seconds,source_frame_number,crosses_cut
from shuttlecoach.player_recovery import rally_slot


def test_original_clocks_and_cuts_survive_assembled_play(tmp_path):
    original=tmp_path/'original.mp4';original.write_bytes(b'original source')
    mapping=tmp_path/'map.csv'
    pd.DataFrame({'clip_frame':[0,1,2],'original_source_frame':[12,13,100],
                  'original_source_time_s':[.2,.3,2.]}).to_csv(mapping,index=False)
    config={'source_frame_map':{'path':str(mapping),'sha256':sha256(mapping)},
            'original_source':{'path':str(original),'sha256':sha256(original),'duration_s':3.,'width':640,'height':480}}
    meta=attach_timeline({'frame_count':3,'times':[0.,.1,.2],'duration_s':.3,'width':640,'height':480},config)
    assert source_seconds(meta,2)==2. and source_frame_number(meta,2)==100
    assert crosses_cut(meta,1,2) and not crosses_cut(meta,0,1)
    intervals=[{'start_s':0.,'end_s':.2,'near_player':'Leon','far_player':'Gary'},
               {'start_s':.2,'end_s':.3,'near_player':'Gary','far_player':'Leon'}]
    observations=build_observations([],meta,intervals)
    assert observations[observations.side=='near'].player.tolist()==['Leon','Leon','Gary']
    assert observations.youtube_time_s.tolist()==[.2,.2,.3,.3,2.,2.]
    assert not observations.observed.any()
    mapping.write_text('changed')
    with pytest.raises(ValueError,match='map changed'):attach_timeline(meta,config)


def test_context_after_end_change_uses_next_athlete_assignment():
    rallies=[{'start_s':1.,'end_s':5.},{'start_s':7.,'end_s':11.}]
    assert rally_slot(5.5,rallies)==0
    assert rally_slot(6.5,rallies)==1


def test_coaching_links_use_original_match_clock():
    from shuttlecoach.coaching import coaching_payload
    rows=[];rallies=[]
    for i in range(3):
        rallies.append({'start_s':i*20.,'end_s':i*20.+20.,'original_start_s':500.+100*i,'original_end_s':520.+100*i})
        rows.append({'rally_id':i+1,'player':'Gary','side':'far','observed':True,'zone':'rear','sample_duration_s':20.})
    passed={'status':'passed','criteria':{'all':True}}
    payload=coaching_payload(pd.DataFrame(rows),{'rallies':rallies},passed,passed)
    assert [r['start_s'] for r in payload['suggestions'][0]['evidence']]==[500.,600.,700.]
