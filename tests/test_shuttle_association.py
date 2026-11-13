import numpy as np
import json
import pandas as pd
import pytest

from shuttlecoach.shuttle_association import white_components, flight_path, validate_review, associate_candidates, bidirectional_flight, merge_activity_intervals, on_player


def test_foreground_requires_observed_white_pixels_and_background_change():
    background=np.full((100,100,3),[50,120,60],np.uint8);image=background.copy()
    image[49:52,49:52]=220
    floor=np.array([[0,0],[99,0],[99,99],[0,99]],np.float32)
    found=white_components(image,background,floor,[(np.array([50.,45.]),100.)])
    assert len(found)==1 and found[0]['method']=='held_foreground'
    assert found[0]['image_x']==50 and found[0]['image_y']==50
    assert not white_components(image,image,floor,[(np.array([50.,45.]),100.)])


def test_airborne_white_support_excludes_large_racket_blur():
    background=np.full((100,100,3),[50,80,120],np.uint8);image=background.copy()
    image[20:25,20:25]=220;image[20:25,50:80]=220
    floor=np.array([[0,60],[99,60],[99,99],[0,99]],np.float32)
    found=white_components(image,background,floor,[])
    assert len(found)==1 and found[0]['method']=='air_foreground'
    assert found[0]['image_x']==22 and found[0]['image_y']==22


def test_flight_association_uses_target_contact_and_breaks_teleports_without_filling():
    times=np.arange(4)/30
    options=[[{'image_x':x,'image_y':100.,'confidence':.95}] for x in [100.,110.,120.,800.]]
    wrists=[[(np.array([100.,100.]),100.)]]*4
    path=flight_path(options,wrists,times)
    assert path[:3]==[0,0,0] and path[3] is None
    assert flight_path([[],[]],wrists[:2],times[:2])==[None,None]


def test_activity_review_rejects_wrong_source_and_overlaps():
    manifest={'video':{'sha256':'ok','times':[0.,1.]},'start_frame':0,'config':{'rallies':[{'end_s':1.}]}}
    review={'source_sha256':'bad','reviewer':'source review','reviewed_rallies':[{'start_s':0.,'end_s':1.}], 'target_flight_intervals':[[0.,1.]]}
    with pytest.raises(ValueError,match='different source'):validate_review(review,manifest)
    review['source_sha256']='ok';review['target_flight_intervals']=[[0.,.8],[.5,1.]]
    with pytest.raises(ValueError,match='non-overlapping'):validate_review(review,manifest)
    review['reviewed_rallies']=[{'start_s':-4e-7,'end_s':1.,'near_player':'A','far_player':'B'}]
    review['target_flight_intervals']=[[-4e-7,1.]]
    validate_review(review,manifest)
    review['target_flight_intervals']=[[-.001,1.]]
    with pytest.raises(ValueError,match='tracked span'):validate_review(review,manifest)


def test_backward_contact_recovers_entering_high_shot_without_inventing_missing_points():
    options=[[{'image_x':300.,'image_y':y,'confidence':.9,'area':10}] for y in [50.,80.,110.,140.,170.,200.]]
    wrists=[[(np.array([300.,200.]),40.)]]*6
    options[2]=[]
    path=bidirectional_flight(options,wrists,np.arange(6)/30)
    assert path[0]==0 and path[2] is None and path[-1]==0


def test_foreground_recovery_requires_real_pixels_and_recent_network_anchor():
    manifest={'tracking_stride':1,'homography':np.eye(3).tolist(),'video':{'cut_frames':[]},
              'config':{'calibration':{'corners_px':[[100,400],[500,400],[450,100],[150,100]]},
                        'shuttle_candidates':True,'shuttle_association':True}}
    table=pd.DataFrame({'source_frame':range(72),'source_time_s':np.arange(72)/30,
        'candidates':[json.dumps([{'image_x':300.,'image_y':200.,'confidence':.9,'area':10}])]+['[]']*71})
    points=np.zeros((17,3));points[:,2]=.9
    raw=[{'frame_num':0,'court_x':3.,'court_y':3.,'img_x':10.,'img_y':10.,'pose_conf':.9,
          'bbox':[10,10,10,10],'keypoints':points.flatten().tolist()}]
    review={'reviewed_rallies':[{'start_s':0.,'end_s':2.4}],'target_flight_intervals':[[0.,2.4]]}
    real={'image_x':300.,'image_y':200.,'method':'floor_foreground','area':4}
    found=associate_candidates(table,[[real]]*72,raw,manifest,review)
    assert found[0] is not None and found[60] is not None and all(p is None for p in found[61:])
    assert all(p is None for p in associate_candidates(table,[[]]*72,raw,manifest,review)[1:])
    manifest['video']['cut_frames']=[3]
    assert all(p is None for p in associate_candidates(table,[[real]]*72,raw,manifest,review)[3:])
    manifest['video']['cut_frames']=[]
    raw[0]['bbox']=[300,200,60,100]
    assert all(p is None for p in associate_candidates(table,[[real]]*72,raw,manifest,review)[1:])


def test_pose_mask_keeps_empty_box_space_and_rejects_white_body_pixels():
    joints=np.zeros((17,3));joints[:,2]=.9
    joints[[5,6,12,11],:2]=[[40,30],[60,30],[60,65],[40,65]]
    joints[[7,9,8,10,13,15,14,16],:2]=[[30,45],[20,55],[70,45],[80,55],[40,80],[35,95],[60,80],[65,95]]
    joints[0,:2]=[50,18]
    row={'bbox':[50,55,90,90],'keypoints':joints.flatten().tolist()}
    assert on_player([50,50],row) and on_player([30,45],row)
    assert not on_player([75,80],row)


def test_activity_editor_merges_existing_transfer_windows_with_edited_play():
    assert merge_activity_intervals([[2.,5.],[1.,3.],[7.,8.]])==[[1.,5.],[7.,8.]]
    with pytest.raises(ValueError):merge_activity_intervals([[1.,float('nan')]])
