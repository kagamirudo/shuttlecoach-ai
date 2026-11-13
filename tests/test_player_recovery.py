import numpy as np

from shuttlecoach.player_recovery import shirt_histogram, select_candidate, appearance_distance
from shuttlecoach.player_recovery import identity_anchors
from shuttlecoach.analysis import build_observations, summarize


def candidate(x,shirt):
    points=np.ones((17,3))
    return {'img_x':x,'img_y':200.,'bbox':[x,150.,60.,100.],
            'shirt':shirt,'pose_conf':.9,'keypoints':points.ravel().tolist()}


def test_empty_rally_uses_same_athlete_same_half_without_cross_cut_interpolation():
    rallies=[{'start_s':0,'end_s':2,'near_player':'A','far_player':'B'},
             {'start_s':3,'end_s':5,'near_player':'A','far_player':'B'},
             {'start_s':6,'end_s':8,'near_player':'B','far_player':'A'}]
    anchors={(i,s):[] for i in range(3) for s in ['near','far']}
    anchors[(0,'near')]=[0,1];anchors[(2,'near')]=[6,7]
    assert identity_anchors(4,'near',anchors,rallies,list(range(9))).tolist()==[1]
    assert identity_anchors(4,'far',anchors,rallies,list(range(9))).tolist()==[]


def test_recovery_prefers_real_target_and_rejects_wrong_shirt_or_teleport():
    black=np.zeros((300,300,3),np.uint8);white=np.full_like(black,255)
    target=shirt_histogram(black,[150,150,60,100]);other=shirt_histogram(white,[150,150,60,100])
    wrong=candidate(149.,other);right=candidate(160.,target)
    assert select_candidate([wrong,right],np.array([150.,200.]),100.,[target],80.) is right
    assert select_candidate([wrong,candidate(400.,target)],np.array([150.,200.]),100.,[target],80.) is None


def test_recovery_cannot_reuse_a_trusted_pose_even_when_new_prediction_is_a_copy():
    shirt=np.ones(10,dtype=np.float32)/10
    trusted=candidate(150.,shirt);duplicate=candidate(151.,shirt);far=candidate(180.,shirt)
    assert select_candidate([duplicate,far],np.array([150.,200.]),100.,[shirt],80.,[trusted]) is far
    assert select_candidate([duplicate],np.array([150.,200.]),100.,[shirt],80.,[trusted]) is None


def test_shorts_disambiguate_identical_shirts_without_rejecting_bent_torso():
    plain=np.zeros((300,300,3),np.uint8);blue=plain.copy();blue[155:181]=[255,0,0]
    box=[150,150,60,100]
    assert appearance_distance(shirt_histogram(plain,box,torso_only=True),[shirt_histogram(blue,box,torso_only=True)])==0
    assert appearance_distance(shirt_histogram(plain,box),[shirt_histogram(blue,box)])>.65


def test_confirmed_offcourt_tracking_never_enters_occupancy():
    row={'frame_num':0,'player_id':'near','court_x':-.2,'court_y':3.,'img_x':150.,'img_y':200.,
         'bbox':[150,150,60,100],'pose_conf':.9,'keypoints':np.ones((17,3)).ravel().tolist()}
    meta={'times':[0.,.1],'duration_s':.2};rallies=[{'start_s':0.,'end_s':.2,'near_player':'A','far_player':'B'}]
    baseline=build_observations([row],meta,rallies)
    assert not baseline.tracking_observed.any()
    row['identity_verified']=True
    repaired=build_observations([row],meta,rallies)
    assert repaired.tracking_observed.sum()==1 and not repaired.observed.any()
    assert not repaired.zone.notna().any()
    assert all(s['observed_time_s']==0 for s in summarize(repaired))


def test_far_crop_candidates_are_mapped_back_to_source_pixels_without_mutation():
    from types import SimpleNamespace
    from shuttlecoach.player_recovery import pose_candidates,recovery_crop
    class Tensor:
        def __init__(self,array):self.array=array
        def cpu(self):return self
        def numpy(self):return self.array
    boxes=np.array([[50.,40.,20.,40.]])
    xy=np.full((1,17,2),[50.,60.]);confidence=np.ones((1,17))
    prediction=SimpleNamespace(boxes=SimpleNamespace(xywh=Tensor(boxes)),
                               keypoints=SimpleNamespace(xy=Tensor(xy),conf=Tensor(confidence)))
    class Model:
        def predict(self,image,**kwargs):
            assert image.shape==(100,150,3)
            return [prediction]
    image=np.zeros((400,500,3),np.uint8)
    selected=pose_candidates(Model(),image,np.eye(3),12,'cpu',[100,200,250,300])
    assert len(selected)==1 and selected[0]['frame_num']==12
    assert selected[0]['img_x']==150 and selected[0]['img_y']==260
    assert selected[0]['court_x']==150 and selected[0]['court_y']==260
    assert selected[0]['bbox']==[150.,240.,20.,40.]
    assert selected[0]['keypoints'][:2]==[150.,260.]
    np.testing.assert_array_equal(boxes,[[50.,40.,20.,40.]])
    np.testing.assert_array_equal(xy,np.full((1,17,2),[50.,60.]))
    assert recovery_crop(image,[10,20],80)==[0,0,250,60]
