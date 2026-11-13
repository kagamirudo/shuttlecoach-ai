import numpy as np
import pandas as pd
import pytest

from shuttlecoach.analysis import validate_config
from shuttlecoach.calibration import LANDMARKS, fit_calibration
from shuttlecoach.shuttle import court_zone, court_mask, project_shuttle
from shuttlecoach.upstream import court
from shuttlecoach.validation import evaluate


@pytest.fixture
def cropped_court():
    # Both near corners are below the image; four service-line intersections are visible.
    corners = np.array([[-200.,850.],[1480.,850.],[1020.,150.],[260.,150.]])
    h = court.compute_homography(corners)
    names = [f'{row} × {side} singles sideline' for row in ['Near short service','Far short service'] for side in ['left','right']]
    names += ['Far baseline × centre line','Far doubles long service × centre line']
    xy = court.image_to_court(np.array([LANDMARKS[n] for n in names]), np.linalg.inv(h))
    calibration = {'mode':'intersections','frame':0,'reviewed':True,
                   'points':[{'landmark':n,'image_x':float(x),'image_y':float(y)} for n,(x,y) in zip(names,xy)]}
    meta = {'width':1280,'height':720,'frame_count':2,'times':[0.,.1],'duration_s':.2}
    return calibration,meta,h,corners


def test_visible_intersections_infer_offscreen_corners_and_feed_tracking(cropped_court):
    calibration,meta,h,corners = cropped_court
    cfg = {'singles_confirmed':True,'camera_stable_confirmed':True,'calibration':calibration,
           'rallies':[{'start_s':0.,'end_s':.2,'near_player':'A','far_player':'B'}]}
    # A stale stored corner estimate cannot override the actual landmark fit.
    calibration['corners_px'] = [[0,0]]*4
    fitted = validate_config(cfg,meta)
    np.testing.assert_allclose(calibration['corners_px'],corners,atol=.002)
    np.testing.assert_allclose(court.image_to_court(np.array([[300,400],[800,300]]),fitted),
                               court.image_to_court(np.array([[300,400],[800,300]]),h),atol=.0001)
    parent = {'config':cfg,'video':meta,'homography':fitted.tolist()}
    assert len(court_zone(parent)['corners_px'])==4
    assert court_mask(parent).any()
    xy=court.image_to_court(np.array([[3.,6.]]),np.linalg.inv(fitted))[0]
    table=project_shuttle(pd.DataFrame({'detected':[True],'image_x':[xy[0]],'image_y':[xy[1]]}),parent)
    assert table.projection_available.iloc[0]
    np.testing.assert_allclose(table[['projection_x_m','projection_y_m']],[[3.,6.]],atol=.0001)


def test_four_well_spread_matches_are_enough(cropped_court):
    calibration,meta,_,corners=cropped_court
    calibration['points']=calibration['points'][:4]
    _,actual=fit_calibration(calibration,meta)
    np.testing.assert_allclose(actual,corners,atol=.002)


@pytest.mark.parametrize('problem',['few','duplicate_id','duplicate_pixel','outside','nan','collinear','pixel_collinear','wrong_name','mismatched'])
def test_invalid_landmark_fit_fails_closed(cropped_court,problem):
    calibration,meta,_,_=cropped_court
    points=calibration['points']
    if problem=='few':calibration['points']=points[:3]
    if problem=='duplicate_id':points[-1]['landmark']=points[0]['landmark']
    if problem=='duplicate_pixel':points[-1].update(image_x=points[0]['image_x'],image_y=points[0]['image_y'])
    if problem=='outside':points[0]['image_y']=800.
    if problem=='nan':points[0]['image_x']=float('nan')
    if problem=='collinear':
        calibration['points']=[{'landmark':n,'image_x':100.+i*100,'image_y':400.} for i,n in enumerate(list(LANDMARKS)[:5])]
    if problem=='pixel_collinear':
        for i,p in enumerate(points):p.update(image_x=100.+i*100,image_y=400.)
    if problem=='wrong_name':points[0]['landmark']='net post'
    if problem=='mismatched':points[-1]['image_x']+=150
    with pytest.raises(ValueError):fit_calibration(calibration,meta)


def test_fitted_intersections_cannot_validate_themselves(cropped_court):
    calibration,_,h,_=cropped_court
    template=pd.DataFrame([{'source_frame':0,'side':'near'}])
    obs=template.assign(observed=True,image_x=500.,image_y=400.,court_x_m=3.,court_y_m=3.)
    labels=template.assign(visible=True,identity_correct=True,image_x=500.,image_y=400.)
    checks=[{'image_x':p['image_x'],'image_y':p['image_y'],
             'court_x_m':LANDMARKS[p['landmark']][0],'court_y_m':LANDMARKS[p['landmark']][1]} for p in calibration['points'][:4]]
    with pytest.raises(ValueError,match='other than the fitted'):
        evaluate(obs,labels,checks,h,template,calibration)
    # Different visible intersections remain eligible as independent checks.
    names=['Near short service × left doubles sideline','Near short service × right doubles sideline',
           'Far short service × left doubles sideline','Far short service × right doubles sideline']
    target=np.array([LANDMARKS[n] for n in names]);pixels=court.image_to_court(target,np.linalg.inv(h))
    checks=[dict(image_x=float(p[0]),image_y=float(p[1]),court_x_m=float(q[0]),court_y_m=float(q[1])) for p,q in zip(pixels,target)]
    result=evaluate(obs,labels,checks,h,template,calibration)
    assert result['criteria']['four_unused_calibration_points_le_0_25m']
