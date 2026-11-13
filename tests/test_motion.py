import numpy as np
import pandas as pd
import pytest

from shuttlecoach.motion import motion_estimates,masked_diagnostic


def tracks():
    return pd.DataFrame([{'rally_id':1,'player':'A','side':'near','source_frame':i,'source_time_s':i/10,
                         'original_source_time_s':100+i/10,'observed':i<7 or i>=12,'tracking_observed':i<7 or i>=12,
                         'court_x_m':1+i/10 if i<7 or i>=12 else np.nan,'court_y_m':3.} for i in range(20)])


def test_forecast_is_past_only_bounded_and_never_mutates_observations():
    table=tracks();before=table.copy(deep=True)
    result=motion_estimates(table,{})
    assert result.source_frame.tolist()==[7,8,9]
    assert result.court_x_m.tolist()==pytest.approx([1.7,1.8,1.9])
    assert result.original_source_time_s.tolist()==[100.7,100.8,100.9]
    assert result.position_source.eq('estimated_past_only_regression').all()
    future=table.copy();future.loc[future.source_frame>=12,'court_x_m']=7.
    pd.testing.assert_frame_equal(motion_estimates(future,{}),result)
    pd.testing.assert_frame_equal(table,before)


def test_cuts_end_changes_and_insufficient_history_stop_estimation():
    table=tracks()
    assert motion_estimates(table,{'cut_frames':[7]}).empty
    table.loc[table.source_frame>=7,'rally_id']=2
    table.loc[table.source_frame>=7,'player']='B'
    assert motion_estimates(table,{}).empty


@pytest.mark.parametrize('problem',['fast','noisy','sparse','no_history'])
def test_unreliable_regression_is_withheld(problem):
    table=tracks()
    if problem=='fast':table.loc[:6,'court_x_m']=np.arange(7)*1.5
    if problem=='noisy':table.loc[:6,'court_x_m']=[1,3,1,3,1,3,1]
    if problem=='sparse':table=table[table.source_frame.isin([0,3,6,7,8,9])]
    if problem=='no_history':table.loc[:6,['observed','tracking_observed']]=False
    assert motion_estimates(table,{}).empty


def test_masked_diagnostic_reports_tracker_agreement_not_ground_truth():
    table=tracks();table['observed']=table['tracking_observed']=True
    table['court_x_m']=1+table.source_time_s
    result=masked_diagnostic(table,{})
    assert result['masked_positions']==3 and result['predicted_positions']==3
    assert result['median_difference_m']==pytest.approx(0.,abs=1e-10)
    assert 'not manual truth' in result['limitation']
