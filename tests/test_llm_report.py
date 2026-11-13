import copy
import json
from urllib.error import HTTPError

import pytest

from shuttlecoach import llm_report as llm


def evidence(gate='passed'):
    return {'gate':gate,'evidence_signature':'verified','validation':{'players':{'status':'passed','both_players_correct_rate':.97},'shuttle':{'status':'passed'}},
            'suggestions':[{'player':'Gary','pattern':'Rear third in three reviewed rallies.','suggestion':'Controlled rear-corner feed and recovery.','evidence':[{'start_s':20.,'end_s':30.}]}]}


def test_input_is_anonymous_and_failed_gates_never_send_drills():
    data=llm.report_input(evidence());text=json.dumps(data)
    assert 'Gary' not in text and data['supported_items'][0]['player']=='P1'
    failed=llm.report_input(evidence('blocked'))
    assert failed['mode']=='descriptive_analysis' and not failed['supported_items']


@pytest.mark.parametrize('bad',[
    {'summary':'You were late by 3 seconds.','sections':[]},
    {'summary':'See [link](https://example.com).','sections':[]},
    {'summary':'Quality review.','sections':[{'id':1,'explanation':'Invented fault.','practice':'New drill.'}]},
    {'summary':'Quality review.','sections':[],'extra':'injected'},
])
def test_unapproved_sections_numbers_and_links_are_rejected(bad):
    with pytest.raises(ValueError):llm.validate_response(json.dumps(bad),llm.report_input(evidence('blocked')))


def test_http_request_cache_and_key_never_persist(tmp_path,monkeypatch):
    monkeypatch.setattr(llm,'write_coaching_report',lambda run:evidence())
    calls=[]
    output={'analyses':[{'player_id':'P1','observations':[{'finding':'The approved pattern concerns rear-court positioning.','evidence_ids':['F1']}]}],'sections':[{'id':1,'explanation':'Rear-court positioning repeated.','practice':'Use controlled feeds and review recovery.'}]}
    class Response:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def read(self,limit):return json.dumps({'choices':[{'finish_reason':'stop','message':{'content':json.dumps(output)}}],'usage':{'total_tokens':200}}).encode()
    def send(req,timeout):
        calls.append(req)
        assert req.full_url==llm.ENDPOINT and req.get_header('Authorization')=='Bearer secret-test-key'
        assert req.get_header('User-agent')=='ShuttleCoach/0.2'
        assert req.get_header('Accept')=='application/json'
        body=json.loads(req.data)
        assert body['model']==llm.MODEL and 'Gary' not in body['messages'][1]['content']
        return Response()
    monkeypatch.setattr(llm.request,'urlopen',send)
    result=llm.generate_llm_report(tmp_path,'secret-test-key')
    assert result['mode']=='coaching' and len(calls)==1
    assert llm.generate_llm_report(tmp_path)==result and len(calls)==1
    assert all('secret-test-key' not in p.read_text() for p in tmp_path.iterdir())


def test_quota_error_is_not_retried_or_persisted(tmp_path,monkeypatch):
    monkeypatch.setattr(llm,'write_coaching_report',lambda run:evidence('blocked'))
    calls=[]
    def send(req,timeout):
        calls.append(req)
        raise HTTPError(req.full_url,429,'secret-test-key',{},None)
    monkeypatch.setattr(llm.request,'urlopen',send)
    with pytest.raises(ValueError,match='quota reached') as error:llm.generate_llm_report(tmp_path,'secret-test-key')
    assert 'secret-test-key' not in str(error.value) and len(calls)==1 and not list(tmp_path.iterdir())


@pytest.mark.parametrize('body,expected',[
    (b'error code: 1010 cloudflare secret-test-key','network gateway'),
    (b'{"error":{"code":"model_permission_blocked_org","message":"secret-test-key"}}','organization settings'),
    (b'{"error":{"code":"model_permission_blocked_project"}}','project settings'),
    (b'{"error":"secret-test-key"}','does not by itself prove'),
    (b'<html>secret-test-key</html>','does not by itself prove'),
])
def test_forbidden_responses_are_classified_without_exposing_provider_bodies(body,expected):
    import io
    exc=HTTPError(llm.ENDPOINT,403,'Forbidden',{},io.BytesIO(body))
    message=llm.provider_error(exc)
    assert expected in message and 'secret-test-key' not in message


def measured_context():
    import pandas as pd
    observations=pd.DataFrame([
        {'player':p,'side':side,'rally_id':1,'source_time_s':t,'original_source_time_s':1000+t,
         'sample_duration_s':30.,'observed':good,'tracking_observed':good,'zone':zone}
        for p,side,t,good,zone in [('Gary','near',0.,True,'middle'),('Gary','near',30.,True,'rear'),
                                  ('Leon','far',0.,True,'rear'),('Leon','far',30.,False,None)]])
    data=evidence('blocked');data['validation']['players'].update(status='failed',criteria={'no_reviewed_identity_errors':True},visible_references=186)
    data['footwork']={'summaries':[{'player':'Gary','pose_seconds':60.,'pose_coverage':1.,'candidate_counts':{'lunge':4,'stance_opening':6,'left_step':8,'right_step':7}}]}
    return llm.report_input(data,observations)


def analysis_response(facts):
    return {'analyses':[{'player_id':p,'observations':[{'finding':'The measurements describe accepted positioning, not a diagnosed weakness.',
                'evidence_ids':[next(r['id'] for r in facts['facts'] if r['player_id']==p)]}]} for p in facts['players']], 'sections':[]}


def test_failed_gate_still_supplies_measured_position_changes_and_candidate_context():
    facts=measured_context();serialized=json.dumps(facts)
    assert 'Gary' not in serialized and 'Leon' not in serialized
    assert facts['players']==['P1','P2'] and not facts['supported_items']
    assert facts['validation']['players']['criteria']['no_reviewed_identity_errors'] is True
    windows=[r for r in facts['facts'] if r['player_id']=='P1' and r['kind']=='positioning_time_window']
    assert windows[0]['metrics']['middle']=='100.0%' and windows[1]['metrics']['rear']=='100.0%'
    assert windows[0]['start_s']==1000.
    far=next(r for r in facts['facts'] if r['player_id']=='P2' and r['kind']=='selected_window_overview')
    assert far['metrics']['tracking_coverage']=='50.0%' and far['metrics']['missing_time']=='30.0 s'
    assert any(r['kind']=='unconfirmed_footwork_candidates' and r['metrics']['lunge']=='4' for r in facts['facts'])
    response=analysis_response(facts)
    assert len(llm.validate_response(json.dumps(response),facts)['analyses'])==2


@pytest.mark.parametrize('problem',['invented_number','missing_reference','wrong_player','duplicate_player','unapproved_drill','wrong_denominator'])
def test_analysis_requires_measured_numbers_and_player_specific_evidence(problem):
    facts=measured_context();response=analysis_response(facts);finding=response['analyses'][0]['observations'][0]
    if problem=='invented_number':finding['finding']='Tracking coverage is 99.9%.'
    if problem=='missing_reference':finding['evidence_ids']=['unknown']
    if problem=='wrong_player':finding['evidence_ids']=[next(r['id'] for r in facts['facts'] if r['player_id']=='P2')]
    if problem=='duplicate_player':response['analyses'][1]=copy.deepcopy(response['analyses'][0])
    if problem=='unapproved_drill':response['sections']=[{'id':1,'explanation':'Fault.','practice':'Invented drill.'}]
    if problem=='wrong_denominator':finding['finding']='The player spent 100.0% of the match in the rear.'
    with pytest.raises(ValueError):llm.validate_response(json.dumps(response),facts)


def test_cited_measurement_numbers_are_allowed_in_findings():
    facts=measured_context();response=analysis_response(facts)
    response['analyses'][0]['observations'][0]['finding']='Tracking availability is 100.0%, with 0.0 s missing.'
    assert llm.validate_response(json.dumps(response),facts)


def test_candidate_counts_never_become_claims_of_confirmed_activity():
    facts=measured_context();response=analysis_response(facts)
    item=response['analyses'][0]['observations'][0]
    item.update(finding='These counts prove frequent steps and superior movement.',
                evidence_ids=[next(r['id'] for r in facts['facts'] if r['kind']=='unconfirmed_footwork_candidates')])
    checked=llm.validate_response(json.dumps(response),facts)
    finding=checked['analyses'][0]['observations'][0]
    assert finding['basis']=='review_required' and 'do not establish step frequency' in finding['finding']
    assert 'superior' not in finding['finding']
