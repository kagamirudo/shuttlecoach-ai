from streamlit.testing.v1 import AppTest
from pathlib import Path
import pytest


@pytest.fixture(autouse=True)
def isolated_credential_store(monkeypatch):
    from shuttlecoach import credentials
    monkeypatch.setattr(credentials,'load_key',lambda:None)
    monkeypatch.setattr(credentials,'can_save_key',lambda:True)
    def forbidden(*args):raise AssertionError('AppTest cannot access the real Keychain')
    monkeypatch.setattr(credentials,'_keychain',forbidden)


def test_dashboard_starts_without_video_or_results(tmp_path,monkeypatch):
    import shuttlecoach.paths as paths
    monkeypatch.setattr(paths,'RESULTS',tmp_path)
    app=AppTest.from_file(Path(__file__).resolve().parents[1]/"app.py",default_timeout=30).run()
    assert not app.exception
    assert app.title[0].value=="ShuttleCoach"
    assert app.radio[0].value=="Local file"
    assert [t.label for t in app.tabs]==['Prepare a clip','1 · Footage','2 · Players & time','3 · Court & tracking','Review & validate']
    assert not next(e for e in app.expander if e.label=='Use a file path instead').proto.expanded
    app.radio[0].set_value("YouTube").run()
    assert not app.exception
    assert app.button[0].disabled


def test_review_explains_coverage_and_zones_without_exposing_raw_measurements(tmp_path,monkeypatch):
    import json
    import numpy as np
    import pandas as pd
    from PIL import Image
    import shuttlecoach.media as media
    import shuttlecoach.paths as paths

    results=tmp_path/'results';run=results/'review-fixture';run.mkdir(parents=True)
    source=tmp_path/'clip.mp4';source.write_bytes(b'fixture')
    monkeypatch.setattr(paths,'RESULTS',results)
    monkeypatch.setattr(media,'read_frame',lambda *args:np.zeros((240,320,3),np.uint8))
    manifest={'video':{'path':str(source),'times':[0.,1.],'pts_origin_s':0.,'fps':1.,'duration_s':2.},
              'config':{'rallies':[{'start_s':0.,'end_s':2.,'near_player':'Alice','far_player':'Bob'}]},
              'overlay_source_start_s':0.,'end_frame_exclusive':2,'summaries':[]}
    (run/'analysis.json').write_text(json.dumps(manifest))
    (run/'validation.json').write_text(json.dumps({'status':'failed','criteria':{'both_players_correct_95pct':False},
        'both_players_correct_rate':.83,'median_projected_error_m':.30,'sample_frames':100,
        'visible_references':186,'calibration_check_errors_m':[.09,.08,.03,.03]}))
    (run/'report.md').write_text('Test report')
    (run/'overlay.mp4').write_bytes(b'fixture')
    Image.new('RGB',(320,240)).save(run/'heatmap.png')
    pd.DataFrame([{'source_frame':0,'side':side,'visible':None,'image_x':None,'image_y':None,'identity_correct':None} for side in ['near','far']]).to_csv(run/'reference_template.csv',index=False)
    pd.DataFrame([{'player':player,'side':side,'sample_duration_s':1.,'observed':visible,'tracking_observed':visible,'zone':'front' if visible else None}
                  for player,side,visible in [('Alice','near',True),('Alice','near',True),('Bob','far',True),('Bob','far',False)]]).to_csv(run/'observations.csv',index=False)
    app=AppTest.from_file(Path(__file__).resolve().parents[1]/'app.py',default_timeout=30).run()
    assert not app.exception
    tabs={t.label:t for t in app.tabs}
    assert [m.value for m in app.metric if m.label=='Player tracking']==['Needs fixes']
    assert [m.value for m in app.metric if m.label=='Birdie tracking']==['Not reviewed']
    assert any(h.value=='Your tracking results' for h in tabs['Overview'].subheader)
    assert not any(h.value=='AI analysis' for h in tabs['Overview'].subheader)
    assert any(h.value=='AI analysis' for h in tabs['AI & practice'].subheader)
    assert any(h.value=='Annotate the frozen validation sample' for h in tabs['Validation'].subheader)
    assert [m.value for m in app.metric if m.label=='Tracking coverage']==['100.0%','50.0%']
    assert next(m.value for m in app.metric if m.label=='Reviewed player pairs usable')=='83.0%'
    assert any('not passed validation' in w.value for w in app.warning)
    text=' '.join(m.value for m in app.markdown)
    assert 'Tracked for **1.0 of 2.0 seconds**' in text
    assert 'Front · near the net: **100.0%**' in text
    assert any('does not measure' in m.value for m in app.markdown)
    details=next(e for e in app.expander if e.label=='Detailed measurements and CSV')
    assert not details.proto.expanded
    assert len(details.dataframe)==1
    assert not next(e for e in app.expander if e.label=='Detailed validation measurements').proto.expanded

    # Practice cards must appear only for a current report whose gates passed.
    import shuttlecoach.coaching as coaching
    monkeypatch.setattr(coaching,'evidence_signature',lambda path:'current')
    report={'gate':'passed','evidence_signature':'current','blocked_reasons':[],
            'suggestions':[{'player':'Alice','title':'Reviewed lunge recovery','pattern':'Three reviewed rallies.',
                            'suggestion':'Controlled feed and recovery practice.','evidence':[{'start_s':0.,'end_s':1.}]}]}
    (run/'coaching_report.md').write_text('Reviewed report')
    for gate,signature,visible in [('passed','current',True),('blocked','current',False),('passed','stale',False)]:
        report.update(gate=gate,evidence_signature=signature)
        (run/'coaching.json').write_text(json.dumps(report))
        app.run()
        assert not app.exception
        assert any(h.value=='Alice · Reviewed lunge recovery' for h in app.subheader)==visible

    from shuttlecoach import credentials
    saved=[]
    monkeypatch.setattr(credentials,'save_key',saved.append)
    next(t for t in app.text_input if t.label=='Groq API key').set_value('fixture-key').run()
    next(b for b in app.button if b.label=='Save API key on this Mac').click().run()
    assert saved==['fixture-key'] and not app.exception
    monkeypatch.setattr(credentials,'load_key',lambda:'fixture-key')
    app=AppTest.from_file(Path(__file__).resolve().parents[1]/'app.py',default_timeout=30).run()
    assert not app.exception
    assert next(t for t in app.text_input if t.label=='Groq API key').value==''
    assert not next(b for b in app.button if b.label=='Analyze tracking and footwork').disabled
    assert any('saved in macOS Keychain' in s.value for s in app.success)

    # Findings are visible in the main page, not inside connection settings or a download.
    from shuttlecoach.llm_report import REPORT_VERSION
    draft={'report_version':REPORT_VERSION,'evidence_signature':'current','mode':'descriptive_analysis',
           'summary':'Measured positioning changes are available for review.', 'limitations':'Measured data, not unseen video.',
           'player_labels':{'P1':'Alice'},'facts':[{'id':'F1','player_id':'P1','kind':'selected_window_overview',
               'start_s':0.,'end_s':2.,'metrics':{'tracking_coverage':'100.0%','tracked_time':'2.0 s','missing_time':'0.0 s','front':'100.0%','middle':'0.0%','rear':'0.0%'}}],
           'analyses':[{'player_id':'P1','observations':[{'topic':'positioning','finding':'Accepted positions were concentrated near the net.','meaning':'Only accepted positions contribute to this distribution.','review_focus':'Check whether pauses contribute to this distribution.','evidence_ids':['F1']}],
                       'limitations':'This does not establish a technique weakness.'}],'sections':[]}
    (run/'llm_report.json').write_text(json.dumps(draft))
    app.run()
    assert not app.exception
    assert any(m.value=='Accepted positions were concentrated near the net.' for m in app.markdown)
    assert any(m.value=='Only accepted positions contribute to this distribution.' for m in app.markdown)
    assert any(m.value=='Check whether pauses contribute to this distribution.' for m in app.markdown)
    assert any('Time comparison unavailable' in c.value for c in app.caption)
    settings=next(e for e in app.expander if e.label=='AI connection settings')
    assert not any(m.value=='Accepted positions were concentrated near the net.' for m in settings.markdown)
    assert not any('Download AI' in b.label for b in app.button)
    monkeypatch.setattr(credentials,'can_save_key',lambda:False)
    app.run()
    assert not app.exception
    assert not any(b.label=='Save API key on this Mac' for b in app.button)
    assert not any(c.label=='Remember API key on this Mac' for c in app.checkbox)
    assert any('Docker runtime secret' in s.value for s in app.success)
    # A comparison can open either cited source moment, rather than only the first.
    for fact_id,start,end in [('F2',0.,30.),('F3',60.,90.)]:
        draft['facts'].append({'id':fact_id,'player_id':'P1','kind':'positioning_time_window',
            'start_s':start,'end_s':end,'metrics':{**draft['facts'][0]['metrics'],'in_court_time':'30.0 s'}})
    draft['analyses'][0]['observations'].append({'topic':'change',
        'finding':'Two measured windows are available.','meaning':'Compare accepted court time only.',
        'review_focus':'Check both windows for pauses.','evidence_ids':['F2','F3']})
    (run/'llm_report.json').write_text(json.dumps(draft))
    app.run()
    next(s for s in app.selectbox if s.label=='Source window').set_value(1).run()
    assert not app.exception
    assert any(v.proto.start_time==59 for v in app.get('video'))
    draft['report_version']=REPORT_VERSION-1
    (run/'llm_report.json').write_text(json.dumps(draft))
    app.run()
    assert not app.exception
    assert any('stricter report format' in i.value for i in app.info)
    assert not any(m.value=='Accepted positions were concentrated near the net.' for m in app.markdown)


def test_changed_duration_and_names_can_start_a_separate_tracking_job(tmp_path,monkeypatch):
    import json
    import numpy as np
    import subprocess
    import shuttlecoach.media as media
    import shuttlecoach.paths as paths

    source=tmp_path/'clip.mp4';source.write_bytes(b'fixture')
    results=tmp_path/'results';results.mkdir()
    (tmp_path/'.cache').mkdir()
    monkeypatch.setattr(paths,'ROOT',tmp_path)
    monkeypatch.setattr(paths,'RESULTS',results)
    meta={'path':str(source),'sha256':'rerun-test','width':640,'height':480,
          'duration_s':375.8,'fps':10.,'frame_count':3758,
          'times':(np.arange(3758)/10).tolist(),'pts_origin_s':0.}
    monkeypatch.setattr(media,'probe',lambda path:meta)
    monkeypatch.setattr(media,'read_frame',lambda *args:np.zeros((480,640,3),np.uint8))
    processes=[];configs=[]
    class Process:
        returncode=None
        def poll(self):return self.returncode
    def launch(command,**kwargs):
        configs.append(Path(command[command.index('analyze')+1]))
        process=Process();processes.append(process)
        return process
    monkeypatch.setattr(subprocess,'Popen',launch)
    app=AppTest.from_file(Path(__file__).resolve().parents[1]/'app.py',default_timeout=30)
    app.session_state['video_path']=str(source)
    app.session_state['source_hash']=meta['sha256']
    app.session_state['calibration_context']=(meta['sha256'],0,'Four outer corners')
    app.session_state['corners']=[[80,450],[560,450],[490,40],[150,40]]
    app.session_state['landmark_points']=[]
    app.session_state['calibration_review']=True
    app.run()
    for label in ['Exactly two singles players are visible','The selected camera view stays fixed without cuts']:
        next(c for c in app.checkbox if c.label==label).check()
    app.run()
    button=lambda:next(b for b in app.button if b.label=='Run player tracking')
    assert not app.exception and not button().disabled
    button().click().run()
    assert not app.exception and button().disabled and len(configs)==1
    old_text=configs[0].read_text()
    assert json.loads(old_text)['rallies'][0]['end_s']==30.

    # Simulate the worker finishing, then the user's duration/name edits.
    processes[0].returncode=0
    app.session_state['rallies_rerun-test']={'edited_rows':{0:{'end_s':375.,'near_player':'Shi Yu Qi','far_player':'Random China'}},'added_rows':[],'deleted_rows':[]}
    app.run()
    assert not app.exception and not button().disabled
    assert app.session_state['process_completion_refreshed']
    assert app.session_state['calibration_review']
    button().click().run()
    assert not app.exception and button().disabled and len(configs)==2
    assert configs[0]!=configs[1] and configs[0].read_text()==old_text
    new=json.loads(configs[1].read_text())
    assert new['rallies']==[{'start_s':0.,'end_s':375.,'near_player':'Shi Yu Qi','far_player':'Random China'}]
    assert new['calibration']['reviewed']

    # Opening a completed worker selects its result and takes the user to review.
    processes[1].returncode=0
    Path(app.session_state['log']).write_text('RESULT=/saved/completed-run\n')
    app.run()
    next(b for b in app.button if b.label=='Open completed analysis').click().run()
    assert not app.exception
    assert app.session_state['selected_run']=='/saved/completed-run'
    assert app.session_state['workspace']=='Review & validate'
    assert 'process' not in app.session_state
