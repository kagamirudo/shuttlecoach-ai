"""Exercise Make orchestration without downloads or model inference."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def runner(tmp_path):
    record = tmp_path / 'commands.jsonl'
    stub = tmp_path / 'python'
    stub.write_text(f'#!{sys.executable}\n' + '''import json, os, sys
from pathlib import Path
with open(os.environ['RECORD'], 'a') as output:
    output.write(json.dumps({'tool':Path(sys.argv[0]).name, 'args':sys.argv[1:], 'url':os.environ.get('SHUTTLECOACH_URL')})+'\\n')
''')
    stub.chmod(0o755)
    shutil.copy2(stub, tmp_path / 'uv')
    env = os.environ | {'RECORD':str(record), 'PATH':str(tmp_path)+os.pathsep+os.environ['PATH']}
    # Ignore any unrelated caller settings when testing required arguments.
    for name in ['CONFIG','RUN','VIDEO','URL','REVIEW','OUT','PLAYER_REFS','SHUTTLE_REFS','CHECKS','HOST','PORT','DEVICE']:
        env.pop(name, None)
    def run(*args, cwd=ROOT):
        result = subprocess.run(['make','--no-print-directory','-f',str(ROOT/'Makefile'),f'PYTHON={stub}',*args],
                                cwd=cwd, env=env, capture_output=True, text=True, timeout=15)
        calls = [json.loads(line) for line in record.read_text().splitlines()] if record.exists() else []
        return result,calls
    return run


def test_default_help_needs_no_environment(runner):
    result,calls=runner('PYTHON=/missing/project/python')
    assert result.returncode==0 and not calls
    assert 'setup' in result.stdout and 'validate' in result.stdout and 'make run PORT=8502' in result.stdout


@pytest.mark.parametrize('target,missing',[
    ('analyze','CONFIG'),('ingest','URL'),('probe','VIDEO'),('prepare-play','REVIEW'),
    ('recover-players','RUN'),('footwork','RUN'),('llm-report','RUN'),('motion-preview','RUN'),('shuttle','RUN'),('associate-shuttle','RUN'),
    ('validate-players','RUN'),('validate-shuttle','RUN'),('report','RUN'),('test-ui','VIDEO')])
def test_missing_inputs_stop_before_running_tools(runner,target,missing):
    result,calls=runner(target)
    assert result.returncode!=0 and f'Missing {missing}' in result.stdout+result.stderr
    assert not calls


@pytest.mark.parametrize('target,variables,expected',[
    ('run',['PORT=18501'],['-m','streamlit','run','app.py','--server.headless','true','--server.address','127.0.0.1','--server.port','18501','--browser.gatherUsageStats','false']),
    ('test',[],['-m','pytest','-q']),
    ('ingest',['URL=https://www.youtube.com/watch?v=example&list=abc'],['-m','shuttlecoach.cli','ingest','https://www.youtube.com/watch?v=example&list=abc']),
    ('probe',['VIDEO=clips/a b.mp4'],['-m','shuttlecoach.cli','probe','clips/a b.mp4']),
    ('prepare-play',['REVIEW=reviews/a b.json','OUT=data/selected play'],['scripts/prepare_active_play.py','reviews/a b.json','--out','data/selected play']),
    ('analyze',['CONFIG=clips/a "quoted" & b.json','DEVICE=mps'],['-m','shuttlecoach.cli','analyze','clips/a "quoted" & b.json','--device','mps']),
    ('recover-players',['RUN=results/my run'],['-m','shuttlecoach.cli','recover-players','results/my run','--device','auto']),
    ('footwork',['RUN=results/my run'],['-m','shuttlecoach.cli','footwork','results/my run']),
    ('llm-report',['RUN=results/my run'],['-m','shuttlecoach.cli','llm-report','results/my run']),
    ('motion-preview',['RUN=results/my run'],['-m','shuttlecoach.cli','motion-preview','results/my run']),
    ('shuttle',['RUN=results/my run'],['-m','shuttlecoach.cli','shuttle','results/my run','--new-court-review','--quality','--candidates','--device','auto']),
    ('associate-shuttle',['RUN=results/my run','REVIEW=reviews/activity.json'],['-m','shuttlecoach.cli','associate-shuttle','results/my run','--review','reviews/activity.json']),
    ('validate-players',['RUN=results/a','PLAYER_REFS=players.csv','CHECKS=checks.json'],['-m','shuttlecoach.cli','validate','results/a','--references','players.csv','--checks','checks.json']),
    ('validate-shuttle',['RUN=results/a','SHUTTLE_REFS=birdie.csv'],['-m','shuttlecoach.cli','validate-shuttle','results/a','--references','birdie.csv']),
    ('report',['RUN=results/a'],['-m','shuttlecoach.cli','report','results/a']),
    ('test-ui',['VIDEO=clips/a b.mp4','RUN=results/my run/','PORT=18501'],['scripts/browser_check.py','clips/a b.mp4','my run']),
    ('test-calibration-ui',['PORT=18501'],['scripts/calibration_browser_check.py']),
])
def test_commands_forward_quoted_arguments(runner,target,variables,expected):
    result,calls=runner(target,*variables)
    assert result.returncode==0,result.stdout+result.stderr
    assert len(calls)==1 and calls[0]['args']==expected
    if target in ['test-ui','test-calibration-ui']:
        assert calls[0]['url']=='http://127.0.0.1:18501'


def test_setup_syncs_dependencies_before_fetching_models_even_with_parallel_make(runner):
    result,calls=runner('-j4','setup')
    assert result.returncode==0,result.stdout+result.stderr
    assert [c['tool'] for c in calls]==['uv','python','python']
    assert calls[0]['args'][:2]==['pip','sync']
    assert [c['args'] for c in calls[1:]]==[['scripts/download_model.py'],['scripts/download_tracknet_model.py']]


def test_combined_validation_checks_all_inputs_before_mutating_a_run(runner):
    result,calls=runner('validate','RUN=results/a','PLAYER_REFS=players.csv','CHECKS=checks.json')
    assert result.returncode!=0 and 'Missing SHUTTLE_REFS' in result.stdout+result.stderr
    assert not calls


def test_combined_validation_is_sequential_even_with_parallel_make(runner):
    result,calls=runner('-j4','validate','RUN=results/a','PLAYER_REFS=players.csv','SHUTTLE_REFS=birdie.csv','CHECKS=checks.json')
    assert result.returncode==0,result.stdout+result.stderr
    assert [c['args'][2] for c in calls]==['validate','validate-shuttle']


def test_clean_preserves_media_models_results_and_provenance(runner,tmp_path):
    for directory in ['data','models','results','.cache','.pytest_cache','shuttlecoach/__pycache__']:
        path=tmp_path/directory;path.mkdir(parents=True,exist_ok=True);(path/'keep').write_text('fixture')
    result,_=runner('clean',cwd=tmp_path)
    assert result.returncode==0,result.stderr
    assert all((tmp_path/d/'keep').is_file() for d in ['data','models','results','.cache'])
    assert not (tmp_path/'.pytest_cache').exists() and not (tmp_path/'shuttlecoach/__pycache__').exists()
