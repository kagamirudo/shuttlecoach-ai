from pathlib import Path
import yaml

from scripts.docker_compose import compose_call

ROOT=Path(__file__).resolve().parents[1]


def test_container_preserves_media_paths_and_only_publishes_loopback():
    config=yaml.safe_load((ROOT/'compose.yaml').read_text())['services']['app']
    assert config['ports']==['127.0.0.1:${DOCKER_PORT:-8502}:8501']
    assert len(config['volumes'])==4
    assert all('${SHUTTLECOACH_APP_DIR' in value for value in config['volumes'])
    assert config['build']['args']['APP_DIR']=='${SHUTTLECOACH_APP_DIR:-${PWD:-/app}}'
    assert 'GROQ_API_KEY' not in config.get('environment',{})


def test_up_bridges_key_as_runtime_secret_without_putting_it_in_arguments():
    command,env=compose_call(['up','-d'],{},lambda:'dummy-secret')
    assert str(ROOT/'compose.groq.yaml') in command and 'dummy-secret' not in ' '.join(command)
    assert env['GROQ_API_KEY']=='dummy-secret' and env['SHUTTLECOACH_APP_DIR']==str(ROOT)
    secret=yaml.safe_load((ROOT/'compose.groq.yaml').read_text())
    assert secret['secrets']['groq_api_key']['environment']=='GROQ_API_KEY'
    assert secret['services']['app']['environment']['GROQ_API_KEY_FILE']=='/run/secrets/groq_api_key'


def test_build_down_and_tests_do_not_load_saved_credentials():
    def forbidden():raise AssertionError('This operation must not load a key')
    for arguments,env in [(['build'],{}),(['down'],{}),(['run','--rm','app','python','-m','pytest'],{'SHUTTLECOACH_NO_SECRET':'1'})]:
        command,_=compose_call(arguments,env,forbidden)
        assert str(ROOT/'compose.groq.yaml') not in command


def test_missing_or_locked_keychain_does_not_prevent_docker_start():
    def locked():raise ValueError('Keychain locked')
    command,env=compose_call(['up','-d'],{},locked)
    assert str(ROOT/'compose.groq.yaml') not in command and 'GROQ_API_KEY' not in env


def test_build_context_is_explicit_and_excludes_private_runtime_files():
    ignore=(ROOT/'.dockerignore').read_text().splitlines()
    assert ignore[0]=='**'
    assert not any(line in ignore for line in ['!data/**','!models/**','!results/**','!.cache/**','!.env','!.env.*'])
    assert '**/.git' in ignore and '**/secrets.toml' in ignore
