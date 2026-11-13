import pytest
from shuttlecoach import credentials


def test_key_resolution_uses_explicit_environment_then_keychain(monkeypatch):
    monkeypatch.setattr(credentials,'load_key',lambda:'remembered-key')
    monkeypatch.delenv('GROQ_API_KEY',raising=False)
    assert credentials.resolve_key()=='remembered-key'
    monkeypatch.setenv('GROQ_API_KEY','environment-key')
    assert credentials.resolve_key()=='environment-key'
    assert credentials.resolve_key(' entered-key ')=='entered-key'


@pytest.mark.parametrize('value',['','bad\nkey','bad key','x'*513])
def test_invalid_key_is_rejected_before_storage(value,monkeypatch):
    called=[]
    monkeypatch.setattr(credentials,'_keychain',lambda *args:called.append(args))
    with pytest.raises(ValueError):credentials.save_key(value)
    assert not called


def test_other_platforms_never_fall_back_to_plaintext_files(monkeypatch):
    monkeypatch.delenv('GROQ_API_KEY_FILE',raising=False)
    monkeypatch.setattr(credentials.sys,'platform','linux')
    assert credentials.load_key() is None
    with pytest.raises(ValueError,match='macOS Keychain'):credentials.save_key('test-key')


def test_runtime_secret_can_be_read_without_native_keychain(tmp_path,monkeypatch):
    path=tmp_path/'secret';path.write_text('runtime-test-key\n')
    monkeypatch.setenv('GROQ_API_KEY_FILE',str(path))
    monkeypatch.delenv('GROQ_API_KEY',raising=False)
    monkeypatch.setattr(credentials.sys,'platform','linux')
    assert credentials.resolve_key()=='runtime-test-key'
    assert not credentials.can_save_key()
    path.unlink()
    with pytest.raises(ValueError,match='runtime secret'):credentials.load_key()
