"""Compose launcher: preserve media paths/ownership; bridge Keychain as a runtime secret."""
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def compose_call(arguments, environment=None, key_loader=None):
    env=dict(os.environ if environment is None else environment)
    env['SHUTTLECOACH_APP_DIR']=str(ROOT)
    env['SHUTTLECOACH_UID']=str(os.getuid()) if hasattr(os,'getuid') else '1000'
    env['SHUTTLECOACH_GID']=str(os.getgid()) if hasattr(os,'getgid') else '1000'
    files=['-f',str(ROOT/'compose.yaml')]
    if arguments and arguments[0] in {'up','run','start'} and not env.get('SHUTTLECOACH_NO_SECRET'):
        key=env.get('GROQ_API_KEY','').strip()
        if not key:
            if key_loader is None:
                from shuttlecoach.credentials import load_key
                key_loader=load_key
            try:key=key_loader()
            except (ValueError,OSError):key=None
        if key:
            from shuttlecoach.credentials import clean_key
            env['GROQ_API_KEY']=clean_key(key)
            files+=['-f',str(ROOT/'compose.groq.yaml')]
    return ['docker','compose',*files,*arguments],env


def main():
    if sys.argv[1:2] and sys.argv[1] in {'up','run'}:
        for name in ['data','models','results','.cache']:(ROOT/name).mkdir(exist_ok=True)
    command,env=compose_call(sys.argv[1:])
    try:return subprocess.call(command,cwd=ROOT,env=env)
    except FileNotFoundError:
        print('Docker was not found. Install and start Docker Desktop, then retry.',file=sys.stderr)
        return 1


if __name__=='__main__':raise SystemExit(main())
