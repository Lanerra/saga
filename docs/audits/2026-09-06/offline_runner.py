"""Execute SAGA checks in the secret-free scratch snapshot, never in the live repo.

The Python audit hook forbids network connections and subprocess launch. This is
an audit harness, not a security sandbox for arbitrary hostile Python/extensions.
"""
import os
from pathlib import Path
import runpy
import sys

SCRATCH = Path('/tmp/saga-audit-20260906/source')
os.chdir(SCRATCH)
sys.path.insert(0, str(SCRATCH))
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'


def deny_external(event, args):
    if event in {'socket.connect', 'socket.getaddrinfo', 'socket.sendto', 'subprocess.Popen', 'os.system', 'os.posix_spawn'}:
        raise RuntimeError(f'AUDIT_OFFLINE_BLOCK: {event}')
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        path = Path(os.fsdecode(args[0]))
        if path.name == '.env' or path.name.startswith('.env.'):
            raise RuntimeError('AUDIT_SECRET_READ_BLOCK: environment file')


sys.addaudithook(deny_external)
if sys.argv[1] == 'pytest':
    import pytest
    raise SystemExit(pytest.main(['-p','pytest_asyncio.plugin','-p','pytest_cov.plugin','-p','pytest_timeout','-p','no:cacheprovider','-o','addopts=','-q','--tb=short','--timeout=8',*sys.argv[2:]]))
elif sys.argv[1] == 'cli':
    sys.argv = ['main.py', *sys.argv[2:]]
    runpy.run_path('main.py', run_name='__main__')
else:
    runpy.run_path(sys.argv[1], run_name='__main__')
