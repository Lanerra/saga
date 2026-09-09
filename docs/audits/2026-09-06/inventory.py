"""Read-only source census for SAGA; writes audit evidence only."""
import ast
import collections
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path('/home/dlewis3/Desktop/AI/saga')
OUT = ROOT / 'docs/audits/2026-09-06'
SOURCE_ROOTS = ['config', 'core', 'data_access', 'models', 'orchestration', 'processing', 'prompts', 'ui', 'utils', 'tests']

def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT).decode().rstrip('\n')

def main():
    paths = sorted(set([p for d in SOURCE_ROOTS for p in (ROOT / d).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in {'.py', '.j2', '.md'}] + list(ROOT.glob('*.py'))))
    tracked = set(git('ls-files').splitlines())
    rows, symbols, imports, errors = [], [], [], []
    for p in paths:
        relative = str(p.relative_to(ROOT))
        raw = p.read_bytes()
        text = raw.decode('utf-8')
        role = 'tests' if relative.startswith('tests/') else 'production_python' if p.suffix == '.py' else 'prompt'
        row = dict(path=relative, role=role, tracked=relative in tracked, bytes=len(raw), lines=len(text.splitlines()), sha256=hashlib.sha256(raw).hexdigest())
        if p.suffix == '.py':
            try:
                tree = ast.parse(text, filename=relative)
                row['purpose'] = (ast.get_docstring(tree) or '').split('\n')[0]
                row['functions'] = sum(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in ast.walk(tree))
                row['classes'] = sum(isinstance(n, ast.ClassDef) for n in ast.walk(tree))
                row['broad_except'] = sum(isinstance(n, ast.ExceptHandler) and (n.type is None or isinstance(n.type, ast.Name) and n.type.id in {'Exception','BaseException'}) for n in ast.walk(tree))
                for n in ast.walk(tree):
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        symbols.append(dict(path=relative, name=n.name, kind=type(n).__name__, line=n.lineno, end=n.end_lineno))
                    elif isinstance(n, ast.ImportFrom):
                        imports.append(dict(path=relative, line=n.lineno, module=n.module, level=n.level, names=[a.name for a in n.names]))
                    elif isinstance(n, ast.Import):
                        imports.append(dict(path=relative, line=n.lineno, module=None, level=0, names=[a.name for a in n.names]))
            except SyntaxError as e:
                errors.append(dict(path=relative, line=e.lineno, error=e.msg))
        rows.append(row)
    summary = dict(head=git('rev-parse','HEAD'), branch=git('branch','--show-current'), syntax_errors=errors)
    for role in ['production_python','tests','prompt']:
        group=[r for r in rows if r['role']==role]
        summary[role] = dict(files=len(group),lines=sum(r['lines'] for r in group),bytes=sum(r['bytes'] for r in group),broad_except=sum(r.get('broad_except',0) for r in group))
    summary['by_directory'] = {d:dict(files=len(g),lines=sum(r['lines'] for r in g)) for d in SOURCE_ROOTS for g in [[r for r in rows if r['path'].startswith(d+'/')]]}
    summary['largest_production'] = sorted([r for r in rows if r['role']=='production_python'], key=lambda r:r['lines'],reverse=True)[:20]
    summary['status_counts'] = dict(collections.Counter(line[:2] for line in git('status','--porcelain=v1','--untracked-files=all').splitlines()))
    summary['tracked_files'] = len(tracked)
    for name, data in [('source-inventory.json',rows),('symbols.json',symbols),('imports.json',imports),('inventory-summary.json',summary)]:
        (OUT/name).write_text(json.dumps(data,indent=2)+'\n')
    (OUT/'git-status-before.txt').write_text(git('status','--porcelain=v1','--branch','--untracked-files=all')+'\n')
    (OUT/'git-content-status-before.txt').write_text(git('-c','core.fileMode=false','status','--short','--branch','--untracked-files=all')+'\n')
    (OUT/'git-branches-before.txt').write_text(git('branch','-vv')+'\n')
    print(json.dumps(summary,indent=2))

if __name__ == '__main__':
    main()
