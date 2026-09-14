"""Plan, prepare and execute the archived paper workflows in isolated workspaces.

Listing and planning never import or execute research drivers. Preparation
copies source and explicitly supplied data. Only `run` executes a driver.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESEARCH = HERE / 'research'
RECEIPT = 'pipeline-run.json'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def relative(name):
    p = Path(name)
    if p.is_absolute() or '..' in p.parts or not p.parts:
        raise ValueError(f'Expected a safe relative path: {name}')
    return p

def recipes():
    result = {}
    for path in sorted((HERE / 'recipes').glob('*.json')):
        payload = json.loads(path.read_text())
        entries = payload if isinstance(payload, list) else payload['recipes']
        for item in entries:
            if item['id'] in result:
                raise ValueError(f'Duplicate stage: {item["id"]}')
            relative(item['entrypoint'])
            for name in item.get('inputs', []) + item.get('outputs', []) + item.get('prepare_directories', []):
                relative(name)
            result[item['id']] = item
    return result

def resolve_inputs(recipe, external_data=None, input_map=None):
    mapping = {}
    if input_map:
        map_path = Path(input_map).resolve()
        mapping = json.loads(map_path.read_text())
        mapping = {k: (Path(v) if Path(v).is_absolute() else map_path.parent / v).resolve()
                   for k, v in mapping.items()}
    base = Path(external_data).resolve() if external_data else None
    result = {}
    for name in recipe.get('inputs', []):
        rel = relative(name)
        source = mapping.get(name, base / rel if base else None)
        legacy = name.replace('gpt6_astra_validation_scripts/', '.paper_agent/', 1)
        if name not in mapping and legacy != name:
            if legacy in mapping:
                source = mapping[legacy]
            elif base and not source.exists() and (base / legacy).is_file():
                source = base / legacy
        result[name] = source
    return result

def plan(recipe, external_data=None, input_map=None):
    supplied = resolve_inputs(recipe, external_data, input_map)
    inputs = []
    for name, path in supplied.items():
        present = path is not None and path.is_file()
        inputs.append({'destination': name, 'source': str(path) if path else None,
                       'present': present, 'bytes': path.stat().st_size if present else None})
    missing_modules = [m for m in recipe.get('requires_modules', [])
                       if importlib.util.find_spec(m) is None]
    return {'stage': recipe['id'], 'description': recipe['description'],
            'entrypoint': recipe['entrypoint'], 'args': recipe.get('args', []),
            'inputs': inputs, 'input_bytes': sum(i['bytes'] or 0 for i in inputs),
            'missing_modules': missing_modules, 'outputs': recipe.get('outputs', []),
            'expected_resources': recipe.get('expected_resources', 'Not measured'),
            'limitations': recipe.get('limitations', []),
            'ready_to_prepare': all(i['present'] for i in inputs)}

def prepare(recipe, workspace, external_data=None, input_map=None):
    work = Path(workspace).resolve()
    if work.exists():
        raise ValueError('Workspace must not exist: preparation never overwrites previous runs.')
    if work.is_relative_to(ROOT) and not work.is_relative_to(ROOT/'outputs'):
        raise ValueError('Workspace must be separate from bundled source and data.')
    report = plan(recipe, external_data, input_map)
    if not report['ready_to_prepare']:
        missing = [i['destination'] for i in report['inputs'] if not i['present']]
        raise ValueError('Missing external inputs: ' + ', '.join(missing))
    if not (RESEARCH / recipe['entrypoint']).is_file():
        raise ValueError('Bundled entrypoint is missing.')
    inputs = resolve_inputs(recipe, external_data, input_map)
    for name in inputs:
        target = RESEARCH / relative(name)
        if target.exists():
            raise ValueError(f'Input would replace bundled code/configuration: {name}')
    work.parent.mkdir(parents=True, exist_ok=True)
    # copytree requires a fresh path; external files are copied, never symlinked.
    shutil.copytree(RESEARCH, work, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    receipt = {'stage': recipe['id'], 'recipe': recipe, 'workspace': str(work),
               'state': 'PREPARING', 'created_unix': time.time(), 'inputs': {},
               'source_hashes': {}, 'python': sys.version}
    (work / RECEIPT).write_text(json.dumps(receipt, indent=2))
    for source in sorted(RESEARCH.rglob('*')):
        if source.is_file() and '__pycache__' not in source.parts and source.suffix != '.pyc':
            receipt['source_hashes'][str(source.relative_to(RESEARCH))] = sha(source)
    reference_path = HERE / 'INPUT_MANIFEST.json'
    references = ({i['path']: i['sha256'] for i in json.loads(reference_path.read_text())['files']}
                  if reference_path.exists() else {})
    for name, source in inputs.items():
        target = work / relative(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        before = sha(source)
        shutil.copy2(source, target)
        if sha(target) != before or sha(source) != before:
            raise RuntimeError(f'Input changed during copy: {source}')
        receipt['inputs'][name] = {'source': str(source), 'sha256': before,
                                   'bytes': target.stat().st_size,
                                   'reference_sha256': references.get(name),
                                   'matches_reference': before == references[name] if name in references else None}
    receipt['path_relocations'] = []
    for rule in recipe.get('input_json_relocations', []):
        name = str(relative(rule['file']))
        if name not in receipt['inputs']:
            raise ValueError('Relocation must target an explicitly copied input.')
        path = work / name
        content = json.loads(path.read_text())
        rows = content if rule.get('array_key') is None else content[rule['array_key']]
        directory = relative(rule['target_directory'])
        for index, row in enumerate(rows):
            old_path = row[rule['field']]
            target_name = str(directory / Path(old_path).name)
            if target_name not in receipt['inputs'] or not (work / target_name).is_file():
                raise ValueError(f'Relocated target is not a supplied input: {target_name}')
            row[rule['field']] = str(work / target_name)
            receipt['path_relocations'].append(dict(file=name, row=index, field=rule['field'],
                                                     original=old_path, prepared=row[rule['field']]))
        receipt['inputs'][name]['original_sha256'] = receipt['inputs'][name]['sha256']
        path.write_text(json.dumps(content, indent=2) + '\n')
        receipt['inputs'][name]['sha256'] = sha(path)
    for name in recipe.get('prepare_directories', []):
        (work / relative(name)).mkdir(parents=True, exist_ok=True)
    receipt['state'] = 'PREPARED'
    (work / RECEIPT).write_text(json.dumps(receipt, indent=2))
    return receipt

def execute(workspace, timeout_seconds=None):
    work = Path(workspace).resolve()
    receipt_path = work / RECEIPT
    receipt = json.loads(receipt_path.read_text())
    if receipt['workspace'] != str(work) or receipt['state'] != 'PREPARED':
        raise ValueError('Run requires a PREPARED workspace at its recorded path. Prepare a fresh workspace for retries.')
    recipe = receipt['recipe']
    for name, expected in receipt['source_hashes'].items():
        if sha(work / relative(name)) != expected:
            raise ValueError(f'Prepared code/configuration changed: {name}')
    for name, info in receipt['inputs'].items():
        if sha(work / relative(name)) != info['sha256']:
            raise ValueError(f'Prepared input changed: {name}')
    missing = [m for m in recipe.get('requires_modules', [])
               if importlib.util.find_spec(m) is None]
    if missing:
        raise ValueError('Missing dependencies (nothing installed automatically): ' + ', '.join(missing))
    command = [sys.executable, str(work / relative(recipe['entrypoint']))]
    command += [arg.replace('{workspace}', str(work)) for arg in recipe.get('args', [])]
    env = os.environ.copy()
    env.update(PYTHONPATH=str(work), MPLBACKEND='Agg',
               MPLCONFIGDIR=str(work/'.runtime/matplotlib'),
               NUMBA_CACHE_DIR=str(work/'.runtime/numba'), PYTHONDONTWRITEBYTECODE='1')
    receipt.update(state='RUNNING', command=command, started_unix=time.time())
    receipt_path.write_text(json.dumps(receipt, indent=2))
    result = None
    try:
        with (work/'pipeline.stdout.log').open('w') as stdout, (work/'pipeline.stderr.log').open('w') as stderr:
            result = subprocess.run(command, cwd=work, env=env, stdout=stdout, stderr=stderr, timeout=timeout_seconds)
        receipt['returncode'] = result.returncode
        receipt['outputs_present'] = {name: (work/relative(name)).exists() for name in recipe.get('outputs', [])}
        receipt['inputs_unchanged'] = all(sha(work/relative(name)) == info['sha256'] for name, info in receipt['inputs'].items())
        receipt['state'] = ('COMPLETED' if result.returncode == 0 and
                            all(receipt['outputs_present'].values()) and receipt['inputs_unchanged'] else 'FAILED')
    except BaseException as error:
        receipt.update(state='INTERRUPTED_OR_FAILED', error=repr(error))
        raise
    finally:
        receipt['finished_unix'] = time.time()
        receipt_path.write_text(json.dumps(receipt, indent=2))
    return receipt

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('list', help='List available stages without importing drivers')
    for action in ['plan', 'prepare']:
        p = sub.add_parser(action)
        p.add_argument('stage')
        p.add_argument('--external-data', type=Path)
        p.add_argument('--input-map', type=Path)
        if action == 'prepare':
            p.add_argument('--workspace', type=Path, required=True)
    p = sub.add_parser('run', help='Execute the stage in a prepared workspace; may be expensive')
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--timeout-seconds', type=float, help='Optional wall-clock limit; original resource assertions are retained')
    args = parser.parse_args()
    try:
        registry = recipes()
        if args.action == 'list':
            for name, recipe in registry.items():
                print(f'{name:28s} {recipe["description"]}')
            return
        if args.action == 'run':
            result = execute(args.workspace, args.timeout_seconds)
        else:
            if args.stage not in registry:
                parser.error(f'Unknown stage {args.stage}; use list.')
            recipe = registry[args.stage]
            if args.action == 'plan':
                result = plan(recipe, args.external_data, args.input_map)
            else:
                result = prepare(recipe, args.workspace, args.external_data, args.input_map)
        print(json.dumps(result, indent=2))
        if args.action == 'run' and result['state'] != 'COMPLETED':
            raise SystemExit(1)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f'{error}\n')

if __name__ == '__main__':
    main()
