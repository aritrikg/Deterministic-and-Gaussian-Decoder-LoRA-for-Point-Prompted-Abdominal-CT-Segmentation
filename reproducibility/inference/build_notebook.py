"""Build a self-contained, size-limited Kaggle notebook from verified archives."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from textwrap import dedent, wrap
from zipfile import ZipFile, ZIP_DEFLATED
import argparse
import ast
import base64
import hashlib
import io
import json

HERE = Path(__file__).resolve().parent
REFERENCE_SHA = 'ccc8581ea2b5dfac4909056585448a9121f6e1d2a122352f6039d5680c185340'
SOURCE_MANIFEST_SHA = '7a1c30bb594633075838cfb886cd77d494dd7e605ed6ab076905a690eb2584cb'
LEDGER_NAME = 'final_accounting/full_run_ledger_dd08f2875a634d048ce29cbe2e4e49d8.json'
WEIGHT_SHA = {
    'deterministic':'40ff778fe706fda8ebac4ad9b5bd5215a2c1e6b89674f654016b22a54dff70d8',
    'balora':'cb0d82ca0b01cb4328ef6afa1c54e246c42093c3320efd519b931063398da0ec',
}


def json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',',':'), allow_nan=False).encode()


def get_reference(completed, accounting, examples):
    with completed.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != REFERENCE_SHA:
            raise ValueError('The supplied full-run ZIP is not the completed F1 archive')
    with ZipFile(completed) as z:
        prefix = z.namelist()[0].split('/')[0]
        def load(name):
            return json.loads(z.read(prefix + '/' + name))
        state = load('run_status.json')
        assert state['full_new_protocol_complete'] and state['selected_policy'] == 'none'
        assert all(state[m+'_training']['selected_checkpoint_sha256'] == sha for m,sha in WEIGHT_SHA.items())
        data = load('data_audit.json')
        test_cases = sorted(load('splits/test.json')['cases'])
        prompts, slices = {}, defaultdict(list)
        archived_dice = {m:{} for m in ['frozen','deterministic','balora']}
        for key in sorted(data['expected_test_keys']):
            for method in archived_dice:
                row = load(f'{method}/test/records/{key.replace(":","_")}.json')
                assert row['key'] == key
                archived_dice[method][key] = row['dice']
                assert prompts.setdefault(key, row['prompt_xy']) == row['prompt_xy']
            slices[key.rsplit(':',1)[0]].append(key)
        assert len(test_cases) == 6 and len(slices) == 460 and len(prompts) == 2466
        configs = {m:load(f'{m}/configuration.json') for m in archived_dice}
    if accounting.suffix.lower() == '.json':
        carried = json.loads(accounting.read_text())
    else:
        with ZipFile(accounting) as z:
            paths = [p for p in z.namelist() if p.endswith('/' + LEDGER_NAME)]
            if len(paths) != 1:
                raise ValueError(f'Expected one accounting file named {LEDGER_NAME}; found {len(paths)}')
            carried = json.loads(z.read(paths[0]))
    assert carried['original_declared_cap_hours'] == 30
    assert carried['active_total_cumulative_cap_hours'] == 240
    assert all(s['closed'] for s in carried['sessions'].values())
    assert abs(carried['accounted_hours'] - 46.367446082027854) < 1e-8
    assert all(row['key'] in prompts for row in examples)
    return {
        'archive_sha256':REFERENCE_SHA, 'archive_prefix':prefix, 'selected_weights':WEIGHT_SHA,
        'configs':configs, 'test_cases':test_cases, 'test_file_sha256':{c:data['data_sha256'][c] for c in test_cases},
        'slices':slices, 'prompts':prompts, 'archived_dice':archived_dice,
        'carried_ledger':carried, 'failure_examples':examples,
    }


def payload(source, reference):
    buf = io.BytesIO()
    with ZipFile(source) as original, ZipFile(buf,'w', ZIP_DEFLATED, compresslevel=9) as package:
        name = next(n for n in original.namelist() if n.endswith('/base_v4/source_hashes.json'))
        prefix = name.rsplit('/',1)[0]
        manifest = json.loads(original.read(name))
        if hashlib.sha256(json_bytes(manifest)).hexdigest() != SOURCE_MANIFEST_SHA:
            raise ValueError('Source manifest differs from the completed run')
        root_files = {'protocol.py','run_budget.py','data_root.py','full_worker.py','inference_common.py',
                      'assets/original_config.json'}
        included = {}
        for file, sha in sorted(manifest.items()):
            is_module = file.endswith('.py') and file.startswith(
                ('auditlib/','source/datasets/','source/models/','source/training/','source/utils/','source/losses/'))
            if file not in root_files and not is_module:
                continue
            data = original.read(f'{prefix}/{file}')
            assert hashlib.sha256(data).hexdigest() == sha, file
            if file.endswith('.py'):
                ast.parse(data, filename=file)
            package.writestr(file, data)
            included[file] = sha
        package.writestr('original_source_subset_sha256.json', json_bytes(included))
        files = {'inference_audit.py':HERE/'inference_audit.py',
                 'replay_diagnostics.py':HERE/'replay_diagnostics.py',
                 'replay_risk.py':HERE.parent/'analysis'/'reanalyze.py'}
        for name, path in files.items():
            data = path.read_bytes()
            ast.parse(data, filename=name)
            package.writestr(name, data)
        package.writestr('reference.json', json_bytes(reference))
    return buf.getvalue()


def cell(kind, content, identity):
    result = {'cell_type':kind,'id':identity,'metadata':{},'source':dedent(content).strip().splitlines(keepends=True)}
    if kind == 'code':
        ast.parse(''.join(result['source']))
        result.update(execution_count=None, outputs=[])
    return result


def build_notebook(raw):
    sha = hashlib.sha256(raw).hexdigest()
    encoded = '\n'.join(wrap(base64.b64encode(raw).decode(), width=100))
    setup = '''\
    from pathlib import Path
    import base64
    import hashlib
    import io
    import json
    import os
    import subprocess
    import sys
    import zipfile

    INPUT_ROOT = Path("/kaggle/input")
    WORK = Path("/kaggle/working")
    RESULT_PATH = ""            # Optional exact completed ZIP or extracted result directory
    BTCV_ROOT = ""              # Optional exact labeled Training directory
    BACKBONE_PATH = ""          # Optional local verified SAM2.1 Hiera Base+ .pt
    RESUME_PATH = ""            # Optional previous Round 3 Results ZIP
    SCOPE = "full"              # "smoke" runs two slices separately
    SESSION_WORK_HOURS = 2.0
    TOTAL_CUMULATIVE_HOURS = 240.0
    RECONCILIATIONS = {}        # Only for an interrupted, unclosed prior session with host-log evidence
    '''
    unpack = (
        f'PAYLOAD_SHA256 = "{sha}"\n'
        'payload = base64.b64decode("""\n' + encoded + '\n""")\n'
        'assert hashlib.sha256(payload).hexdigest() == PAYLOAD_SHA256\n'
        'source = WORK / ("round3_source_" + PAYLOAD_SHA256[:12])\n'
        'source.mkdir(parents=True, exist_ok=True)\n'
        'with zipfile.ZipFile(io.BytesIO(payload)) as z:\n'
        '    assert z.testzip() is None\n'
        '    assert all((source / n).resolve().is_relative_to(source.resolve()) for n in z.namelist())\n'
        '    z.extractall(source)\n'
        'print(f"Verified replay source {PAYLOAD_SHA256[:12]}")'
    )
    run_cell = '''\
    output = WORK / ("BALoRA_Round3_Inference" if SCOPE == "full" else "BALoRA_Round3_Smoke")
    settings = {
        "scope": SCOPE, "input_root": str(INPUT_ROOT), "output_root": str(output),
        "work_root": str(WORK / "round3_work"), "result_path": RESULT_PATH, "btcv_root": BTCV_ROOT,
        "backbone_path": BACKBONE_PATH, "resume_path": RESUME_PATH, "payload_sha256": PAYLOAD_SHA256,
        "session_work_hours": SESSION_WORK_HOURS, "total_cumulative_hours": TOTAL_CUMULATIVE_HOURS,
        "reconciliations": RECONCILIATIONS,
    }
    settings_path = source / ("settings_" + SCOPE + ".json")
    temp_path = settings_path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(settings, indent=2))
    os.replace(temp_path, settings_path)
    args = [sys.executable, "-u", str(source / "inference_audit.py"), str(settings_path)]
    with subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as job:
        for line in job.stdout:
            print(line, end="", flush=True)
        code = job.wait()
    status = output / "status.json"
    if status.is_file():
        state = json.loads(status.read_text())
        print(json.dumps({k: state.get(k) for k in ("status", "full_cohort_complete", "coverage")}, indent=2))
    if code:
        raise RuntimeError(f"Replay exited with code {code}. Review the traceback above and last_error.txt if written.")
    '''
    cells = [
        cell('markdown', '''
        # BALoRA third-round inference audit

        Replay the completed seed-42 checkpoints on the same six BTCV test volumes, 460 slices, and 2,466 oracle-point prompts. The notebook fits no segmentation model.

        1. Attach the labeled 30-volume BTCV cohort and **BALoRA_Full_Run_Results(1).zip** or its extracted contents. This is the completed archive, SHA-256 `ccc8581ea2b5dfac4909056585448a9121f6e1d2a122352f6039d5680c185340`.
        2. Select a CUDA GPU. Enable Internet for the pinned official SAM2 checkout and any missing dependencies. The verified backbone may also be attached as an input. Set exact paths in the next cell if you have multiple copies.
        3. Run all cells. The full run uses the complete test coverage. To check setup on two slices, set `SCOPE = "smoke"` first. The smoke output is separate.
        4. Download **BALoRA_Round3_Inference_Results.zip**. If paused, attach a Round 3 progress ZIP in a later session with the same accelerator and runtime and set `RESUME_PATH`. Completed slices are reused. Round 2 records use their original notebook. Rerunning after a caught error in the same session also reuses saved slices.

        The first two arms compare learned and fixed decoder SD on the matched cohort. The remaining arms measure decoder MC dropout, frozen multimask selection by predicted IoU, and reference-based best Dice over the returned candidates. Each phase logs actual SAM2 substitutions with call and prompt denominators. Predicted-mask region scores and perimeter support a matched-label regional risk analysis. The paired encoder timing uses alternating model order after warm-up.

        The archived cumulative ledger carries 46.3674 hours and records a 240-hour expanded cap. The original 30-hour cap was exceeded by the completed programme. This notebook appends new inference time and has a two-hour cooperative session allowance by default. Its estimates and outputs describe replay only. A complete full test requires all four arms. Actual GPU runtime was unverified when this notebook was prepared.
        ''', 'readme'),
        cell('code', setup, 'configure'),
        cell('code', unpack, 'verified-source'),
        cell('code', run_cell, 'replay'),
        cell('markdown', '''
        The result ZIP includes `status.json`, `per_prompt.csv`, encoder_timing.json, resumable slice records, the cumulative ledger, and actual regenerated arrays and an example figure if the two selected pancreas slices were replayed. Use the overlap estimates only when `full_cohort_complete` is true. The figure illustrates the highest and lowest archived MI-ranked failures of the same organ and volume chosen by the documented rule. The examples have no prespecified detection threshold.
        ''', 'results'),
    ]
    return {'cells':cells,'metadata':{
        'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python','file_extension':'.py'},
        'kaggle':{'accelerator':'gpu','isInternetEnabled':True,'language':'python','sourceType':'notebook'},
    },'nbformat':4,'nbformat_minor':5}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--completed', type=Path, required=True)
    accounting = parser.add_mutually_exclusive_group(required=True)
    accounting.add_argument('--ledger', type=Path, help='Reconciled accounting JSON from the replay input')
    accounting.add_argument('--strengthening', type=Path, help='Strengthening results ZIP containing that ledger')
    parser.add_argument('--original-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    examples = json.loads((HERE.parent/'analysis'/'data'/'replay_failure_targets.json').read_text())
    reference = get_reference(args.completed, args.ledger or args.strengthening, examples)
    raw = payload(args.original_source, reference)
    content = (json.dumps(build_notebook(raw), indent=1) + '\n').encode()
    if len(content) >= 1_000_000:
        raise ValueError('Notebook exceeds the reported Kaggle commit limit')
    args.output.write_bytes(content)
    print(f'{args.output}: {len(content):,} bytes, payload {hashlib.sha256(raw).hexdigest()}')


if __name__ == '__main__':
    main()
