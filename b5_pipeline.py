"""Continue an active B5 run, refine one model, validate and export predictions.

Each candidate is evaluated independently. Predictions never combine checkpoints.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main(a):
    root = Path(a.workspace).resolve()
    run = root / a.run
    source = root / 'outputs/UAV-QuickBaseline'
    results = root / 'outputs/round9_b5'
    results.mkdir(parents=True, exist_ok=True)
    status = results / 'pipeline_status.json'

    def record(stage, **details):
        payload = {'stage': stage, 'time': time.strftime('%Y-%m-%d %H:%M:%S'), **details}
        temporary = status.with_suffix('.tmp')
        temporary.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        temporary.replace(status)
        print(json.dumps(payload), flush=True)

    def execute(script, arguments, log_name):
        with (results / log_name).open('w', encoding='utf-8') as log:
            subprocess.run([sys.executable, '-u', str(source / script), *map(str, arguments)],
                           cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)

    try:
        config = read_json(run / 'config.json')
        record('waiting_for_initial_training', run=str(run))
        deadline = time.monotonic() + 24*3600
        while True:
            history = run / 'history.jsonl'
            rows = [json.loads(s) for s in history.read_text(encoding='utf-8').splitlines() if s] if history.exists() else []
            last = run / 'last.pth'
            if rows and rows[-1]['epoch'] == config['epochs'] and last.exists():
                # history is written before checkpoints; wait for the final save to settle.
                snapshot = (last.stat().st_size, last.stat().st_mtime_ns)
                time.sleep(15)
                if snapshot == (last.stat().st_size, last.stat().st_mtime_ns):
                    import torch
                    saved = torch.load(last, map_location='cpu', weights_only=False)
                    epoch = saved['epoch']
                    del saved
                    if epoch == config['epochs']:
                        break
            training_log = root / 'work/round9_b5_train.log'
            if time.monotonic() > deadline:
                raise RuntimeError('Initial training did not complete within 24 hours')
            if training_log.exists():
                if time.time()-training_log.stat().st_mtime > 3600:
                    raise RuntimeError('Training log has not advanced for one hour; inspect the run')
                with training_log.open('rb') as log:
                    log.seek(max(0, training_log.stat().st_size-4000))
                    if b'Traceback (most recent call last)' in log.read():
                        raise RuntimeError('Initial training failed; inspect training log')
            time.sleep(30)

        best_initial = max(r['mIoU_present_nonignored'] for r in rows)
        if best_initial < .70:
            raise RuntimeError(f'Initial validation mIoU {best_initial:.4f} is too low; diagnose before export')
        refined = root / 'outputs/runs/round9_b5_640'
        if refined.exists():
            raise RuntimeError('Refinement directory already exists; inspect before repeating')
        record('refining_640', initial_validation_miou=best_initial)
        shared = []
        for key in ('images', 'masks', 'split', 'class_balance', 'source'):
            shared += ['--'+key.replace('_', '-'), config[key]]
        execute('b5_train.py', shared + [
            '--init', run/'best.pth', '--out', refined, '--epochs', 3,
            '--size', 640, '--batch', 2, '--accum', 4,
            '--lr', 2e-6, '--head-lr', 2e-5, '--seed', 20261002], 'refinement.log')

        candidates = []
        for name, directory in [('b5_512', run), ('b5_640', refined)]:
            report = results / (name+'_tta.json')
            record('validating', candidate=name)
            execute('round4_predict.py', [
                '--source', directory/'model_config', '--checkpoint', directory/'best.pth',
                '--images', config['images'], '--masks', config['masks'], '--split', config['split'],
                '--profiles', root/'outputs/round3/image_profiles.json',
                '--sizes', 512, 640, 768, '--hflip', '--out', report], name+'_validation.log')
            metrics = read_json(report)
            if metrics['ensemble_checkpoint'] is not None:
                raise RuntimeError('Unexpected ensemble configuration')
            candidates.append({'name': name, 'run': str(directory),
                               'report': str(report), 'groups': metrics['groups']})
        selected = max(candidates, key=lambda x: x['groups']['all']['mIoU_present_nonignored'])
        chosen = Path(selected['run'])
        record('predicting_test2', selected=selected['name'], groups=selected['groups'])
        prediction_dir = root / 'outputs/submission_round9_b5_single'
        execute('round4_predict.py', [
            '--source', chosen/'model_config', '--checkpoint', chosen/'best.pth',
            '--images', root/'work/round2_data/images', '--sizes', 512, 640, 768,
            '--hflip', '--out', prediction_dir], 'prediction.log')
        archive = prediction_dir.with_suffix('.zip')
        import zipfile
        with zipfile.ZipFile(archive) as z:
            if len(z.namelist()) != 1300 or z.testzip() is not None:
                raise RuntimeError('Prediction archive failed validation')
        selection = {'selected': selected, 'candidates': candidates,
                     'official_previous': 68.45, 'official_current': None,
                     'single_checkpoint': True, 'sizes': [512, 640, 768], 'hflip': True,
                     'zip': str(archive), 'sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}
        (results/'selection.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')
        record('complete', **selection)
    except Exception as error:
        record('failed', error=str(error))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', default='.')
    parser.add_argument('--run', default='outputs/runs/round9_b5_512')
    main(parser.parse_args())

