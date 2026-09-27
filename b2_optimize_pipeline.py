"""Refine B2 and export only a confirmed improvement over the original B2."""
import argparse
import json
import subprocess
import sys
import shutil
from pathlib import Path


def main(a):
    root = Path(a.workspace).resolve()
    source = Path(__file__).resolve().parent
    old = root / 'outputs/segformer_b2'
    out = root / 'outputs/segformer_b2_optimized'
    initial_report = out / 'calibration_initial.json'
    initial = json.loads(initial_report.read_text(encoding='utf-8'))
    config = json.loads((old/'pipeline_config.json').read_text(encoding='utf-8'))
    old_run = old / 'train512'
    run = out / 'robust640'
    if run.exists():
        raise ValueError('Optimization run already exists')

    def status(stage, **details):
        data = {'stage': stage, **details}
        temporary = out / 'status.tmp'
        temporary.write_text(json.dumps(data, indent=2), encoding='utf-8')
        temporary.replace(out / 'status.json')
        print(json.dumps(data), flush=True)

    def execute(script, args, log):
        with (out/log).open('w', encoding='utf-8') as stream:
            subprocess.run([sys.executable, '-u', str(source/script), *map(str, args)],
                           cwd=root, stdout=stream, stderr=subprocess.STDOUT, check=True)

    try:
        shared = ['--images', config['images'], '--masks', config['masks'], '--split', config['split']]
        if initial['approved']:
            status('predicting_initial_calibration', validation_miou=initial['selected']['all']['mIoU_present_nonignored'])
            execute('b2_predict.py', ['--checkpoint', old_run/'best.pth', '--source', old_run/'model_config',
                '--images', config['test_images'], '--out', out/'submission_b2_calibrated',
                '--sizes', 512, 640, 768, '--hflip', '--class-scales', *initial['selected']['class_scales']],
                'prediction_calibrated.log')
        status('training_robust640', epochs=4, class_weight_power=.5)
        execute('b2_train.py', shared + ['--class-balance', old/'class_balance.json',
            '--source', old_run/'model_config', '--init', old_run/'best.pth', '--out', run,
            '--epochs', 4, '--size', 640, '--batch', 2, '--accum', 4, '--workers', 2,
            '--lr', 8e-6, '--head-lr', 8e-5, '--seed', 20260928, '--robust',
            '--class-weight-power', .5], 'robust640.log')
        status('calibrating_refined')
        report_path = out/'calibration_refined.json'
        execute('b2_calibrate.py', shared + ['--checkpoint', run/'best.pth',
            '--source', run/'model_config', '--class-balance', run/'class_balance_used.json',
            '--out', report_path], 'calibration_refined.log')
        refined = json.loads(report_path.read_text(encoding='utf-8'))
        baseline = initial['candidates'][0]
        choices = []
        for name, report, directory in [('original_calibrated', initial, old_run), ('robust640', refined, run)]:
            selected = report['selected']
            confirmed = all(selected[group]['mIoU_present_nonignored'] > baseline[group]['mIoU_present_nonignored']
                            for group in ('calibration', 'confirmation', 'all'))
            choices.append({'name': name, 'run': str(directory), 'candidate': selected, 'confirmed': confirmed})
        eligible = [c for c in choices if c['confirmed']]
        if not eligible:
            status('no_confirmed_improvement', candidates=choices, original_prediction=str(old/'submission_b2.zip'))
            return
        chosen = max(eligible, key=lambda c: c['candidate']['all']['mIoU_present_nonignored'])
        best = chosen['candidate']['all']['mIoU_present_nonignored']
        original = baseline['all']['mIoU_present_nonignored']
        selection = {'selected': chosen, 'candidates': choices, 'original_b2_miou': original,
                     'optimized_miou': best, 'gain_percentage_points': (best-original)*100,
                     'historical_best_miou': .7841157214158992,
                     'exceeds_historical_best': best > .7841157214158992,
                     'official_score': None, 'single_checkpoint': True,
                     'selection_note': 'Tuning/confirmation subsets reuse historical validation; official gain is unverified.'}
        (out/'selection.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')
        status('predicting', selected=chosen['name'], validation_miou=best,
               gain_percentage_points=(best-original)*100)
        directory = Path(chosen['run'])
        if chosen['name'] == 'original_calibrated':
            shutil.copyfile(out/'submission_b2_calibrated.zip', out/'submission_b2_optimized.zip')
            manifest = json.loads((out/'submission_b2_calibrated.manifest.json').read_text(encoding='utf-8'))
            manifest['archive'] = str(out/'submission_b2_optimized.zip')
            (out/'submission_b2_optimized.manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        else:
            execute('b2_predict.py', ['--checkpoint', directory/'best.pth', '--source', directory/'model_config',
                '--images', config['test_images'], '--out', out/'submission_b2_optimized',
                '--sizes', 512, 640, 768, '--hflip', '--class-scales', *chosen['candidate']['class_scales']], 'prediction.log')
        status('complete', validation_miou=best, gain_percentage_points=(best-original)*100,
               exceeds_historical_best=best > .7841157214158992,
               archive=str(out/'submission_b2_optimized.zip'), official_score=None)
    except Exception as error:
        status('failed', error=str(error))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', default='.')
    main(parser.parse_args())
