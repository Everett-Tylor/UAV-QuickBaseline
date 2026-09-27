"""Wait for round two, then test higher resolution using one B2 model at a time."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def select_confirmed(baseline, candidates):
    proposed = max([baseline, *candidates],
                   key=lambda c: c['candidate']['calibration']['mIoU_present_nonignored'])
    improved = all(proposed['candidate'][g]['mIoU_present_nonignored'] >
                   baseline['candidate'][g]['mIoU_present_nonignored']
                   for g in ('calibration', 'confirmation', 'all'))
    return (proposed, True) if improved else (baseline, False)


def main(a):
    root = Path(a.workspace).resolve()
    source = Path(__file__).resolve().parent
    previous = root/'outputs/segformer_b2_optimized'
    original = root/'outputs/segformer_b2'
    out = root/'outputs/segformer_b2_highres'
    if out.exists() and any(out.iterdir()):
        raise ValueError('High-resolution experiment directory already exists')
    out.mkdir(parents=True, exist_ok=True)

    def status(stage, **details):
        payload = {'stage': stage, 'updated_at': time.strftime('%Y-%m-%d %H:%M:%S'), **details}
        temp = out/'status.tmp'
        temp.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        temp.replace(out/'status.json')
        print(json.dumps(payload), flush=True)

    def execute(script, args, log):
        with (out/log).open('w', encoding='utf-8') as stream:
            subprocess.run([sys.executable, '-u', str(source/script), *map(str, args)],
                           cwd=root, stdout=stream, stderr=subprocess.STDOUT, check=True)

    try:
        status('waiting_for_previous_optimization')
        deadline = time.monotonic() + 8*3600
        while True:
            previous_status = json.loads((previous/'status.json').read_text(encoding='utf-8'))
            if previous_status['stage'] == 'complete':
                break
            if previous_status['stage'] in ('failed', 'no_confirmed_improvement'):
                raise RuntimeError('Previous experiment did not complete successfully; inspect it before continuing')
            if time.monotonic() > deadline:
                raise TimeoutError('Previous experiment has not completed within eight hours')
            time.sleep(30)
        prior = json.loads((previous/'selection.json').read_text(encoding='utf-8'))
        config = json.loads((original/'pipeline_config.json').read_text(encoding='utf-8'))
        baseline = {**prior['selected'], 'name': 'previous_best', 'sizes': [512, 640, 768]}
        base_run = Path(baseline['run'])
        weights = base_run/'class_balance_used.json'
        if not weights.exists():
            weights = original/'class_balance.json'
        shared = ['--images', config['images'], '--masks', config['masks'], '--split', config['split']]
        candidates = []

        def evaluate(name, run, balance, sizes):
            status('validating', candidate=name, sizes=sizes)
            report_path = out/(name+'.json')
            execute('b2_calibrate.py', shared + ['--source', run/'model_config', '--checkpoint', run/'best.pth',
                '--class-balance', balance, '--out', report_path, '--sizes', *sizes], name+'.log')
            report = json.loads(report_path.read_text(encoding='utf-8'))
            item = {'name': name, 'run': str(run), 'sizes': sizes, 'candidate': report['selected']}
            candidates.append(item)
            return item

        high_base = evaluate('previous_best_highres', base_run, weights, [640, 768, 896])
        # This changes the inference recipe only; both choices use the same model weights.
        before_training, _ = select_confirmed(baseline, [high_base])
        (out/'inference_comparison.json').write_text(json.dumps({
            'baseline': baseline, 'candidate': high_base, 'selected': before_training}, indent=2), encoding='utf-8')
        run = out/'finetune768'
        train_args = shared + ['--class-balance', original/'class_balance.json',
            '--source', base_run/'model_config', '--init', base_run/'best.pth',
            '--size', 768, '--batch', 1, '--accum', 8, '--workers', 2,
            '--lr', 2e-6, '--head-lr', 2e-5, '--seed', 20260929, '--robust', '--class-weight-power', .5]
        status('smoke_testing_768')
        execute('b2_train.py', train_args + ['--out', out/'smoke768', '--epochs', 1, '--smoke'], 'smoke768.log')
        status('training_768', epochs=3, initialization=str(base_run/'best.pth'))
        execute('b2_train.py', train_args + ['--out', run, '--epochs', 3], 'finetune768.log')
        evaluate('finetuned_standard', run, run/'class_balance_used.json', [512, 640, 768])
        evaluate('finetuned_highres', run, run/'class_balance_used.json', [640, 768, 896])
        selected, improved = select_confirmed(baseline, candidates)
        score = selected['candidate']['all']['mIoU_present_nonignored']
        base_score = baseline['candidate']['all']['mIoU_present_nonignored']
        result = {'selected': selected, 'baseline': baseline, 'candidates': candidates,
                  'improved': improved, 'validation_miou': score,
                  'gain_percentage_points': (score-base_score)*100,
                  'historical_best_miou': .7841157214158992,
                  'exceeds_historical_best': score > .7841157214158992,
                  'official_score': None,
                  'note': 'All comparisons reuse historical validation. Confirmation is not a new independent test set.'}
        (out/'selection.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        if not improved:
            status('complete_no_improvement', validation_miou=score,
                   retained_archive=str(previous/'submission_b2_optimized.zip'))
            return
        status('predicting', validation_miou=score, gain_percentage_points=(score-base_score)*100)
        chosen = Path(selected['run'])
        execute('b2_predict.py', ['--source', chosen/'model_config', '--checkpoint', chosen/'best.pth',
            '--images', config['test_images'], '--out', out/'submission_b2_highres',
            '--sizes', *selected['sizes'], '--hflip', '--class-scales', *selected['candidate']['class_scales']], 'prediction.log')
        status('complete', validation_miou=score, gain_percentage_points=(score-base_score)*100,
               archive=str(out/'submission_b2_highres.zip'), official_score=None)
    except Exception as error:
        status('failed', error=str(error))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', default='.')
    main(parser.parse_args())
