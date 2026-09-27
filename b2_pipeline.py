"""Train B2, refine at higher resolution, select on validation, export test PNGs."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main(a):
    source = Path(__file__).resolve().parent
    out = Path(a.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a fresh pipeline output directory')
    for name in ('images', 'masks', 'test_images', 'split'):
        value = Path(getattr(a, name)).resolve()
        if not value.exists():
            raise FileNotFoundError(f'{name}: {value}')
        setattr(a, name, str(value))
    out.mkdir(parents=True, exist_ok=True)

    def record(stage, **details):
        payload = {'stage': stage, **details}
        temporary = out / 'status.tmp'
        temporary.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        temporary.replace(out / 'status.json')
        print(json.dumps(payload), flush=True)

    def execute(script, arguments, log_name):
        with (out / log_name).open('w', encoding='utf-8') as log:
            subprocess.run([sys.executable, '-u', str(source / script), *map(str, arguments)],
                           stdout=log, stderr=subprocess.STDOUT, check=True)

    try:
        record('checking_data')
        from improvedseg import paired_paths, split_pairs, training_class_weights
        from quickseg import images
        from PIL import Image
        training, validation, split = split_pairs(paired_paths(a), a.split, .1, a.seed)
        test = images(a.test_images)
        if not test or len({p.stem for p in test}) != len(test):
            raise ValueError('Empty test data or duplicate stems')
        for path in test:
            with Image.open(path) as im:
                if im.size != (1024, 1024):
                    raise ValueError(f'Test image must be 1024x1024: {path}')
        # No validation or test labels enter the training weights.
        weights, counts = training_class_weights(training)
        balance = out / 'class_balance.json'
        balance.write_text(json.dumps({'weights': weights.tolist(), 'train_pixel_counts': counts}, indent=2), encoding='utf-8')
        (out / 'split.json').write_text(json.dumps(split), encoding='utf-8')
        (out / 'pipeline_config.json').write_text(json.dumps(vars(a), indent=2), encoding='utf-8')
        shared = ['--images', a.images, '--masks', a.masks, '--split', a.split,
                  '--class-balance', balance, '--batch', a.batch, '--accum', a.accum,
                  '--workers', a.workers, '--seed', a.seed]
        first, refined = out / 'train512', out / 'refine640'
        record('training_512', train_images=len(training), validation_images=len(validation), test_images=len(test))
        execute('b2_train.py', shared + ['--source', a.source, '--out', first,
                '--epochs', a.epochs, '--size', 512, '--lr', 2e-5, '--head-lr', 2e-4], 'train512.log')
        record('refining_640')
        execute('b2_train.py', shared + ['--source', first / 'model_config', '--init', first / 'best.pth',
                '--out', refined, '--epochs', a.refine_epochs, '--size', 640,
                '--lr', 2e-6, '--head-lr', 2e-5], 'refine640.log')
        candidates = []
        for name, run in [('b2_512', first), ('b2_640', refined)]:
            for profile, sizes, flip in [('single512', [512], False), ('multiscale', [512, 640, 768], True)]:
                key = name + '_' + profile
                record('validating', candidate=key)
                report = out / (key + '.json')
                args = ['--checkpoint', run / 'best.pth', '--source', run / 'model_config',
                        '--images', a.images, '--masks', a.masks, '--split', a.split,
                        '--out', report, '--sizes', *sizes]
                if flip:
                    args += ['--hflip']
                execute('b2_predict.py', args, key + '.log')
                metrics = json.loads(report.read_text(encoding='utf-8'))
                candidates.append({'name': key, 'run': str(run), 'sizes': sizes, 'hflip': flip,
                                   'validation': metrics})
        selected = max(candidates, key=lambda x: x['validation']['mIoU_present_nonignored'])
        score = selected['validation']['mIoU_present_nonignored']
        selection = {'selected': selected, 'candidates': candidates, 'official_score': None,
                     'single_checkpoint': True,
                     'historical_single_model_validation_best': 0.7841157214158992,
                     'validation_delta_percentage_points': (score - 0.7841157214158992) * 100,
                     'exceeds_historical_validation': score > 0.7841157214158992,
                     'historical_official_single_model_best_user_reported': 68.45,
                     'official_improvement_verified': False}
        (out / 'selection.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')
        chosen = Path(selected['run'])
        record('predicting', candidate=selected['name'])
        args = ['--checkpoint', chosen / 'best.pth', '--source', chosen / 'model_config',
                '--images', a.test_images, '--out', out / 'submission_b2', '--sizes', *selected['sizes']]
        if selected['hflip']:
            args += ['--hflip']
        execute('b2_predict.py', args, 'prediction.log')
        record('complete', selection=str(out / 'selection.json'), archive=str(out / 'submission_b2.zip'),
               validation_miou=score, exceeds_historical_validation=score > 0.7841157214158992,
               official_improvement_verified=False)
    except Exception as error:
        record('failed', error=str(error))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('images', 'masks', 'test-images', 'out'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--split', default=str(Path(__file__).resolve().parent / 'reports/round2/split.json'))
    p.add_argument('--source', default='nvidia/segformer-b2-finetuned-ade-512-512')
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--refine-epochs', type=int, default=3)
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--accum', type=int, default=8)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--seed', type=int, default=20260927)
    main(p.parse_args())
