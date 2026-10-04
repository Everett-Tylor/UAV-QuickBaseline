"""Train DINOv3 ViT-B/16, refine with guarded pseudo labels, and export predictions."""
import argparse
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path

from dino_pseudo import digest


def improved(student, teacher):
    old, new = teacher['groups']['all'], student['groups']['all']
    return (new['mIoU_present_nonignored'] > old['mIoU_present_nonignored']
            and new['per_class_IoU']['5'] > old['per_class_IoU']['5'])


def main(args):
    code = Path(__file__).resolve().parent
    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a fresh output directory')
    out.mkdir(parents=True, exist_ok=True)

    def record(stage, **details):
        status = {'stage': stage, 'time': time.strftime('%Y-%m-%d %H:%M:%S'), **details}
        temporary = out / 'status.tmp'
        temporary.write_text(json.dumps(status, indent=2), encoding='utf-8')
        temporary.replace(out / 'status.json')
        print(json.dumps(status), flush=True)

    def run(script, values, log_name):
        with (out / log_name).open('w', encoding='utf-8') as log:
            subprocess.run([sys.executable, '-u', str(code / script), *map(str, values)],
                           stdout=log, stderr=subprocess.STDOUT, check=True)

    try:
        paths = {name: Path(getattr(args, name)).resolve() for name in
                 ('source', 'images', 'masks', 'test_images', 'split', 'class_balance')}
        missing = [f'{name}: {path}' for name, path in paths.items() if not path.exists()]
        if missing:
            record('awaiting_required_files', missing=missing)
            return
        paths['profiles'] = Path(args.profiles).resolve() if args.profiles else out / 'audit' / 'image_profiles.json'
        if args.profiles and not paths['profiles'].exists():
            raise FileNotFoundError(f'Supplied image audit missing: {paths["profiles"]}')
        source = paths['source']
        config = json.loads((source / 'config.json').read_text(encoding='utf-8'))
        expected = {'model_type': 'dinov3_vit', 'hidden_size': 768,
                    'num_hidden_layers': 12, 'num_attention_heads': 12,
                    'use_gated_mlp': False, 'patch_size': 16}
        if any(config.get(key) != value for key, value in expected.items()):
            raise ValueError('Source is not the DINOv3 ViT-B/16 architecture')
        if not args.teacher and not any((source / name).exists() for name in
                   ('model.safetensors', 'model.safetensors.index.json', 'pytorch_model.bin',
                    'pytorch_model.bin.index.json')):
            record('awaiting_pretrained_weights', source=str(source))
            return
        from quickseg import images
        if len(images(paths['test_images'])) != 1300:
            raise ValueError('Expected 1300 test2 images')
        if not paths['profiles'].exists():
            record('auditing_images')
            run('round3_audit.py', ['--images', paths['images'], '--test-images', paths['test_images'],
                '--split', paths['split'], '--out', paths['profiles'].parent], 'audit.log')
            audit = json.loads((paths['profiles'].parent / 'data_audit.json').read_text(encoding='utf-8'))
            if audit['cross_group_exact_duplicates']:
                raise ValueError('Exact duplicate pixels found across train/validation/test2')
        if len(json.loads(paths['profiles'].read_text(encoding='utf-8'))) != 8296:
            raise ValueError('Expected audited 6996 labeled and 1300 test2 images')

        common = ['--images', paths['images'], '--masks', paths['masks'],
                  '--split', paths['split'], '--class-balance', paths['class_balance'],
                  '--source', source, '--size', args.size, '--workers', args.workers,
                  '--augmentation', 'robust', '--focus-bare', '--mixstyle']
        teacher_args = common + ['--batch', 16, '--accum', 1,
                       '--freeze-epochs', 1, '--epochs', args.teacher_epochs,
                       '--lr', 1e-5, '--head-lr', 1e-4]
        if args.teacher:
            teacher = Path(args.teacher).resolve()
            if not teacher.is_file():
                raise FileNotFoundError(f'Teacher checkpoint missing: {teacher}')
            record('using_supplied_teacher', teacher=str(teacher), sha256=digest(teacher))
        else:
            record('supervised_smoke')
            run('dino_train.py', teacher_args + ['--out', out / 'teacher_smoke', '--smoke'],
                'teacher_smoke.log')
            record('supervised_training')
            run('dino_train.py', teacher_args + ['--out', out / 'teacher'], 'teacher_train.log')
            teacher = out / 'teacher' / 'best.pth'
        views = ['--source', source, '--sizes', 512, 640, 768, 896,
                 '--hflip', '--brightness-floor', .35, '--max-brightening', 1.25]
        record('teacher_validation')
        teacher_report = out / 'teacher_validation.json'
        run('dino_predict.py', views + ['--checkpoint', teacher, '--images', paths['images'],
            '--masks', paths['masks'], '--split', paths['split'], '--profiles', paths['profiles'],
            '--out', teacher_report, '--workers', args.workers], 'teacher_validation.log')

        pseudo = out / 'pseudo_masks'
        record('pseudo_generation', teacher_sha256=digest(teacher))
        run('dino_pseudo.py', ['--source', source, '--checkpoint', teacher,
            '--images', paths['test_images'], '--profiles', paths['profiles'], '--out', pseudo,
            '--sizes', 512, 640, 768, 896, '--confidence', .95, '--agreement', .75,
            '--bare-confidence', .98, '--bare-agreement', .875,
            '--workers', args.workers], 'pseudo_generation.log')
        manifest = json.loads((pseudo / 'manifest.json').read_text(encoding='utf-8'))
        usable = [row for row in manifest['images'] if row['coverage'] >= manifest['min_coverage']]
        bare_pixels = sum(row['retained_bare_pixels'] for row in usable)
        if len(usable) < 100 or bare_pixels < 10000:
            record('insufficient_pseudo_labels', usable=len(usable), retained_bare_pixels=bare_pixels)
            return

        student_args = common + ['--batch', 8, '--accum', 2,
                        '--freeze-epochs', 0, '--epochs', args.student_epochs,
                        '--lr', 1e-6, '--head-lr', 1e-5, '--init', teacher,
                        '--pseudo-manifest', pseudo / 'manifest.json',
                        '--pseudo-batch', 1, '--pseudo-weight', .2]
        record('pseudo_smoke', usable=len(usable), retained_bare_pixels=bare_pixels)
        run('dino_train.py', student_args + ['--out', out / 'student_smoke', '--smoke'],
            'student_smoke.log')
        record('pseudo_training')
        run('dino_train.py', student_args + ['--out', out / 'student'], 'student_train.log')
        student = out / 'student' / 'best.pth'
        report = out / 'student_validation.json'
        record('student_validation')
        run('dino_predict.py', views + ['--checkpoint', student, '--images', paths['images'],
            '--masks', paths['masks'], '--split', paths['split'], '--profiles', paths['profiles'],
            '--out', report, '--workers', args.workers], 'student_validation.log')
        old = json.loads(teacher_report.read_text(encoding='utf-8'))
        new = json.loads(report.read_text(encoding='utf-8'))
        selection = {'teacher': old['groups'], 'student': new['groups'],
                     'usable_pseudo_images': len(usable), 'retained_bare_pixels': bare_pixels,
                     'improved_over_teacher': improved(new, old),
                     'exceeds_historical_78_4116_miou':
                         new['groups']['all']['mIoU_present_nonignored'] > .7841157214158992,
                     'official_score': None}
        (out / 'selection.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')

        record('predicting_test2', improved_over_teacher=selection['improved_over_teacher'])
        prediction = out / 'submission_dino_vitb_bare'
        run('dino_predict.py', views + ['--checkpoint', student,
            '--images', paths['test_images'], '--out', prediction,
            '--workers', args.workers], 'prediction.log')
        archive = prediction.with_suffix('.zip')
        with zipfile.ZipFile(archive) as z:
            if len(z.namelist()) != 1300 or z.testzip() is not None:
                raise ValueError('Prediction archive failed integrity check')
        record('complete', archive=str(archive), sha256=digest(archive),
               improved_over_teacher=selection['improved_over_teacher'], official_score=None)
    except Exception as error:
        record('failed', error=str(error))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('source', 'images', 'masks', 'test-images', 'split',
                'class-balance', 'out'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--profiles', help='Optional image audit from round3_audit.py; generated when absent')
    parser.add_argument('--teacher', help='Existing nine-class DINOv3 ViT-B teacher checkpoint; skips supervised bootstrap')
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--teacher-epochs', type=int, default=8)
    parser.add_argument('--student-epochs', type=int, default=3)
    parser.add_argument('--workers', type=int, default=2)
    main(parser.parse_args())
