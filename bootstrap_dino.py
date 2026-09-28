"""Download licensed DINOv3 weights and train a supervised segmentation teacher."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from huggingface_hub import get_token, snapshot_download


MODEL = 'facebook/dinov3-vitb16-pretrain-lvd1689m'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('images', 'masks', 'split', 'class-balance', 'out'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--epochs', type=int, default=8)
    args = parser.parse_args()
    if not get_token():
        raise SystemExit('Hugging Face login required. Run `hf auth login` after access to the official DINOv3 model is granted.')
    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit('Use a fresh output directory')
    out.mkdir(parents=True, exist_ok=True)
    source = out / 'pretrained'
    snapshot_download(repo_id=MODEL, local_dir=source, allow_patterns=['config.json', '*.safetensors'])
    if not (source / 'config.json').exists() or not list(source.glob('*.safetensors')):
        raise RuntimeError('Official pretrained model files are incomplete')
    (out / 'provenance.json').write_text(json.dumps({'model': MODEL, 'source': str(source)}, indent=2), encoding='utf-8')
    command = [sys.executable, '-u', str(Path(__file__).with_name('dino_train.py')),
               '--source', str(source), '--images', args.images, '--masks', args.masks,
               '--split', args.split, '--class-balance', args.class_balance,
               '--out', str(out / 'teacher'), '--epochs', str(args.epochs),
               '--size', '512', '--batch', '1', '--accum', '8', '--workers', '2',
               '--augmentation', 'mixed', '--focus-bare']
    with (out / 'train.log').open('w', encoding='utf-8') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(out / 'teacher' / 'best.pth')


if __name__ == '__main__':
    main()
