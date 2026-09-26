"""Reproduce the round-five 768px fine-tuning recipe using the round-four trainer."""
import argparse
import json
from pathlib import Path
from round4_train import train


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data-root', 'source', 'init', 'out', 'split', 'class-balance'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--size', type=int, default=768)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--accum', type=int, default=2)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--lr', type=float, default=5e-6)
    p.add_argument('--seed', type=int, default=20260925)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    dry_run = a.dry_run
    root = Path(a.data_root)
    del a.dry_run, a.data_root
    a.images = str(root / 'train_images')
    a.masks = str(root / 'train_masks')
    a.mild = True
    if dry_run:
        print(json.dumps(vars(a), indent=2))
    else:
        train(a)


if __name__ == '__main__':
    main()
