"""Validate the labeled split and build reusable 512-pixel PNG caches."""
import argparse
import json
from pathlib import Path

from segformer_common import prepare


def main():
    parser = argparse.ArgumentParser()
    for name in ('images', 'masks', 'split', 'cache', 'out'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    split = json.loads(Path(args.split).read_text(encoding='utf-8'))
    audit = prepare(args, split, out)
    print(f"cache complete: train={audit['training']} val={audit['validation']}",
          flush=True)


if __name__ == '__main__':
    main()
