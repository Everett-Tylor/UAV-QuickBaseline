"""Recreate the selected round-four test2 ZIP from the saved checkpoint."""
import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'unetformer_round3'))
from round3 import MODES, export, make_model, save_json


def main():
    parser = argparse.ArgumentParser()
    for name in ('run', 'images', 'out'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    run, out = Path(args.run), Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    selection = json.loads((run / 'selection.json').read_text(encoding='utf-8'))
    if not selection['improved']:
        raise ValueError('Round-four checkpoint did not outperform the previous submission')
    model = make_model()
    model.load_state_dict(torch.load(run / 'best.pth', map_location='cpu', weights_only=True)['model'])
    export(model.cuda().eval(), args.images, out, MODES['512_dihedral8'])
    (out / 'submission_unetformer_round3_test2.zip').rename(
        out / 'submission_unetformer_round4_test2.zip')
    manifest = json.loads((out / 'submission_manifest.json').read_text(encoding='utf-8'))
    manifest['file'] = 'submission_unetformer_round4_test2.zip'
    save_json(out / 'submission_manifest.json', manifest)


if __name__ == '__main__':
    main()

