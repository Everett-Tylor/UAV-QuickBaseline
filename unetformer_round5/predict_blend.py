"""Export the best validated round-3/round-4 weight average."""
import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'unetformer_round3'))
from round3 import MODES, export, make_model, save_json


def main():
    parser = argparse.ArgumentParser()
    for name in ('checkpoint', 'images', 'out'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    model = make_model()
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu',
                                     weights_only=True)['model'])
    export(model.cuda().eval(), args.images, out, MODES['512_dihedral8'])
    (out / 'submission_unetformer_round3_test2.zip').rename(
        out / 'submission_unetformer_round5_test2.zip')
    manifest = json.loads((out / 'submission_manifest.json').read_text(encoding='utf-8'))
    manifest['file'] = 'submission_unetformer_round5_test2.zip'
    save_json(out / 'submission_manifest.json', manifest)


if __name__ == '__main__':
    main()

