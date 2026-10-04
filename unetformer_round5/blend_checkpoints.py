"""Compare parameter averages of adjacent UNetFormer checkpoints on the holdout."""
import argparse
import copy
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'unetformer_round3'))
from round3 import MODES, eval_mode, make_model, save_json


def main():
    parser = argparse.ArgumentParser()
    for name in ('older', 'newer', 'images', 'masks', 'split', 'out'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    split = json.loads(Path(args.split).read_text(encoding='utf-8'))
    old = torch.load(args.older, map_location='cpu', weights_only=True)['model']
    new = torch.load(args.newer, map_location='cpu', weights_only=True)['model']
    if old.keys() != new.keys():
        raise ValueError('Checkpoint architecture differs')
    model = make_model().cuda().eval()
    candidates = []
    best = None
    for old_fraction in (.25, .5, .75):
        state = {key: (new[key] * (1-old_fraction) + old[key] * old_fraction
                       if new[key].is_floating_point() else new[key].clone())
                 for key in new}
        model.load_state_dict(state)
        result = eval_mode(model, args, split['val'], MODES['512_dihedral8'])
        record = {'old_fraction': old_fraction, **result}
        candidates.append(record)
        print(f'old_fraction={old_fraction} mIoU={result["mIoU"]:.8f}', flush=True)
        if best is None or result['mIoU'] > best['mIoU']:
            best = record
            torch.save({'model': state, 'old_fraction': old_fraction,
                        'metrics': result}, out / 'best_blend.pth')
        save_json(out / 'scores.json', {'candidates': candidates, 'best': best})


if __name__ == '__main__':
    main()

