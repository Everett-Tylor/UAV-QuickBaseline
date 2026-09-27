"""Predeclared class-weight correction grid with a separate confirmation subset."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from b2_model import Segmenter
from b2_predict import probabilities
from improvedseg import paired_paths, split_pairs, metrics_from_hist


def choose_candidate(candidates):
    baseline = candidates[0]
    proposed = max(candidates, key=lambda c: c['calibration']['mIoU_present_nonignored'])
    approved = (proposed['calibration']['mIoU_present_nonignored'] > baseline['calibration']['mIoU_present_nonignored']
                and proposed['confirmation']['mIoU_present_nonignored'] > baseline['confirmation']['mIoU_present_nonignored']
                and proposed['all']['mIoU_present_nonignored'] > baseline['all']['mIoU_present_nonignored'])
    return proposed if approved else baseline, approved


@torch.inference_mode()
def main(a):
    torch.set_num_threads(4)
    out = Path(a.out)
    if out.exists():
        raise ValueError('Calibration report already exists')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    state = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    model = Segmenter(a.source, config_only=True).to(device).eval()
    model.load_state_dict(state['model'], strict=True)
    _, pairs, split = split_pairs(paired_paths(a), a.split, .1, 2026)
    if split != state['split']:
        raise ValueError('Split mismatch')
    names = sorted(p.name for p, _ in pairs)
    random.Random(20260928).shuffle(names)
    calibration = set(names[:len(names)//2])
    weights = np.array(json.loads(Path(a.class_balance).read_text())['weights'], dtype=np.float64)
    if weights.shape != (9,) or not np.isfinite(weights).all() or np.any(weights[1:] <= 0):
        raise ValueError('Invalid class weights')
    weights[0] = 1.
    strengths = [0., .1, .2, .3, .5]
    scales = [torch.tensor(weights ** (-s), device=device, dtype=torch.float32)[None, :, None, None]
              for s in strengths]
    hist = torch.zeros((len(strengths), 2, 81), dtype=torch.int64, device=device)
    for index, (path, mask) in enumerate(pairs, 1):
        with Image.open(path) as im:
            rgb = im.convert('RGB')
        with Image.open(mask) as im:
            y = torch.from_numpy(np.asarray(im, dtype=np.int64).copy()).to(device)
        if y.shape != (rgb.height, rgb.width) or y.min() < 0 or y.max() > 8:
            raise ValueError(f'Invalid mask: {mask}')
        total = probabilities(model, rgb, [512, 640, 768], True, device)
        group = 0 if path.name in calibration else 1
        valid = y != 0
        for k, scale in enumerate(scales):
            pred = (total * scale).argmax(1)[0]
            hist[k, group] += torch.bincount(9*y[valid] + pred[valid], minlength=81)
        if index == 1 or index % 50 == 0:
            print(f'calibration images={index}/{len(pairs)}', flush=True)
    hist = hist.cpu()
    candidates = []
    for index, strength in enumerate(strengths):
        candidates.append({'strength': strength, 'class_scales': (weights ** (-strength)).tolist(),
                           'calibration': metrics_from_hist(hist[index, 0].reshape(9, 9)),
                           'confirmation': metrics_from_hist(hist[index, 1].reshape(9, 9)),
                           'all': metrics_from_hist(hist[index].sum(0).reshape(9, 9))})
    selected, approved = choose_candidate(candidates)
    result = {'selected': selected, 'approved': approved, 'candidates': candidates,
              'calibration_images': sorted(calibration),
              'confirmation_images': sorted(set(names)-calibration),
              'checkpoint': a.checkpoint, 'source': a.source,
              'sizes': [512, 640, 768], 'hflip': True,
              'note': 'Confirmation excludes calibration fitting but reuses the historical validation set; not an official test score.',
              'official_score': None}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'approved': approved, 'strength': selected['strength'],
                      'baseline_miou': candidates[0]['all']['mIoU_present_nonignored'],
                      'selected_miou': selected['all']['mIoU_present_nonignored']}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'source', 'images', 'masks', 'split', 'class-balance', 'out'):
        p.add_argument('--'+name, required=True)
    main(p.parse_args())
