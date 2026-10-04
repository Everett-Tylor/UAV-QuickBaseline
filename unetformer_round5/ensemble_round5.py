"""Validate and export a two-checkpoint UNetFormer logit ensemble."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'unetformer_round3'))
from round3 import make_model, save_json
from unetformer_pipeline import forward, metrics, tensor


@torch.inference_mode()
def logits(model, image):
    x = tensor(image.resize((512, 512), Image.Resampling.BILINEAR))[None].cuda()
    total = None
    for transpose in (False, True):
        base = x.transpose(-1, -2) if transpose else x
        for dims in ([], [-1], [-2], [-1, -2]):
            input_tensor = base.flip(dims) if dims else base
            with torch.autocast('cuda', dtype=torch.float16):
                output = forward(model, input_tensor).float()
            if dims:
                output = output.flip(dims)
            if transpose:
                output = output.transpose(-1, -2)
            total = output if total is None else total + output
    return F.interpolate(total / 8, size=(image.height, image.width),
                         mode='bilinear', align_corners=False)


def models(paths):
    result = []
    for path in paths:
        model = make_model()
        model.load_state_dict(torch.load(path, map_location='cpu',
                                         weights_only=True)['model'])
        result.append(model.cuda().eval())
    return result


def main():
    parser = argparse.ArgumentParser()
    for name in ('older', 'newer', 'images', 'masks', 'split',
                 'test-images', 'out'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--baseline', type=float, required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    split = json.loads(Path(args.split).read_text(encoding='utf-8'))
    networks = models((args.older, args.newer))
    fractions = (.25, .5, .75)
    hist = {fraction: torch.zeros(81, dtype=torch.int64, device='cuda')
            for fraction in fractions}
    for index, name in enumerate(split['val'], 1):
        with Image.open(Path(args.images) / name) as source:
            image = source.convert('RGB')
        with Image.open(Path(args.masks) / name) as source:
            label = torch.tensor(np.asarray(source, dtype=np.int64), device='cuda')
        old, new = (logits(model, image) for model in networks)
        valid = label != 0
        for fraction in fractions:
            prediction = (fraction * old + (1 - fraction) * new).argmax(1)[0]
            hist[fraction] += torch.bincount(9 * label[valid] + prediction[valid],
                                             minlength=81)
        if index % 100 == 0:
            print(f'validation {index}/700', flush=True)
    results = [{'old_fraction': fraction,
                **metrics(hist[fraction].cpu().reshape(9, 9))}
               for fraction in fractions]
    best = max(results, key=lambda item: item['mIoU'])
    save_json(out / 'selection.json', {
        'baseline_mIoU': args.baseline, 'candidates': results, 'best': best,
        'improved': best['mIoU'] > args.baseline,
        'delta_percentage_points': 100 * (best['mIoU'] - args.baseline),
        'official_score': None})
    print(f'best old_fraction={best["old_fraction"]} '
          f'mIoU={best["mIoU"]:.8f}', flush=True)
    if best['mIoU'] <= args.baseline:
        return

    test = Path(args.test_images)
    paths = sorted(test.glob('*.png'))
    if {path.name for path in paths} != {f'test2_{i}.png' for i in range(1, 1301)}:
        raise ValueError('Incomplete test2')
    predictions = out / 'predictions'
    predictions.mkdir()
    fraction = best['old_fraction']
    for index, path in enumerate(paths, 1):
        with Image.open(path) as source:
            image = source.convert('RGB')
        old, new = (logits(model, image) for model in networks)
        prediction = (fraction * old + (1 - fraction) * new).argmax(1)[0]
        Image.fromarray(prediction.byte().cpu().numpy()).save(predictions / path.name)
        if index % 100 == 0:
            print(f'prediction {index}/1300', flush=True)
    archive = out / 'submission_unetformer_round5_test2.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in paths:
            z.write(predictions / path.name, path.name)
    with zipfile.ZipFile(archive) as z:
        if len(z.namelist()) != 1300 or z.testzip() is not None:
            raise ValueError('ZIP verification failed')
    with archive.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    save_json(out / 'submission_manifest.json', {
        'count': 1300, 'file': archive.name, 'sha256': digest,
        'old_fraction': fraction, 'inference': '512 dihedral8 per checkpoint'})


if __name__ == '__main__':
    main()

