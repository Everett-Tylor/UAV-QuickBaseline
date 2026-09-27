"""Evaluate or export predictions using exactly one SegFormer-B2 checkpoint."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from b2_model import Segmenter
from improvedseg import tensor_image, metrics_from_hist, paired_paths, split_pairs
from quickseg import images


@torch.inference_mode()
def probabilities(model, rgb, sizes, hflip, device):
    shape = (rgb.height, rgb.width)
    total = None
    for size in sizes:
        x = tensor_image(rgb.resize((size, size), Image.Resampling.BILINEAR))[None].to(device)
        for flip in ([False, True] if hflip else [False]):
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == 'cuda' and torch.cuda.is_bf16_supported()):
                logits = model(x.flip(-1) if flip else x)
            if not torch.isfinite(logits).all():
                raise RuntimeError('Non-finite prediction logits')
            if flip:
                logits = logits.flip(-1)
            probs = F.interpolate(logits.float(), size=shape, mode='bilinear', align_corners=False).softmax(1)
            total = probs if total is None else total + probs
    return total


def validate_archive(archive, paths):
    expected = {p.stem + '.png' for p in paths}
    if not paths or len(expected) != len(paths):
        raise ValueError('Empty inputs or duplicate image stems')
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if len(names) != len(expected) or set(names) != expected or z.testzip() is not None:
            raise ValueError('Archive filenames, count or CRC mismatch')
        for name in names:
            with z.open(name) as stream, Image.open(stream) as im:
                values = np.asarray(im)
                if im.mode != 'L' or im.size != (1024, 1024) or values.max() > 8:
                    raise ValueError(f'Invalid prediction: {name}')
    return {'images': len(paths), 'sha256': hashlib.sha256(Path(archive).read_bytes()).hexdigest()}


@torch.inference_mode()
def run(a):
    if not a.sizes or any(s < 32 or s % 32 for s in a.sizes):
        raise ValueError('Inference sizes must be positive multiples of 32')
    torch.set_num_threads(4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    state = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    model = Segmenter(a.source, config_only=True).to(device).eval()
    model.load_state_dict(state['model'], strict=True)
    if model.net.config.num_labels != 9:
        raise ValueError('Expected nine output channels')
    if a.masks:
        if not a.split:
            raise ValueError('Validation requires the original split')
        _, pairs, split = split_pairs(paired_paths(a), a.split, .1, 2026)
        if split != state['split']:
            raise ValueError('Validation split differs from checkpoint training split')
    else:
        pairs = [(p, None) for p in images(a.images)]
    if not pairs or len({p.stem for p, _ in pairs}) != len(pairs):
        raise ValueError('Empty inputs or duplicate image stems')
    out = Path(a.out)
    if not a.masks:
        if out.exists() and any(out.iterdir()):
            raise ValueError('Use an empty prediction directory')
        if out.with_suffix('.zip').exists():
            raise ValueError('Prediction archive already exists')
        out.mkdir(parents=True, exist_ok=True)
    hist = torch.zeros(81, dtype=torch.int64)
    for index, (path, mask) in enumerate(pairs, 1):
        with Image.open(path) as im:
            rgb = im.convert('RGB')
        shape = (rgb.height, rgb.width)
        if not a.masks and shape != (1024, 1024):
            raise ValueError(f'Expected 1024x1024 official test image: {path}')
        total = probabilities(model, rgb, a.sizes, a.hflip, device)
        if getattr(a, 'class_scales', None):
            scales = torch.tensor(a.class_scales, device=device)
            if scales.shape != (9,) or not torch.isfinite(scales).all() or not (scales > 0).all():
                raise ValueError('Expected nine positive finite class scales')
            total = total * scales[None, :, None, None]
        pred = total.argmax(1)[0].cpu()
        if mask is not None:
            with Image.open(mask) as im:
                if im.mode not in ('L', 'P', 'I', 'I;16'):
                    raise ValueError(f'Expected class-ID mask: {mask}')
                y = torch.from_numpy(np.asarray(im, dtype=np.int64).copy())
            if y.shape != pred.shape or y.min() < 0 or y.max() > 8:
                raise ValueError(f'Invalid validation mask: {mask}')
            valid = y != 0
            hist += torch.bincount(9 * y[valid] + pred[valid], minlength=81)
        else:
            Image.fromarray(pred.numpy().astype(np.uint8)).save(out / (path.stem + '.png'))
        if index == 1 or index % 100 == 0:
            print(f'images={index}/{len(pairs)}', flush=True)
    report = {'checkpoint': str(a.checkpoint), 'single_checkpoint': True,
              'sizes': a.sizes, 'hflip': a.hflip, 'official_score': None,
              'class_scales': getattr(a, 'class_scales', None)}
    if a.masks:
        report.update(metrics_from_hist(hist.reshape(9, 9)))
        if report['mIoU_present_nonignored'] is None:
            raise ValueError('Validation has no labeled pixels')
        report['images'] = len(pairs)
        report_path = out
    else:
        archive = out.with_suffix('.zip')
        temporary = archive.with_suffix('.zip.tmp')
        with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as z:
            for path, _ in pairs:
                name = path.stem + '.png'
                z.write(out / name, name)
        report.update(validate_archive(temporary, [p for p, _ in pairs]))
        temporary.replace(archive)
        report['archive'] = str(archive)
        report_path = out.with_suffix('.manifest.json')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'source', 'images', 'out'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--sizes', type=int, nargs='+', default=[512, 640, 768])
    p.add_argument('--hflip', action='store_true')
    p.add_argument('--masks')
    p.add_argument('--split')
    p.add_argument('--class-scales', type=float, nargs=9)
    run(p.parse_args())
