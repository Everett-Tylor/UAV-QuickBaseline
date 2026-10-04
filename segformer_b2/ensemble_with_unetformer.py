"""Evaluate whether the trained B2 adds value to the selected UNetFormer."""
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
from transformers import SegformerConfig, SegformerForSemanticSegmentation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'unetformer_round3'))
from round3 import make_model as make_unetformer
from unetformer_pipeline import forward as unet_forward
from segformer_common import metrics, save_json, tensor
from select_predict_b2 import TRANSFORMS, transformed_logits


@torch.inference_mode()
def unet_logits(model, image):
    x = tensor(image.resize((512, 512), Image.Resampling.BILINEAR))[None].cuda()
    total = None
    for transpose, dims in TRANSFORMS:
        input_tensor = x.transpose(-1, -2) if transpose else x
        if dims:
            input_tensor = input_tensor.flip(dims)
        with torch.autocast('cuda', dtype=torch.float16):
            z = unet_forward(model, input_tensor).float()
        if dims:
            z = z.flip(dims)
        if transpose:
            z = z.transpose(-1, -2)
        total = z if total is None else total + z
    return F.interpolate(total / 8, size=(image.height, image.width),
                         mode='bilinear', align_corners=False)


def b2_logits(model, image, mode):
    size = 640 if mode == '640_hflip' else 512
    count = {'512_single': 1, '512_hflip': 2,
             '512_dihedral8': 8, '640_hflip': 2}[mode]
    return sum(transformed_logits(model, image, size, TRANSFORMS[:count])) / count


def main():
    parser = argparse.ArgumentParser()
    for name in ('b2-run', 'unet-checkpoint', 'images', 'masks', 'split',
                 'test-images', 'out'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--baseline', type=float, required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    split = json.loads(Path(args.split).read_text(encoding='utf-8'))
    b2_run = Path(args.b2_run)
    b2_mode = json.loads((b2_run / 'validation.json').read_text(encoding='utf-8'))['selected']['mode']
    b2_config = SegformerConfig.from_pretrained(b2_run / 'model_config',
                                                 local_files_only=True)
    b2 = SegformerForSemanticSegmentation(b2_config)
    b2.load_state_dict(torch.load(b2_run / 'best.pth', map_location='cpu',
                                  weights_only=True)['model'])
    b2.cuda().eval()
    unet = make_unetformer()
    unet.load_state_dict(torch.load(args.unet_checkpoint, map_location='cpu',
                                    weights_only=True)['model'])
    unet.cuda().eval()
    fractions = (.1, .2, .35, .5)
    hist = {fraction: torch.zeros(81, dtype=torch.int64, device='cuda')
            for fraction in fractions}
    for index, name in enumerate(split['val'], 1):
        with Image.open(Path(args.images) / name) as source:
            image = source.convert('RGB')
        with Image.open(Path(args.masks) / name) as source:
            label = torch.tensor(np.asarray(source, dtype=np.int64), device='cuda')
        old = unet_logits(unet, image).softmax(1)
        new = b2_logits(b2, image, b2_mode).softmax(1)
        valid = label != 0
        for fraction in fractions:
            prediction = ((1 - fraction) * old + fraction * new).argmax(1)[0]
            hist[fraction] += torch.bincount(9 * label[valid] + prediction[valid],
                                             minlength=81)
        if index % 100 == 0:
            print(f'validation {index}/700', flush=True)
    candidates = [{'b2_fraction': fraction,
                   **metrics(hist[fraction].cpu().reshape(9, 9))}
                  for fraction in fractions]
    selected = max(candidates, key=lambda item: item['mIoU'])
    improved = selected['mIoU'] > args.baseline
    save_json(out / 'selection.json', {
        'b2_mode': b2_mode, 'baseline_mIoU': args.baseline,
        'candidates': candidates, 'selected': selected, 'improved': improved,
        'delta_percentage_points': 100 * (selected['mIoU'] - args.baseline),
        'official_score': None})
    print(f'ensemble mIoU={selected["mIoU"]:.8f} improved={improved}', flush=True)
    if not improved:
        return
    test = Path(args.test_images)
    paths = sorted(test.glob('*.png'))
    if {path.name for path in paths} != {f'test2_{i}.png' for i in range(1, 1301)}:
        raise ValueError('Incomplete test2')
    predictions = out / 'predictions'
    predictions.mkdir()
    fraction = selected['b2_fraction']
    for index, path in enumerate(paths, 1):
        with Image.open(path) as source:
            image = source.convert('RGB')
        old = unet_logits(unet, image).softmax(1)
        new = b2_logits(b2, image, b2_mode).softmax(1)
        prediction = ((1 - fraction) * old + fraction * new).argmax(1)[0]
        Image.fromarray(prediction.byte().cpu().numpy()).save(predictions / path.name)
        if index % 100 == 0:
            print(f'prediction {index}/1300', flush=True)
    archive = out / 'submission_b2_unetformer_ensemble_test2.zip'
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
        'b2_fraction': fraction, 'b2_mode': b2_mode})


if __name__ == '__main__':
    main()
