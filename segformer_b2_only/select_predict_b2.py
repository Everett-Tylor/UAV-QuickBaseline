"""Select validation-time transforms and export all SegFormer-B2 test2 masks."""
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

from segformer_common import forward, metrics, save_json, tensor


TRANSFORMS = [(False, ()), (False, (-1,)), (False, (-2,)),
              (False, (-1, -2)), (True, ()), (True, (-1,)),
              (True, (-2,)), (True, (-1, -2))]


@torch.inference_mode()
def transformed_logits(model, image, size, transforms):
    x = tensor(image.resize((size, size), Image.Resampling.BILINEAR))[None].cuda()
    outputs = []
    for transpose, dims in transforms:
        input_tensor = x.transpose(-1, -2) if transpose else x
        if dims:
            input_tensor = input_tensor.flip(dims)
        with torch.autocast('cuda', dtype=torch.float16):
            z = forward(model, input_tensor).float()
        if dims:
            z = z.flip(dims)
        if transpose:
            z = z.transpose(-1, -2)
        outputs.append(F.interpolate(z, size=(image.height, image.width),
                                     mode='bilinear', align_corners=False))
    return outputs


def main():
    parser = argparse.ArgumentParser()
    for name in ('run', 'images', 'masks', 'split', 'test-images'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    run = Path(args.run)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    split = json.loads(Path(args.split).read_text(encoding='utf-8'))
    state = torch.load(run / 'best.pth', map_location='cpu', weights_only=True)
    config = SegformerConfig.from_pretrained(run / 'model_config', local_files_only=True)
    model = SegformerForSemanticSegmentation(config)
    model.load_state_dict(state['model'])
    model.cuda().eval()
    names = ('512_single', '512_hflip', '512_dihedral8', '640_hflip')
    hist = {name: torch.zeros(81, dtype=torch.int64, device='cuda')
            for name in names}
    for index, name in enumerate(split['val'], 1):
        with Image.open(Path(args.images) / name) as source:
            image = source.convert('RGB')
        with Image.open(Path(args.masks) / name) as source:
            label = torch.tensor(np.asarray(source, dtype=np.int64), device='cuda')
        z512 = transformed_logits(model, image, 512, TRANSFORMS)
        z640 = transformed_logits(model, image, 640, TRANSFORMS[:2])
        preds = {'512_single': z512[0].argmax(1)[0],
                 '512_hflip': ((z512[0] + z512[1]) / 2).argmax(1)[0],
                 '512_dihedral8': (sum(z512) / 8).argmax(1)[0],
                 '640_hflip': ((z640[0] + z640[1]) / 2).argmax(1)[0]}
        valid = label != 0
        for mode in names:
            hist[mode] += torch.bincount(9 * label[valid] + preds[mode][valid],
                                         minlength=81)
        if index % 100 == 0:
            print(f'validation {index}/700', flush=True)
    candidates = [{'mode': mode, **metrics(hist[mode].cpu().reshape(9, 9))}
                  for mode in names]
    selected = max(candidates, key=lambda item: item['mIoU'])
    save_json(run / 'validation.json', {
        'best_epoch': state['epoch'], 'candidates': candidates,
        'selected': selected, 'official_score': None})
    print(f'selected {selected["mode"]} mIoU={selected["mIoU"]:.8f}', flush=True)

    test = Path(args.test_images)
    paths = sorted(test.glob('*.png'))
    if {path.name for path in paths} != {f'test2_{i}.png' for i in range(1, 1301)}:
        raise ValueError('Incomplete test2')
    predictions = run / 'predictions'
    predictions.mkdir(exist_ok=False)
    mode = selected['mode']
    size = 640 if mode == '640_hflip' else 512
    transform_count = {'512_single': 1, '512_hflip': 2,
                       '512_dihedral8': 8, '640_hflip': 2}[mode]
    for index, path in enumerate(paths, 1):
        with Image.open(path) as source:
            image = source.convert('RGB')
        outputs = transformed_logits(model, image, size, TRANSFORMS[:transform_count])
        prediction = (sum(outputs) / transform_count).argmax(1)[0]
        Image.fromarray(prediction.byte().cpu().numpy()).save(predictions / path.name)
        if index % 100 == 0:
            print(f'prediction {index}/1300', flush=True)
    archive = run / 'submission_segformer_b2_test2.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in paths:
            z.write(predictions / path.name, path.name)
    with zipfile.ZipFile(archive) as z:
        if len(z.namelist()) != 1300 or z.testzip() is not None:
            raise ValueError('ZIP verification failed')
    with archive.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    save_json(run / 'submission_manifest.json', {
        'file': archive.name, 'count': 1300, 'sha256': digest,
        'mode': mode, 'official_score': None})
    save_json(run / 'status.json', {'stage': 'complete', 'best_epoch': state['epoch'],
                                    'mIoU': selected['mIoU'], 'official_score': None})


if __name__ == '__main__':
    main()
