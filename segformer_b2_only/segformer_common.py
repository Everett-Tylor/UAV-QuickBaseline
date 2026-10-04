"""Shared SegFormer dataset, loss and validation helpers for UAV class-ID masks."""
import concurrent.futures
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

LABELS = ['Ignore', 'Background', 'Building', 'Road', 'Water', 'Barren',
          'Vegetation', 'Agricultural', 'Vehicle']

def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')

def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    random.seed(seed); np.random.seed(seed)

def tensor(im):
    return torch.from_numpy(np.asarray(im, dtype=np.uint8).copy().transpose(2, 0, 1)).float().div_(255)

class Pairs(Dataset):
    def __init__(self, names, images, masks, size, augment=False):
        self.names, self.images, self.masks = names, Path(images), Path(masks)
        self.size, self.augment = size, augment

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        name = self.names[i]
        with Image.open(self.images / name) as im: x = im.convert('RGB')
        with Image.open(self.masks / name) as im: y = im.copy()
        x = x.resize((self.size, self.size), Image.Resampling.BILINEAR)
        if self.augment:
            y = y.resize((self.size, self.size), Image.Resampling.NEAREST)
            for tr in (Image.Transpose.FLIP_LEFT_RIGHT, Image.Transpose.FLIP_TOP_BOTTOM):
                if random.random() < .5: x, y = x.transpose(tr), y.transpose(tr)
            k = random.randrange(4)
            if k: x, y = x.rotate(90*k), y.rotate(90*k)
            for cls in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
                x = cls(x).enhance(random.uniform(.8, 1.2))
        return tensor(x), torch.from_numpy(np.asarray(y, dtype=np.int64).copy())

def loader(ds, batch, workers, shuffle=False):
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=workers,
                      pin_memory=True, persistent_workers=workers > 0,
                      worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(20260928))

def prepare(a, split, out):
    names = split['train'] + split['val']
    if len(set(names)) != len(names): raise ValueError('Split contains duplicate names or leakage')
    actual = {p.name for p in Path(a.images).glob('*.png')}
    if set(names) != actual: raise ValueError('Split must cover exactly the supplied training images')
    cache = Path(a.cache)
    (cache/'images').mkdir(parents=True, exist_ok=True)
    (cache/'masks').mkdir(exist_ok=True)
    training = set(split['train'])
    def one(name):
        ip, mp = Path(a.images)/name, Path(a.masks)/name
        with Image.open(ip) as im: x = im.convert('RGB')
        with Image.open(mp) as im:
            if im.mode not in ('L', 'P', 'I', 'I;16'): raise ValueError(f'Not class IDs: {mp}')
            y = np.asarray(im).copy()
        if y.shape != (x.height, x.width) or y.min() < 0 or y.max() > 8:
            raise ValueError(f'Invalid image/mask: {name}')
        counts = np.bincount(y.ravel(), minlength=9) if name in training else np.zeros(9, np.int64)
        # Lossless cache; validation always uses the original masks below.
        x.resize((512,512), Image.Resampling.BILINEAR).save(cache/'images'/name)
        Image.fromarray(y.astype(np.uint8)).resize((512,512), Image.Resampling.NEAREST).save(cache/'masks'/name)
        return counts
    counts = np.zeros(9, np.int64)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for i, c in enumerate(pool.map(one, names), 1):
            counts += c
            if i % 500 == 0: print(f'prepare {i}/{len(names)}', flush=True)
    freq = counts[1:] / counts[1:].sum()
    weights = np.clip(np.sqrt(np.median(freq) / np.maximum(freq, 1e-8)), .5, 3.)
    weights /= weights.mean()
    result = {'pixel_counts_training_only': counts.tolist(), 'weights': [0.] + weights.tolist(),
              'training': len(split['train']), 'validation': len(split['val']),
              'split_sha256': hashlib.sha256(Path(a.split).read_bytes()).hexdigest()}
    save_json(out/'data_audit.json', result)
    return result

def forward(model, x):
    mean = x.new_tensor([.485, .456, .406])[None,:,None,None]
    std = x.new_tensor([.229, .224, .225])[None,:,None,None]
    return model(pixel_values=(x-mean)/std).logits

def loss_fn(logits, y, weights):
    z = F.interpolate(logits.float(), size=y.shape[-2:], mode='bilinear', align_corners=False)
    valid = y != 0
    if not valid.any(): return z.sum()*0
    ce = F.cross_entropy(z, y, weight=weights, ignore_index=0)
    p = z.softmax(1) * valid[:,None]
    truth = F.one_hot(y,9).permute(0,3,1,2).float()*valid[:,None]
    inter = (p*truth).sum((0,2,3))[1:]
    total = (p+truth).sum((0,2,3))[1:]
    present = truth.sum((0,2,3))[1:] > 0
    dice = (2*inter+1e-6)/(total+1e-6)
    return ce + .5*(1-dice[present].mean())

def metrics(hist):
    h = hist.double()
    union = h.sum(0)+h.sum(1)-h.diag()
    iou = h.diag()/union.clamp_min(1)
    return {'mIoU': float(iou[1:].mean()),
            'per_class_IoU': {LABELS[i]: float(iou[i]) for i in range(1,9)},
            'confusion_matrix': hist.tolist()}

@torch.inference_mode()
def evaluate(model, val, flip=False):
    model.eval()
    hist = torch.zeros(81, dtype=torch.int64, device='cuda')
    for x,y in val:
        x,y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        with torch.autocast('cuda', dtype=torch.float16):
            z = forward(model,x).float()
            if flip: z = (z+forward(model,x.flip(-1)).float().flip(-1))*.5
        p = F.interpolate(z, size=y.shape[-2:], mode='bilinear', align_corners=False).argmax(1)
        valid = y != 0
        hist += torch.bincount(9*y[valid]+p[valid], minlength=81)
    return metrics(hist.cpu().reshape(9,9))

