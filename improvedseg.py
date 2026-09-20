"""Detail-preserving UAV segmentation. Official data only; no pretrained weights."""
import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from quickseg import images, SmallUNet, N_CLASSES, IGNORE, pack


def read_pair(image, mask):
    with Image.open(image) as im:
        rgb = im.convert('RGB')
    with Image.open(mask) as im:
        if im.mode not in ('L', 'P', 'I', 'I;16'):
            raise ValueError(f'Expected class-ID mask, got {im.mode}: {mask}')
        label = np.asarray(im, dtype=np.int64).copy()
    if label.shape != (rgb.height, rgb.width):
        raise ValueError(f'Image/mask dimensions differ: {image}, {mask}')
    if label.min() < 0 or label.max() >= N_CLASSES:
        raise ValueError(f'Unexpected IDs in {mask}: {np.unique(label)}')
    return rgb, Image.fromarray(label.astype(np.uint8))


def tensor_image(rgb):
    return torch.from_numpy(np.asarray(rgb, dtype=np.uint8).copy().transpose(2, 0, 1)).float() / 255.


class CropDataset(Dataset):
    def __init__(self, pairs, size=384, augment=False, repeats=1, sampling='crop'):
        self.pairs, self.size, self.augment, self.repeats = pairs, size, augment, repeats
        self.sampling = sampling

    def __len__(self):
        return len(self.pairs) * self.repeats

    def __getitem__(self, index):
        rgb, mask = read_pair(*self.pairs[index % len(self.pairs)])
        if self.augment:
            # Scale the original image, then crop: do not shrink the whole scene to one tile.
            if self.sampling == 'resize':
                shape = (self.size, self.size)
            else:
                scale = random.uniform(.75, 1.5)
                shape = (max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale)))
            rgb = rgb.resize(shape, Image.Resampling.BILINEAR)
            mask = mask.resize(shape, Image.Resampling.NEAREST)
            pad = (0, 0, max(0, self.size-rgb.width), max(0, self.size-rgb.height))
            rgb, mask = ImageOps.expand(rgb, pad), ImageOps.expand(mask, pad, fill=IGNORE)
            # Bounded retry prevents spending training steps on wholly ignored tiles.
            for _ in range(10):
                left = random.randint(0, rgb.width-self.size)
                top = random.randint(0, rgb.height-self.size)
                box = (left, top, left+self.size, top+self.size)
                crop_mask = mask.crop(box)
                if np.any(np.asarray(crop_mask) != IGNORE):
                    break
            rgb, mask = rgb.crop(box), crop_mask
            for transform in (Image.Transpose.FLIP_LEFT_RIGHT, Image.Transpose.FLIP_TOP_BOTTOM):
                if random.random() < .5:
                    rgb, mask = rgb.transpose(transform), mask.transpose(transform)
            rotation = random.choice([None, Image.Transpose.ROTATE_90,
                                      Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270])
            if rotation is not None:
                rgb, mask = rgb.transpose(rotation), mask.transpose(rotation)
            for enhancer in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
                rgb = enhancer(rgb).enhance(random.uniform(.85, 1.15))
        return tensor_image(rgb), torch.from_numpy(np.asarray(mask, dtype=np.int64).copy())


class DilatedContext(nn.Module):
    """Residual multi-scale context at 1/8 resolution, keeping fine-resolution skip paths."""
    def __init__(self, channels):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv2d(channels, channels, 3, padding=d, dilation=d,
                                    groups=channels, bias=False),
                          nn.Conv2d(channels, channels // 4, 1, bias=False),
                          nn.BatchNorm2d(channels // 4), nn.ReLU(inplace=True))
            for d in (1, 2, 4)
        ])
        self.project = nn.Sequential(nn.Conv2d(3*(channels // 4), channels, 1, bias=False),
                                     nn.BatchNorm2d(channels))

    def forward(self, x):
        return F.relu(x + self.project(torch.cat([b(x) for b in self.branches], dim=1)))


def make_model(width, architecture='context'):
    if architecture not in ('unet', 'context'):
        raise ValueError(f'Unknown architecture: {architecture}')
    model = SmallUNet(width)
    if architecture == 'context':
        model.b = nn.Sequential(model.b, DilatedContext(width * 8))
    return model


def segmentation_loss(logits, target, dice_weight=.5, class_weights=None):
    # Accumulate in FP32 under AMP; ignored pixels contribute to neither term.
    logits = logits.float()
    valid = target != IGNORE
    if not valid.any():
        return logits.sum() * 0
    ce = F.cross_entropy(logits, target, ignore_index=IGNORE, weight=class_weights)
    if dice_weight == 0:
        return ce
    probabilities = logits.softmax(1) * valid.unsqueeze(1)
    truth = F.one_hot(target, N_CLASSES).permute(0, 3, 1, 2).float() * valid.unsqueeze(1)
    intersection = (probabilities * truth).sum((0, 2, 3))
    denominator = probabilities.sum((0, 2, 3)) + truth.sum((0, 2, 3))
    present = truth.sum((0, 2, 3))[1:] > 0
    dice = (2 * intersection[1:] + 1e-6) / (denominator[1:] + 1e-6)
    return ce + dice_weight * (1 - dice[present].mean())


def starts(length, tile, stride):
    if length <= tile:
        return [0]
    positions = list(range(0, length-tile+1, stride))
    if positions[-1] != length-tile:
        positions.append(length-tile)
    return positions


@torch.inference_mode()
def predict_logits(model, x, device, size, mode='tile', overlap=.25):
    """One full-resolution CPU image -> blended CPU logits; bounded GPU tile memory."""
    if x.ndim != 4 or x.shape[0] != 1:
        raise ValueError('Prediction expects a batch of one image')
    if size < 16 or size % 8 or not 0 <= overlap < 1:
        raise ValueError('Tile size must be >=16 and divisible by 8; overlap must be in [0,1)')
    height, width = x.shape[-2:]
    if mode == 'resize':
        # Match legacy quickseg.infer's PIL antialiasing and uint8 rounding exactly.
        rgb = Image.fromarray((x[0].cpu().permute(1, 2, 0).numpy()*255).round().clip(0, 255).astype(np.uint8))
        small = tensor_image(rgb.resize((size, size), Image.Resampling.BILINEAR)).unsqueeze(0)
        result = model(small.to(device)).float().cpu()
        return F.interpolate(result, size=(height, width), mode='bilinear', align_corners=False)
    if mode != 'tile':
        raise ValueError(f'Unknown prediction mode: {mode}')
    stride = max(1, round(size * (1-overlap)))
    padded = F.pad(x, (0, max(0, size-width), 0, max(0, size-height)))
    h, w = padded.shape[-2:]
    total = torch.zeros((1, N_CLASSES, h, w))
    count = torch.zeros((1, 1, h, w))
    # Positive edge weight avoids zero divisions at image borders.
    axis = torch.hann_window(size, periodic=False).clamp_min(.1)
    weight = (axis[:, None] * axis[None, :])[None, None]
    for top in starts(h, size, stride):
        for left in starts(w, size, stride):
            tile = padded[:, :, top:top+size, left:left+size].to(device)
            result = model(tile).float().cpu()
            total[:, :, top:top+size, left:left+size] += result * weight
            count[:, :, top:top+size, left:left+size] += weight
    return (total / count)[:, :, :height, :width]


def metrics_from_hist(hist):
    intersection = hist.diag().double()
    union = hist.sum(0) + hist.sum(1) - hist.diag()
    iou = intersection / union.clamp_min(1)
    present = union[1:] > 0
    return {
        'mIoU_present_nonignored': float(iou[1:][present].mean()) if present.any() else None,
        'mIoU_all_8_absent_zero': float(iou[1:].mean()) if present.any() else None,
        'per_class_IoU': {str(k): float(iou[k]) if union[k] > 0 else None for k in range(1, N_CLASSES)},
        'confusion_matrix': hist.tolist(),
    }


def evaluate(model, loader, device, size, mode='tile', overlap=.25):
    was_training = model.training
    model.eval()
    hist = torch.zeros((N_CLASSES, N_CLASSES), dtype=torch.int64)
    for index, (x, y) in enumerate(loader, 1):
        prediction = predict_logits(model, x, device, size, mode, overlap).argmax(1)
        valid = y != IGNORE
        hist += torch.bincount((y[valid]*N_CLASSES + prediction[valid]).flatten(),
                               minlength=N_CLASSES*N_CLASSES).reshape(N_CLASSES, N_CLASSES)
        if index % 50 == 0:
            print(f'validation={index}/{len(loader)}', flush=True)
    model.train(was_training)
    return metrics_from_hist(hist)


def paired_paths(args):
    masks = {p.stem: p for p in images(args.masks)}
    pairs = []
    for image in images(args.images):
        if image.stem not in masks:
            raise FileNotFoundError(f'Missing label for {image.name}')
        pairs.append((image, masks[image.stem]))
    return pairs


def split_pairs(pairs, split_file, fraction, seed):
    names = [im.name for im, _ in pairs]
    if split_file:
        split = json.loads(Path(split_file).read_text(encoding='utf-8'))
        train_names, val_names = split['train'], split['val']
        if not train_names or not val_names:
            raise ValueError('A nonempty training and validation split is required')
        if len(set(train_names + val_names)) != len(train_names + val_names):
            raise ValueError('Duplicate names or training/validation overlap in split')
        if set(train_names + val_names) != set(names):
            raise ValueError('Split must cover exactly the current image filenames')
    else:
        if len(pairs) < 2:
            raise ValueError('At least two paired images are required for held-out validation')
        random.Random(seed).shuffle(names)
        nval = min(len(names)-1, max(1, round(len(names)*fraction)))
        val_names, train_names = names[:nval], names[nval:]
        split = {'train': train_names, 'val': val_names}
        print('Random image split: use --split with scene-separated names for reliable validation.', flush=True)
    lookup = {im.name: (im, mask) for im, mask in pairs}
    return [lookup[n] for n in train_names], [lookup[n] for n in val_names], split


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def lr_factor(step, total, warmup):
    if step < warmup:
        return (step+1) / max(1, warmup)
    progress = min(1., max(0., (step-warmup) / max(1, total-warmup-1)))
    return .01 + .99 * .5 * (1 + math.cos(math.pi * progress))


def device_for(args):
    return 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'


def training_class_weights(pairs):
    counts = np.zeros(N_CLASSES, dtype=np.int64)
    for _, path in pairs:
        with Image.open(path) as im:
            if im.mode not in ('L', 'P', 'I', 'I;16'):
                raise ValueError(f'Expected class-ID mask: {path}')
            target = np.asarray(im, dtype=np.int64)
        if target.min() < 0 or target.max() >= N_CLASSES:
            raise ValueError(f'Unexpected mask IDs: {path}')
        counts += np.bincount(target.ravel(), minlength=N_CLASSES)
    positive = counts[1:][counts[1:] > 0]
    if not len(positive):
        raise ValueError('Training masks contain no labeled pixels')
    weights = np.sqrt(positive.mean() / np.maximum(counts, 1)).clip(.25, 4).astype(np.float32)
    weights[0] = 0
    return torch.from_numpy(weights), counts.tolist()


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_pairs, val_pairs, split = split_pairs(paired_paths(args), args.split, args.val_fraction, args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'last.pth').exists() or (out/'history.jsonl').exists():
        raise ValueError('Output already contains a run; use a new --output for a reproducible comparison')
    (out/'split.json').write_text(json.dumps(split, ensure_ascii=False, indent=2), encoding='utf-8')
    (out/'config.json').write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding='utf-8')
    device = device_for(args)
    sampling = getattr(args, 'sampling', 'crop')
    inference_mode = 'resize' if sampling == 'resize' else 'tile'
    loader = DataLoader(CropDataset(train_pairs, args.size, True, args.crops_per_image, sampling),
                        batch_size=args.batch, shuffle=True, num_workers=args.workers,
                        pin_memory=device == 'cuda', worker_init_fn=seed_worker,
                        generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(CropDataset(val_pairs), batch_size=1, num_workers=args.workers)
    class_weights = None
    if getattr(args, 'class_balance', 'none') == 'sqrt':
        class_weights, counts = training_class_weights(train_pairs)
        (out/'class_balance.json').write_text(json.dumps({'train_pixel_counts': counts,
                                                          'weights': class_weights.tolist()}, indent=2), encoding='utf-8')
        print(f'Train-only class weights: {class_weights.tolist()}', flush=True)
        class_weights = class_weights.to(device)
    model = make_model(args.width, args.architecture).to(device)
    if args.init_weights:
        ck = torch.load(args.init_weights, map_location='cpu', weights_only=True)
        if ck.get('architecture', 'unet') != args.architecture or ck['width'] != args.width:
            raise ValueError('--init-weights must match --architecture and --width')
        if ck.get('classes', N_CLASSES) != N_CLASSES or ck.get('ignore_index', IGNORE) != IGNORE:
            raise ValueError('Initialization weights use a different label convention')
        if 'split' in ck and set(ck['split']['train']) != set(split['train']):
            raise ValueError('Initialization checkpoint training split differs from this run')
        model.load_state_dict(ck['model'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=device == 'cuda')
    total_steps = args.epochs * len(loader)
    warmup = min(args.warmup_epochs * len(loader), max(0, total_steps-1))
    best = -1.
    for epoch in range(args.epochs):
        model.train()
        total_loss, updates = 0., 0
        for step, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            if not (y != IGNORE).any():
                continue  # Avoid AdamW weight decay updates on completely ignored batches.
            lr = args.lr * lr_factor(epoch*len(loader)+step, total_steps, warmup)
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, enabled=device == 'cuda'):
                loss = segmentation_loss(model(x), y, args.dice_weight, class_weights)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach())
            updates += 1
            if (step+1) % 30 == 0:
                print(f'epoch={epoch+1} step={step+1}/{len(loader)} loss={total_loss/updates:.4f}', flush=True)
        if updates == 0:
            raise ValueError('No labeled training pixels sampled; check masks and data split')
        metrics = evaluate(model, val_loader, device, args.size, mode=inference_mode, overlap=args.overlap)
        value = metrics['mIoU_present_nonignored']
        if value is None:
            raise ValueError('Validation has no nonignored pixels; cannot select best mIoU')
        record = {'epoch': epoch+1, 'loss': total_loss/updates, 'lr': lr, 'metrics': metrics}
        with (out/'history.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')
        checkpoint = {'model': model.state_dict(), 'width': args.width, 'size': args.size,
                      'architecture': args.architecture, 'inference_mode': inference_mode, 'overlap': args.overlap,
                      'epoch': epoch+1, 'metrics': metrics, 'classes': N_CLASSES, 'ignore_index': IGNORE,
                      'split': split, 'config': vars(args)}
        torch.save(checkpoint, out/'last.pth')
        if value > best:
            best = value
            torch.save(checkpoint, out/'best.pth')
        print(f'epoch={epoch+1} loss={total_loss/updates:.4f} mIoU={value:.6f} best={best:.6f}', flush=True)
    print(f'Training complete: {out / "best.pth"}')


def load_checkpoint(args):
    ck = torch.load(args.weights, map_location='cpu', weights_only=True)
    if ck.get('classes', N_CLASSES) != N_CLASSES or ck.get('ignore_index', IGNORE) != IGNORE:
        raise ValueError('Checkpoint label convention does not match this project')
    model = make_model(ck['width'], ck.get('architecture', 'unet')).to(device_for(args))
    model.load_state_dict(ck['model'])
    return model.eval(), ck


def infer(args):
    model, ck = load_checkpoint(args)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    paths = images(args.images)
    for index, path in enumerate(paths, 1):
        with Image.open(path) as im:
            if im.size != (1024, 1024):
                raise ValueError(f'Official test image must be 1024x1024: {path}')
            x = tensor_image(im.convert('RGB')).unsqueeze(0)
        mode = getattr(args, 'mode', 'auto')
        mode = ck.get('inference_mode', 'resize') if mode == 'auto' else mode
        logits = predict_logits(model, x, device_for(args), ck['size'], mode, ck.get('overlap', .25))
        Image.fromarray(logits.argmax(1)[0].byte().numpy()).save(out/(path.stem+'.png'))
        if index % 100 == 0:
            print(f'Predicted {index}/{len(paths)}', flush=True)
    print(f'Predictions saved: {out}')


def validate(args):
    model, ck = load_checkpoint(args)
    _, val_pairs, _ = split_pairs(paired_paths(args), args.split, .1, 42)
    loader = DataLoader(CropDataset(val_pairs), batch_size=1, num_workers=args.workers)
    mode = getattr(args, 'mode', 'auto')
    mode = ck.get('inference_mode', 'resize') if mode == 'auto' else mode
    metrics = evaluate(model, loader, device_for(args), ck['size'], mode, ck.get('overlap', .25))
    result = {'weights': args.weights, 'checkpoint_epoch': ck.get('epoch'),
              'split': args.split, 'resolution': 'native',
              'inference_mode': mode, 'metrics': metrics}
    print(json.dumps(result, indent=2))
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(result, indent=2), encoding='utf-8')


def main():
    # Large host thread pools make per-tile CPU blending disproportionately slow.
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='cmd', required=True)
    tr = sub.add_parser('train')
    tr.add_argument('--images', required=True)
    tr.add_argument('--masks', required=True)
    tr.add_argument('--output', default='runs/detail')
    tr.add_argument('--split', help='JSON with disjoint train/val image filenames; supports quickseg split.json')
    tr.add_argument('--init-weights', help='Fine-tune your own checkpoint; reuse its original --split')
    tr.add_argument('--epochs', type=int, default=60)
    tr.add_argument('--size', type=int, default=384)
    tr.add_argument('--batch', type=int, default=4)
    tr.add_argument('--width', type=int, default=24)
    tr.add_argument('--architecture', choices=['unet', 'context'], default='context')
    tr.add_argument('--sampling', choices=['crop', 'resize'], default='crop',
                    help='Use resize to preserve the image scale of legacy checkpoints')
    tr.add_argument('--lr', type=float, default=.001)
    tr.add_argument('--warmup-epochs', type=int, default=3)
    tr.add_argument('--dice-weight', type=float, default=.5)
    tr.add_argument('--class-balance', choices=['none', 'sqrt'], default='none',
                    help='Bounded inverse-square-root weights from training masks only')
    tr.add_argument('--crops-per-image', type=int, default=2)
    tr.add_argument('--overlap', type=float, default=.25)
    tr.add_argument('--val-fraction', type=float, default=.1)
    tr.add_argument('--seed', type=int, default=42)
    for name in ('infer', 'eval'):
        p = sub.add_parser(name)
        p.add_argument('--images', required=True)
        p.add_argument('--weights', required=True)
        p.add_argument('--mode', choices=['auto', 'resize', 'tile'], default='auto',
                       help='Override inference only for controlled validation comparisons')
        if name == 'infer':
            p.add_argument('--output', default='predictions_detail')
        else:
            p.add_argument('--masks', required=True)
            p.add_argument('--split', required=True)
            p.add_argument('--report')
        p.add_argument('--cpu', action='store_true')
        p.add_argument('--workers', type=int, default=2)
    tr.add_argument('--cpu', action='store_true')
    tr.add_argument('--workers', type=int, default=2)
    pk = sub.add_parser('pack')
    pk.add_argument('--images', required=True)
    pk.add_argument('--predictions', required=True)
    pk.add_argument('--zip', default='submission.zip')
    args = parser.parse_args()
    if args.cmd == 'train':
        if args.size < 16 or args.size % 8:
            parser.error('--size must be >=16 and divisible by 8')
        if min(args.epochs, args.batch, args.width, args.crops_per_image) < 1:
            parser.error('epochs, batch, width and crops-per-image must be positive')
        if not 0 < args.val_fraction < 1 or not 0 <= args.overlap < 1:
            parser.error('val-fraction must be in (0,1); overlap must be in [0,1)')
        if args.lr <= 0 or args.dice_weight < 0 or args.warmup_epochs < 0:
            parser.error('lr must be positive; dice-weight and warmup-epochs must be nonnegative')
    if hasattr(args, 'workers') and args.workers < 0:
        parser.error('--workers must be nonnegative')
    {'train': train, 'infer': infer, 'eval': validate, 'pack': pack}[args.cmd](args)


if __name__ == '__main__':
    main()
