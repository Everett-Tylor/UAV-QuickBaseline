"""Contest-only minimal segmentation baseline. No external data or pretrained weights."""
import argparse
import json
import random
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

EXT = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
N_CLASSES = 9
IGNORE = 0


def images(folder):
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f'Folder missing: {folder}')
    paths = sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in EXT)
    if not paths:
        raise RuntimeError(f'No images in {folder}')
    if len({p.stem for p in paths}) != len(paths):
        raise ValueError('Duplicate image stems: cannot map labels or submission names uniquely')
    return paths


class PairedDataset(Dataset):
    def __init__(self, imgdir, maskdir, size=384, subset=None, augment=False):
        self.pairs = []
        mask_index = {p.stem: p for p in images(maskdir)}
        for im in images(imgdir):
            if im.stem not in mask_index:
                raise FileNotFoundError(f'Missing mask for {im.name}')
            self.pairs.append((im, mask_index[im.stem]))
        if subset is not None:
            self.pairs = [self.pairs[i] for i in subset]
        self.size = size
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        im, mask = self.pairs[idx]
        with Image.open(im) as x:
            x = x.convert('RGB').resize((self.size, self.size), Image.Resampling.BILINEAR)
            rgb = np.asarray(x, dtype=np.uint8).copy()
        with Image.open(mask) as y:
            if y.mode not in ('L', 'I', 'I;16'):
                raise ValueError(f'Mask must be single-channel ID image: {mask}; got {y.mode}')
            y = y.resize((self.size, self.size), Image.Resampling.NEAREST)
            label = np.asarray(y, dtype=np.int64).copy()
        if label.min() < 0 or label.max() >= N_CLASSES:
            raise ValueError(f'Unexpected mask IDs in {mask}: {np.unique(label)}')
        if self.augment and random.random() < .5:
            rgb = np.flip(rgb, axis=1).copy()
            label = np.flip(label, axis=1).copy()
        return torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float() / 255., torch.from_numpy(label.copy()).long()


class ConvBlock(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(inc, outc, 3, padding=1, bias=False),
                                 nn.BatchNorm2d(outc), nn.ReLU(inplace=True),
                                 nn.Conv2d(outc, outc, 3, padding=1, bias=False),
                                 nn.BatchNorm2d(outc), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class SmallUNet(nn.Module):
    def __init__(self, width=24):
        super().__init__()
        w = width
        self.e1 = ConvBlock(3, w)
        self.e2 = ConvBlock(w, 2*w)
        self.e3 = ConvBlock(2*w, 4*w)
        self.b = ConvBlock(4*w, 8*w)
        self.d3 = ConvBlock(12*w, 4*w)
        self.d2 = ConvBlock(6*w, 2*w)
        self.d1 = ConvBlock(3*w, w)
        self.head = nn.Conv2d(w, N_CLASSES, 1)

    def up(self, x, skip, layer):
        return layer(torch.cat((F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False), skip), dim=1))

    def forward(self, x):
        a = self.e1(x)
        b = self.e2(F.max_pool2d(a, 2))
        c = self.e3(F.max_pool2d(b, 2))
        d = self.b(F.max_pool2d(c, 2))
        d = self.up(d, c, self.d3)
        d = self.up(d, b, self.d2)
        d = self.up(d, a, self.d1)
        return self.head(d)


def score(model, loader, device):
    model.eval()
    hist = torch.zeros((N_CLASSES, N_CLASSES), dtype=torch.int64)
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).argmax(1).cpu()
            good = (y != IGNORE) & (y >= 0) & (y < N_CLASSES)
            hist += torch.bincount((y[good] * N_CLASSES + pred[good]).reshape(-1), minlength=N_CLASSES*N_CLASSES).reshape(N_CLASSES, N_CLASSES)
    inter = hist.diag().float()
    union = hist.sum(0) + hist.sum(1) - hist.diag()
    iou = inter / union.clamp(min=1)
    valid = union[1:] > 0
    result = {'mIoU_present_nonignored': float(iou[1:][valid].mean()) if valid.any() else None,
              'per_class_IoU': {str(k): (float(iou[k]) if union[k] > 0 else None) for k in range(1, N_CLASSES)}}
    model.train()
    return result


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset = PairedDataset(args.images, args.masks, args.size)
    n = len(dataset)
    ids = list(range(n))
    random.shuffle(ids)
    nval = max(1, round(n * args.val_fraction)) if n > 1 else 0
    val_ids, train_ids = ids[:nval], ids[nval:]
    train_data = PairedDataset(args.images, args.masks, args.size, train_ids, augment=True)
    val_data = PairedDataset(args.images, args.masks, args.size, val_ids) if val_ids else None
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    if device == 'cpu':
        print('WARNING: no CUDA GPU detected; full dataset training may be slow')
    loader = DataLoader(train_data, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=device == 'cuda')
    val_loader = DataLoader(val_data, batch_size=args.batch, num_workers=args.workers) if val_data else None
    model = SmallUNet(args.width).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=device == 'cuda')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out/'split.json').write_text(json.dumps({'train': [dataset.pairs[i][0].name for i in train_ids], 'val': [dataset.pairs[i][0].name for i in val_ids]}, ensure_ascii=False, indent=2), encoding='utf-8')
    best = -1
    for epoch in range(args.epochs):
        model.train()
        total, count = 0., 0
        for step, (x, y) in enumerate(loader, 1):
            x, y = x.to(device), y.to(device)
            optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, enabled=device == 'cuda'):
                z = model(x)
                loss = F.cross_entropy(z, y, ignore_index=IGNORE) if (y != IGNORE).any() else z.sum() * 0
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            total += float(loss.detach())
            count += 1
            if step % 30 == 0:
                print(f'epoch={epoch+1} step={step}/{len(loader)} loss={total/count:.4f}', flush=True)
        metrics = score(model, val_loader, device) if val_loader else {'mIoU_present_nonignored': None}
        print(f'epoch={epoch+1} train_loss={total/max(count,1):.4f} val={metrics}', flush=True)
        checkpoint = {'model': model.state_dict(), 'width': args.width, 'size': args.size,
                      'epoch': epoch+1, 'metrics': metrics, 'classes': N_CLASSES, 'ignore_index': IGNORE}
        torch.save(checkpoint, out/'last.pth')
        value = metrics['mIoU_present_nonignored']
        if value is None or value >= best:
            if value is not None:
                best = value
            torch.save(checkpoint, out/'best.pth')
    print(f'Training finished; checkpoint: {out / "best.pth"}')


def infer(args):
    ck = torch.load(args.weights, map_location='cpu', weights_only=True)
    model = SmallUNet(ck['width'])
    model.load_state_dict(ck['model'])
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    model = model.to(device).eval()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    paths = images(args.images)
    with torch.inference_mode():
        for idx, path in enumerate(paths, 1):
            with Image.open(path) as im:
                original_size = im.size
                arr = np.asarray(im.convert('RGB').resize((ck['size'], ck['size']), Image.Resampling.BILINEAR), dtype=np.uint8).copy()
            if original_size != (1024, 1024):
                raise ValueError(f'Official test image should be 1024x1024: {path}: {original_size}')
            x = torch.from_numpy(arr.transpose(2, 0, 1).copy()).unsqueeze(0).float().to(device) / 255.
            logits = model(x)
            logits = F.interpolate(logits, size=(1024, 1024), mode='bilinear', align_corners=False)
            prediction = logits.argmax(1).squeeze(0).byte().cpu().numpy()
            Image.fromarray(prediction, mode='L').save(out / (path.stem + '.png'))
            if idx % 100 == 0:
                print(f'Predicted {idx}/{len(paths)}', flush=True)
    print(f'Generated {len(paths)} predictions in {out}')


def pack(args):
    tests = images(args.images)
    names = {p.stem + '.png' for p in tests}
    target = Path(args.predictions)
    predicted = list(target.glob('*.png'))
    got = {p.name for p in predicted}
    if names != got or len(predicted) != len(names):
        raise ValueError(f'Prediction mismatch: missing={sorted(names-got)[:10]}, extra={sorted(got-names)[:10]}')
    for p in predicted:
        with Image.open(p) as im:
            if im.mode != 'L' or im.size != (1024, 1024):
                raise ValueError(f'Invalid PNG mode or dimensions: {p}: {im.mode}, {im.size}')
            a = np.asarray(im)
        if a.min() < 0 or a.max() > 8:
            raise ValueError(f'Invalid class IDs: {p}')
    Path(args.zip).parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(predicted):
            z.write(p, arcname=p.name)
    print(f'VALIDATED {len(predicted)} single-channel PNG files; created: {args.zip}')


def main():
    p = argparse.ArgumentParser(description='Minimal UAV segmentation: train, infer, pack')
    sub = p.add_subparsers(dest='cmd', required=True)
    tr = sub.add_parser('train')
    tr.add_argument('--images', required=True)
    tr.add_argument('--masks', required=True)
    tr.add_argument('--output', default='runs/quick')
    tr.add_argument('--epochs', type=int, default=5)
    tr.add_argument('--size', type=int, default=384)
    tr.add_argument('--batch', type=int, default=8)
    tr.add_argument('--width', type=int, default=24)
    tr.add_argument('--lr', type=float, default=0.001)
    tr.add_argument('--workers', type=int, default=2)
    tr.add_argument('--val-fraction', type=float, default=.1)
    tr.add_argument('--seed', type=int, default=42)
    tr.add_argument('--cpu', action='store_true')
    inf = sub.add_parser('infer')
    inf.add_argument('--images', required=True)
    inf.add_argument('--weights', required=True)
    inf.add_argument('--output', default='predictions')
    inf.add_argument('--cpu', action='store_true')
    pk = sub.add_parser('pack')
    pk.add_argument('--images', required=True)
    pk.add_argument('--predictions', required=True)
    pk.add_argument('--zip', default='submission.zip')
    args = p.parse_args()
    if args.cmd == 'train':
        if args.size % 8:
            p.error('--size must be divisible by 8')
        if not 0 <= args.val_fraction < 1:
            p.error('--val-fraction must be between 0 (inclusive) and 1 (exclusive)')
        train(args)
    elif args.cmd == 'infer':
        infer(args)
    else:
        pack(args)


if __name__ == '__main__':
    main()

