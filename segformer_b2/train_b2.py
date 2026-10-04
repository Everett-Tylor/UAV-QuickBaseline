"""Train SegFormer-B2 on the fixed UAV split with resumable epoch checkpoints."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation

from segformer_common import (LABELS, Pairs, evaluate, forward, loader, loss_fn,
                              save_json, seed_worker)


def main():
    parser = argparse.ArgumentParser()
    for name in ('images', 'masks', 'split', 'source', 'cache', 'audit', 'out'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--accum', type=int, default=4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required')
    random.seed(20261004)
    np.random.seed(20261004)
    torch.manual_seed(20261004)
    split = json.loads(Path(args.split).read_text(encoding='utf-8'))
    audit = json.loads(Path(args.audit).read_text(encoding='utf-8'))
    if len(split['train']) != 6296 or len(split['val']) != 700:
        raise ValueError('Unexpected training split')
    if set(split['train']) & set(split['val']):
        raise ValueError('Train/validation overlap')
    if hashlib.sha256(Path(args.split).read_bytes()).hexdigest() != audit['split_sha256']:
        raise ValueError('Split and class weights do not match')
    if len(list((Path(args.cache) / 'images').glob('*.png'))) != 6996:
        raise ValueError('Incomplete image cache')
    if len(list((Path(args.cache) / 'masks').glob('*.png'))) != 6996:
        raise ValueError('Incomplete mask cache')
    weights = torch.tensor(audit['weights'], dtype=torch.float32, device='cuda')
    model = SegformerForSemanticSegmentation.from_pretrained(
        args.source, num_labels=9, ignore_mismatched_sizes=True,
        local_files_only=True).cuda()
    model.config.id2label = dict(enumerate(LABELS))
    model.config.label2id = {name: index for index, name in enumerate(LABELS)}
    model.config.save_pretrained(out / 'model_config')
    optimizer = torch.optim.AdamW([
        {'params': model.segformer.parameters(), 'lr': 4e-5},
        {'params': model.decode_head.parameters(), 'lr': 4e-4}], weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda')
    first_epoch, step, best = 1, 0, -1.
    if args.resume:
        saved = torch.load(out / 'last_training.pth', map_location='cpu',
                           weights_only=True)
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['scaler'])
        first_epoch, step, best = saved['epoch'] + 1, saved['step'], saved['best_mIoU']
    elif (out / 'history.jsonl').exists():
        raise FileExistsError('Use --resume or a fresh output directory')
    else:
        save_json(out / 'run_config.json', vars(args))
        save_json(out / 'data_audit.json', audit)
    print(f'parameters={sum(p.numel() for p in model.parameters())} '
          f'first_epoch={first_epoch}', flush=True)
    train_set = Pairs(split['train'], Path(args.cache) / 'images',
                      Path(args.cache) / 'masks', 512, True)
    val = loader(Pairs(split['val'], args.images, args.masks, 512),
                 2, args.workers)
    microbatches = math.ceil(len(train_set) / args.batch)
    steps_per_epoch = math.ceil(microbatches / args.accum)
    total_steps = args.epochs * steps_per_epoch
    started = time.time()
    for epoch in range(first_epoch, args.epochs + 1):
        generator = torch.Generator().manual_seed(20261004 + epoch)
        train = DataLoader(train_set, batch_size=args.batch, shuffle=True,
                           num_workers=args.workers, pin_memory=True,
                           worker_init_fn=seed_worker, generator=generator)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.
        for index, (x, y) in enumerate(train):
            if index % args.accum == 0:
                factor = min(1., (step + 1) / 100) * \
                         (.1 + .9 * max(0., 1 - step / total_steps) ** .9)
                for group, lr in zip(optimizer.param_groups, (4e-5, 4e-4)):
                    group['lr'] = lr * factor
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            group_size = min(args.accum, len(train) -
                             (index // args.accum) * args.accum)
            with torch.autocast('cuda', dtype=torch.float16):
                loss = loss_fn(forward(model, x), y, weights)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss')
            scaler.scale(loss / group_size).backward()
            if (index + 1) % args.accum == 0 or index + 1 == len(train):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                step += 1
            running += loss.item()
            if index % 200 == 0:
                print(f'epoch={epoch} batch={index+1}/{len(train)} '
                      f'loss={running/(index+1):.4f} '
                      f'elapsed={time.time()-started:.0f}s', flush=True)
            if args.smoke and index == 1:
                print(f'SMOKE_OK peak_allocated_bytes='
                      f'{torch.cuda.max_memory_allocated()}', flush=True)
                return
        del train
        result = evaluate(model, val)
        row = {'epoch': epoch, 'size': 512, 'loss': running / microbatches,
               'elapsed_seconds': time.time() - started, **result}
        with (out / 'history.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row) + '\n')
        print(f'validation epoch={epoch} mIoU={result["mIoU"]:.8f}', flush=True)
        checkpoint = {'model': model.state_dict(), 'epoch': epoch,
                      'metrics': result, 'size': 512}
        if result['mIoU'] > best:
            best = result['mIoU']
            torch.save(checkpoint, out / 'best.pth')
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict(), 'epoch': epoch, 'step': step,
                    'best_mIoU': best}, out / 'last_training.tmp')
        (out / 'last_training.tmp').replace(out / 'last_training.pth')
        save_json(out / 'status.json', {'stage': 'training', 'epoch': epoch,
                                        'best_mIoU': best})
    save_json(out / 'status.json', {'stage': 'trained', 'epoch': args.epochs,
                                    'best_mIoU': best})


if __name__ == '__main__':
    main()
