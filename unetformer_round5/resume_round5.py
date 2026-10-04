"""Resume round-5 training after an interrupted epoch boundary."""
import argparse
import copy
import json
import math
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'unetformer_round3'))
from round3 import (AugmentedPairs, MODES, eval_mode, evaluate_hv, export,
                    freeze_bn, sampling_weights, update_ema)
from unetformer_pipeline import (Pairs, forward, loader, loss_fn, make_model,
                                 save_json, seed_worker)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    cli = parser.parse_args()
    out = Path(cli.out)
    args = argparse.Namespace(**json.loads((out / 'config.json').read_text(encoding='utf-8')))
    recipe = json.loads((out / 'recipe.json').read_text(encoding='utf-8'))
    split = json.loads(Path(args.split).read_text(encoding='utf-8'))
    audit = json.loads(Path(args.audit).read_text(encoding='utf-8'))
    baseline = json.loads((out / 'baseline.json').read_text(encoding='utf-8'))
    previous = baseline['dihedral8_mIoU']
    trial_dir = out / recipe['name']
    saved = torch.load(trial_dir / 'last_training.pth', map_location='cpu', weights_only=True)
    if saved['recipe'] != recipe or saved['epoch'] >= recipe['epochs']:
        raise ValueError('Unexpected restart checkpoint')

    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = make_model().cuda()
    ema = copy.deepcopy(model).eval()
    model.load_state_dict(saved['model'])
    ema.load_state_dict(saved['ema'])
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': recipe['lr']},
        {'params': model.decoder.parameters(), 'lr': recipe['lr'] * 5}],
        weight_decay=.01)
    optimizer.load_state_dict(saved['optimizer'])
    scaler = torch.amp.GradScaler('cuda')
    scaler.load_state_dict(saved['scaler'])
    weights = torch.tensor(audit['weights'], dtype=torch.float32, device='cuda')
    weights[1:] = weights[1:].sqrt()
    weights[1:] /= weights[1:].mean()
    ds = AugmentedPairs(split['train'], Path(args.cache) / 'images',
                        Path(args.cache) / 'masks', 512, strength=recipe['strength'])
    val = loader(Pairs(split['val'], args.images, args.masks, 512), 2, args.workers)
    sample_weights = [1 + .5 * (value - 1) for value in
                      sampling_weights(split['train'], Path(args.cache) / 'masks')]
    batches = math.ceil(len(ds) / 4)
    total = recipe['epochs'] * batches
    step = saved['epoch'] * batches
    best = baseline['hvflip']['mIoU']
    selected = None

    for epoch in range(saved['epoch'] + 1, recipe['epochs'] + 1):
        ds.size = recipe['sizes'][epoch - 1]
        generator = torch.Generator().manual_seed(args.seed + epoch)
        sampler = WeightedRandomSampler(sample_weights, len(ds), replacement=True,
                                         generator=generator)
        train = DataLoader(ds, batch_size=4, sampler=sampler, num_workers=args.workers,
                           pin_memory=True, worker_init_fn=seed_worker,
                           generator=generator)
        model.train()
        freeze_bn(model)
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0
        for i, (x, y) in enumerate(train):
            factor = min(1., (step + 1) / 50) * (.1 + .9 * (1 - step / total) ** .9)
            for group, lr in zip(optimizer.param_groups,
                                 [recipe['lr'], recipe['lr'] * 5]):
                group['lr'] = lr * factor
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            with torch.autocast('cuda', dtype=torch.float16):
                loss = loss_fn(forward(model, x), y, weights)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss')
            group_size = min(2, len(train) - (i // 2) * 2)
            scaler.scale(loss / group_size).backward()
            if (i + 1) % 2 == 0 or i + 1 == len(train):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                update_ema(ema, model)
            total_loss += loss.item()
            step += 1
            if i % 250 == 0:
                print(f'epoch={epoch} size={ds.size} batch={i+1}/{len(train)} '
                      f'loss={total_loss/(i+1):.4f}', flush=True)
        del train
        for variant, net in [('raw', model), ('ema', ema)]:
            result = evaluate_hv(net, val)
            row = {'trial': recipe['name'], 'epoch': epoch, 'variant': variant,
                   'train_size': ds.size, 'loss': total_loss / batches, **result}
            with (trial_dir / 'history.jsonl').open('a', encoding='utf-8') as file:
                file.write(json.dumps(row) + '\n')
            print(f'validation epoch={epoch} {variant} mIoU={result["mIoU"]:.8f}',
                  flush=True)
            if result['mIoU'] > best:
                best = result['mIoU']
                selected = row
                torch.save({'model': net.state_dict(), 'epoch': epoch,
                            'size': 512, 'metrics': result, 'recipe': recipe,
                            'variant': variant}, trial_dir / 'best.pth')
        torch.save({'model': model.state_dict(), 'ema': ema.state_dict(),
                    'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                    'epoch': epoch, 'recipe': recipe}, trial_dir / 'last_training.pth')
        save_json(out / 'status.json', {'stage': 'training', 'trial': recipe['name'],
                                        'epoch': epoch, 'best_mIoU': best})

    if selected is None:
        save_json(out / 'selection.json', {
            'improved': False, 'reason': 'No new checkpoint exceeded the four-flip baseline',
            'baseline_dihedral8_mIoU': previous})
        return
    model.load_state_dict(torch.load(trial_dir / 'best.pth', map_location='cpu',
                                     weights_only=True)['model'])
    model.eval()
    mode = MODES['512_dihedral8']
    result = eval_mode(model, args, split['val'], mode)
    improved = result['mIoU'] > previous
    save_json(out / 'selection.json', {
        'improved': improved, 'baseline_dihedral8_mIoU': previous,
        'candidate_hvflip': selected, 'candidate_dihedral8': result,
        'delta_percentage_points': 100 * (result['mIoU'] - previous),
        'official_score': None})
    print(f'candidate dihedral8={result["mIoU"]:.8f} improved={improved}', flush=True)
    if improved:
        import shutil
        shutil.copyfile(trial_dir / 'best.pth', out / 'best.pth')
        export(model, args.test_images, out, mode)
        (out / 'submission_unetformer_round3_test2.zip').rename(
            out / 'submission_unetformer_round5_test2.zip')
        manifest = json.loads((out / 'submission_manifest.json').read_text(encoding='utf-8'))
        manifest['file'] = 'submission_unetformer_round5_test2.zip'
        save_json(out / 'submission_manifest.json', manifest)


if __name__ == '__main__':
    main()

