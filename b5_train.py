"""Single SegFormer B5 fine tuning from local public pretrained weights."""
import argparse, copy, json, random, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from round2seg import Segmenter, ValidationDataset, evaluate
from round4_train import update_ema, lovasz_softmax
from improvedseg import CropDataset, paired_paths, split_pairs, seed_worker, segmentation_loss


def train(a):
    torch.set_num_threads(4)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark = True
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if (out/'history.jsonl').exists():
        raise ValueError('Use a fresh run directory')
    training, validation, split = split_pairs(paired_paths(a), a.split, .1, a.seed)
    (out/'config.json').write_text(json.dumps(vars(a), indent=2), encoding='utf-8')
    (out/'split.json').write_text(json.dumps(split), encoding='utf-8')
    student = Segmenter(a.source)
    # Preserve the pretrained encoder AND decoder; only the ADE150 classifier changes.
    old = student.net.decode_head.classifier
    student.net.decode_head.classifier = torch.nn.Conv2d(old.in_channels, 9, 1)
    torch.nn.init.normal_(student.net.decode_head.classifier.weight, std=.02)
    torch.nn.init.zeros_(student.net.decode_head.classifier.bias)
    student.net.config.num_labels = 9
    student.net.config.id2label = {i: str(i) for i in range(9)}
    student.net.config.label2id = {str(i): i for i in range(9)}
    student.net.config.semantic_loss_ignore_index = 0
    student.net.config.save_pretrained(out/'model_config')
    if a.init:
        student.load_state_dict(torch.load(a.init, map_location='cpu', weights_only=False)['model'])
    student.cuda()
    student.net.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    ema = copy.deepcopy(student).eval().requires_grad_(False)
    ema.net.gradient_checkpointing_disable()
    optimizer = torch.optim.AdamW([
        {'params': student.net.segformer.parameters(), 'lr': a.lr},
        {'params': student.net.decode_head.parameters(), 'lr': a.head_lr}],
        weight_decay=.02, foreach=False)
    weights = torch.tensor(json.loads(Path(a.class_balance).read_text())['weights'], device='cuda')
    loader = DataLoader(CropDataset(training, a.size, True, sampling='resize'),
        batch_size=a.batch, shuffle=True, generator=torch.Generator().manual_seed(a.seed),
        num_workers=a.workers, pin_memory=True, worker_init_fn=seed_worker,
        persistent_workers=a.workers > 0)
    val = DataLoader(ValidationDataset(validation, a.size), batch_size=1,
        num_workers=a.workers, pin_memory=True, persistent_workers=a.workers > 0)
    updates = 0; step = 0; best = -1.; started = time.time()
    total = a.epochs * len(loader)
    print(f'train={len(training)} val={len(validation)} parameters={sum(p.numel() for p in student.parameters())}', flush=True)
    for epoch in range(1, a.epochs+1):
        student.train(); optimizer.zero_grad(set_to_none=True); sum_loss = 0.
        for batch, (x, y) in enumerate(loader, 1):
            factor = min(1., (step+1)/100) * (.05+.95*(1-step/total)**.9)
            for group, base in zip(optimizer.param_groups, [a.lr, a.head_lr]):
                group['lr'] = base * factor
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = F.interpolate(student(x), size=y.shape[-2:], mode='bilinear', align_corners=False)
                loss = segmentation_loss(logits.float(), y, .5, weights) + .3*lovasz_softmax(logits, y)
            if not torch.isfinite(loss): raise RuntimeError('Non-finite loss')
            divisor = min(a.accum, len(loader)-((batch-1)//a.accum)*a.accum)
            (loss/divisor).backward()
            if batch % a.accum == 0 or batch == len(loader):
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1., error_if_nonfinite=True)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
                updates += 1; update_ema(ema, student, updates)
            sum_loss += loss.item(); step += 1
            if batch == 1 or batch % 50 == 0:
                print(f'epoch={epoch}/{a.epochs} batch={batch}/{len(loader)} loss={sum_loss/batch:.4f} elapsed={time.time()-started:.0f}s peakGB={torch.cuda.max_memory_allocated()/1e9:.2f}', flush=True)
            if a.smoke and batch >= a.accum:
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    prediction = ema(x)
                    assert prediction.shape[1] == 9 and torch.isfinite(prediction).all()
                print('SMOKE_OK optimizer_step_and_ema', flush=True); return
        metrics = evaluate(ema, val)
        row = {'epoch': epoch, 'loss': sum_loss/len(loader), 'elapsed_seconds': time.time()-started, **metrics}
        with (out/'history.jsonl').open('a', encoding='utf-8') as f: f.write(json.dumps(row)+'\n')
        print(json.dumps(row), flush=True)
        state = {'model': ema.state_dict(), 'epoch': epoch, 'metrics': metrics, 'config': vars(a), 'split': split}
        if metrics['mIoU_present_nonignored'] > best:
            best = metrics['mIoU_present_nonignored']; torch.save(state, out/'best.pth')
        torch.save({**state, 'student': student.state_dict(), 'optimizer': optimizer.state_dict()}, out/'last.pth')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('images', 'masks', 'split', 'class-balance', 'source', 'out'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--init')
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--size', type=int, default=512)
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--accum', type=int, default=4)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=20261001)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--head-lr', type=float, default=1e-4)
    p.add_argument('--smoke', action='store_true')
    train(p.parse_args())

