"""Train, validate and export official UNetFormer on UAV class-ID masks.

Model implementation: WangLibo1995/GeoSeg (GPL-3.0); see LICENSE/upstream.json.
"""
import argparse
import concurrent.futures
import hashlib
import json
import math
import random
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from UNetFormer import UNetFormer

def make_model(source=None):
    model = UNetFormer(backbone_name='resnet18.fb_swsl_ig1b_ft_in1k',
                       pretrained=False, num_classes=9)
    if source:
        weights = torch.load(source, map_location='cpu', weights_only=True)
        result = model.backbone.load_state_dict(weights, strict=False)
        if result.missing_keys or set(result.unexpected_keys) != {'fc.weight', 'fc.bias'}:
            raise RuntimeError(f'Unexpected pretrained mapping: {result}')
    return model

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
        x.resize((768,768), Image.Resampling.BILINEAR).save(cache/'images'/name)
        Image.fromarray(y.astype(np.uint8)).resize((768,768), Image.Resampling.NEAREST).save(cache/'masks'/name)
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
    return model((x-mean)/std)

def loss_fn(logits, y, weights):
    if isinstance(logits, tuple):
        main, aux = logits
        return loss_fn(main, y, weights) + .4*loss_fn(aux, y, weights)
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

def train(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if (out/'history.jsonl').exists(): raise FileExistsError('Use a fresh output directory')
    split = json.loads(Path(a.split).read_text())
    save_json(out/'split.json', split); save_json(out/'run_config.json', vars(a))
    audit = prepare(a, split, out)
    weights = torch.tensor(audit['weights'],device='cuda', dtype=torch.float32)
    model = make_model(a.source).cuda()
    save_json(out/'model_config.json', {'architecture':'UNetFormer',
        'backbone':'resnet18.fb_swsl_ig1b_ft_in1k','num_classes':9,'labels':LABELS})
    print(f'parameters={sum(p.numel() for p in model.parameters())}',flush=True)
    scaler = torch.amp.GradScaler('cuda')
    best, epoch_global, started = -1., 0, time.time()
    stages = [(512,a.epochs,6e-5), (768,a.finetune_epochs,1.5e-5)]
    for stage,(size,epochs,lr) in enumerate(stages):
        if not epochs: continue
        if stage:
            model.load_state_dict(torch.load(out/'best.pth',map_location='cpu',weights_only=True)['model'])
        batch = a.batch if size==512 else max(2,a.batch//2)
        accum = max(1, 8//batch)
        ds = Pairs(split['train'],Path(a.cache)/'images',Path(a.cache)/'masks',size,True)
        batches = loader(ds,batch,a.workers,True)
        val = loader(Pairs(split['val'],a.images,a.masks,size),2,a.workers)
        optimizer = torch.optim.AdamW([
            {'params':model.backbone.parameters(),'lr':lr},
            {'params':model.decoder.parameters(),'lr':lr*10}],weight_decay=.01)
        total = epochs*len(batches); step=0
        for epoch in range(epochs):
            epoch_global += 1; model.train(); running=0.; optimizer.zero_grad(set_to_none=True)
            for i,(x,y) in enumerate(batches):
                factor=min(1.,(step+1)/100)*(1-step/total)**.9
                for g,base in zip(optimizer.param_groups,(lr,lr*10)): g['lr']=base*factor
                x,y=x.cuda(non_blocking=True),y.cuda(non_blocking=True)
                with torch.autocast('cuda',dtype=torch.float16): loss=loss_fn(forward(model,x),y,weights)
                if not torch.isfinite(loss): raise RuntimeError('Non-finite training loss')
                group_start=(i//accum)*accum
                group_count=min(accum,len(batches)-group_start)
                scaler.scale(loss/group_count).backward()
                if (i+1)%accum==0 or i+1==len(batches):
                    scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                    scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                running+=loss.item(); step+=1
                if i%100==0: print(f'epoch={epoch_global} size={size} batch={i+1}/{len(batches)} loss={running/(i+1):.4f} elapsed={time.time()-started:.0f}s',flush=True)
                if a.smoke and i==2:
                    print('SMOKE_OK',flush=True);return
            result=evaluate(model,val)
            record={'epoch':epoch_global,'size':size,'loss':running/len(batches),
                    'elapsed_seconds':time.time()-started,**result}
            with (out/'history.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(record)+'\n')
            print(json.dumps(record),flush=True)
            state={'model':model.state_dict(),'epoch':epoch_global,'size':size,'metrics':result}
            torch.save(state,out/'last.pth')
            if result['mIoU']>best:
                best=result['mIoU'];torch.save(state,out/'best.pth')
            save_json(out/'status.json',{'stage':'training','epoch':epoch_global,'best_mIoU':best})
        del batches,val,optimizer
    state=torch.load(out/'best.pth',map_location='cpu',weights_only=True)
    model.load_state_dict(state['model'])
    # Compare inference settings on the same untouched holdout and select one.
    candidates=[]
    for size,flip in [(state['size'],False),(512,True),(768,True)]:
        val=loader(Pairs(split['val'],a.images,a.masks,size),2,a.workers)
        result=evaluate(model,val,flip);del val
        candidates.append({'size':size,'hflip':flip,**result})
        print('inference_validation '+json.dumps(candidates[-1]),flush=True)
    selected=max(candidates,key=lambda x:x['mIoU'])
    save_json(out/'validation.json',{'candidates':candidates,'selected':selected,'official_score':None})
    predict(model,a.test_images,out,selected['size'],selected['hflip'])
    save_json(out/'status.json',{'stage':'complete','best_epoch':state['epoch'],
                               'selected_mIoU':selected['mIoU'],'official_score':None})

@torch.inference_mode()
def predict(model, directory, out, size, flip):
    model.eval(); dest=out/'predictions'; dest.mkdir(exist_ok=True)
    paths=sorted(Path(directory).glob('*.png'))
    expected={f'test2_{i}.png' for i in range(1,1301)}
    if {p.name for p in paths} != expected:
        raise ValueError(f'Expected complete test2_1.png..test2_1300.png, found {len(paths)} files; training weights are preserved')
    if list(dest.glob('*.png')): raise FileExistsError('Prediction directory must be empty')
    for i,p in enumerate(paths,1):
        with Image.open(p) as im:
            shape=(im.height,im.width)
            x=tensor(im.convert('RGB').resize((size,size),Image.Resampling.BILINEAR))[None].cuda()
        with torch.autocast('cuda',dtype=torch.float16):
            z=forward(model,x).float()
            if flip: z=(z+forward(model,x.flip(-1)).float().flip(-1))*.5
        pred=F.interpolate(z,size=shape,mode='bilinear',align_corners=False).argmax(1)[0].byte().cpu().numpy()
        Image.fromarray(pred).save(dest/p.name)
        if i%50==0: print(f'predict {i}/{len(paths)}',flush=True)
    archive=out/'submission_unetformer_test2.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in paths:
            q=dest/p.name
            with Image.open(q) as im,Image.open(p) as original:
                values=np.asarray(im)
                if im.mode!='L' or im.size!=original.size or values.min()<0 or values.max()>8:
                    raise ValueError(f'Invalid prediction: {q}')
            z.write(q,q.name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert sorted(z.namelist())==sorted(p.name for p in paths)
    save_json(out/'submission_manifest.json',{'file':archive.name,'count':len(paths),
        'sha256':hashlib.file_digest(archive.open('rb'),'sha256').hexdigest(),
        'names':[p.name for p in paths],'size':size,'hflip':flip,'official_score':None})
    print(f'ZIP_VERIFIED {archive} count={len(paths)}',flush=True)

def main():
    p=argparse.ArgumentParser()
    for n in ['images','masks','split','source','cache','test-images','out']:p.add_argument('--'+n,required=True)
    p.add_argument('--epochs',type=int,default=24)
    p.add_argument('--finetune-epochs',type=int,default=4)
    p.add_argument('--batch',type=int,default=4)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    random.seed(20260928);np.random.seed(20260928);torch.manual_seed(20260928)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required')
    try:train(a)
    except Exception as e:
        Path(a.out).mkdir(parents=True,exist_ok=True)
        save_json(Path(a.out)/'status.json',{'stage':'failed','error':repr(e)})
        raise

if __name__=='__main__': main()
