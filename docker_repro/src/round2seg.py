"""SegFormer round-two training; fixed holdout, optional feature MixStyle."""
import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import SegformerForSemanticSegmentation, SegformerConfig
from improvedseg import (CropDataset, paired_paths, split_pairs, seed_worker,
                         segmentation_loss, metrics_from_hist, tensor_image)


class MixStyle(nn.Module):
    def __init__(self, p=0., alpha=.1):
        super().__init__()
        self.p, self.alpha = p, alpha

    def forward(self, x):
        if not self.training or x.shape[0] < 2 or random.random() >= self.p:
            return x
        value = x.float()
        mean = value.mean((2, 3), keepdim=True).detach()
        std = (value.var((2, 3), keepdim=True, unbiased=False)+1e-6).sqrt().detach()
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample((len(x), 1, 1, 1)).to(x.device)
        order = torch.randperm(len(x), device=x.device)
        mixed = (value-mean)/std * (lam*std+(1-lam)*std[order]) + lam*mean+(1-lam)*mean[order]
        return mixed.to(x.dtype)


class Segmenter(nn.Module):
    def __init__(self, source, mixstyle=0., config_only=False):
        super().__init__()
        if config_only:
            self.net = SegformerForSemanticSegmentation(SegformerConfig.from_pretrained(source, local_files_only=True))
        else:
            self.net = SegformerForSemanticSegmentation.from_pretrained(source, local_files_only=True)
        self.mixstyle = MixStyle(mixstyle)
        for stage in self.net.segformer.stages[:2]:
            stage.register_forward_hook(self.apply_style)
        self.register_buffer('mean', torch.tensor([.485,.456,.406])[None,:,None,None])
        self.register_buffer('std', torch.tensor([.229,.224,.225])[None,:,None,None])

    def apply_style(self, module, inputs, output):
        return self.mixstyle(output)

    def forward(self, x):
        return self.net(pixel_values=(x-self.mean)/self.std).logits


class ValidationDataset(Dataset):
    def __init__(self, pairs, size):
        self.pairs, self.size = pairs, size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        image, mask = self.pairs[i]
        with Image.open(image) as im:
            x = tensor_image(im.convert('RGB').resize((self.size,self.size), Image.Resampling.BILINEAR))
        with Image.open(mask) as im:
            y = torch.from_numpy(np.asarray(im,dtype=np.int64).copy())
        return x,y


@torch.inference_mode()
def evaluate(model, loader):
    model.eval()
    hist = torch.zeros(81,dtype=torch.int64,device='cuda')
    for x,y in loader:
        x,y=x.cuda(non_blocking=True),y.cuda(non_blocking=True)
        with torch.autocast('cuda',dtype=torch.float16):
            logits=model(x)
        pred=F.interpolate(logits.float(),size=y.shape[-2:],mode='bilinear',align_corners=False).argmax(1)
        valid=y!=0
        hist+=torch.bincount(9*y[valid]+pred[valid],minlength=81)
    return metrics_from_hist(hist.cpu().reshape(9,9))


def train(a):
    torch.set_num_threads(4)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark=True
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    pairs=paired_paths(a)
    training,validation,split=split_pairs(pairs,a.split,.1,a.seed)
    (out/'split.json').write_text(json.dumps(split),encoding='utf-8')
    (out/'config.json').write_text(json.dumps(vars(a),indent=2),encoding='utf-8')
    weights=json.loads(Path(a.class_balance).read_text())['weights']
    weights=torch.tensor(weights,device='cuda')
    generator=torch.Generator().manual_seed(a.seed)
    loader=DataLoader(CropDataset(training,a.size,True,sampling='resize'),batch_size=a.batch,
        shuffle=True,num_workers=a.workers,pin_memory=True,worker_init_fn=seed_worker,
        generator=generator,persistent_workers=a.workers>0)
    val=DataLoader(ValidationDataset(validation,a.size),batch_size=2,num_workers=a.workers,
        pin_memory=True,persistent_workers=a.workers>0)
    model=Segmenter(a.source,a.mixstyle,config_only=bool(a.init)).cuda()
    if a.init:
        model.load_state_dict(torch.load(a.init,map_location='cpu',weights_only=False)['model'])
    optimizer=torch.optim.AdamW([
        {'params':model.net.segformer.parameters(),'lr':a.lr},
        {'params':model.net.decode_head.parameters(),'lr':a.lr*10}],weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda')
    total=a.epochs*len(loader); step=0; best=-1
    started=time.time()
    for epoch in range(1,a.epochs+1):
        model.train(); total_loss=0; optimizer.zero_grad(set_to_none=True)
        for batch,(x,y) in enumerate(loader,1):
            factor=min(1.,(step+1)/100)*(1-step/total)**.9
            for group,base in zip(optimizer.param_groups,[a.lr,a.lr*10]): group['lr']=base*factor
            x,y=x.cuda(non_blocking=True),y.cuda(non_blocking=True)
            with torch.autocast('cuda',dtype=torch.float16):
                logits=F.interpolate(model(x),size=y.shape[-2:],mode='bilinear',align_corners=False)
                loss=segmentation_loss(logits,y,.5,weights)
            if not torch.isfinite(loss): raise RuntimeError('Non-finite loss')
            group_start=((batch-1)//a.accum)*a.accum
            group_count=min(a.accum,len(loader)-group_start)
            scaler.scale(loss/group_count).backward()
            if batch%a.accum==0 or batch==len(loader):
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            total_loss+=loss.item(); step+=1
            if batch%100==0 or batch==1:
                print(f'epoch={epoch}/{a.epochs} batch={batch}/{len(loader)} loss={total_loss/batch:.4f} elapsed={time.time()-started:.0f}s peakGB={torch.cuda.max_memory_allocated()/1e9:.2f}',flush=True)
            if a.smoke and batch==2:
                model.eval()
                with torch.no_grad(): assert torch.isfinite(model(x)).all()
                print('SMOKE_OK',flush=True); return
        metrics=evaluate(model,val)
        record={'epoch':epoch,'loss':total_loss/len(loader),'elapsed_seconds':time.time()-started,**metrics}
        with (out/'history.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
        state={'model':model.state_dict(),'epoch':epoch,'metrics':metrics,'config':vars(a),'split':split,
               'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict()}
        torch.save(state,out/'last.pth')
        if metrics['mIoU_present_nonignored']>best:
            best=metrics['mIoU_present_nonignored']; torch.save(state,out/'best.pth')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('images','masks','split','class-balance','source','out'): p.add_argument('--'+name,required=True)
    p.add_argument('--init'); p.add_argument('--epochs',type=int,default=6)
    p.add_argument('--size',type=int,default=384); p.add_argument('--batch',type=int,default=4)
    p.add_argument('--accum',type=int,default=2); p.add_argument('--workers',type=int,default=4)
    p.add_argument('--seed',type=int,default=2026); p.add_argument('--lr',type=float,default=6e-5)
    p.add_argument('--mixstyle',type=float,default=0.); p.add_argument('--smoke',action='store_true')
    train(p.parse_args())
