"""Train a local DINOv3 encoder and pyramid decoder on the fixed UAV split."""
import argparse,copy,json,random,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from dinoseg import DinoSegmenter
from improvedseg import CropDataset,paired_paths,split_pairs,seed_worker,segmentation_loss,metrics_from_hist
from round2seg import ValidationDataset
from round4_train import lovasz_softmax,update_ema

@torch.inference_mode()
def evaluate(model,loader):
    model.eval();hist=torch.zeros(81,dtype=torch.int64,device='cuda')
    for x,y in loader:
        x,y=x.cuda(non_blocking=True),y.cuda(non_blocking=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(x)
        pred=F.interpolate(logits.float(),size=y.shape[-2:],mode='bilinear',align_corners=False).argmax(1)
        valid=y!=0;hist+=torch.bincount(9*y[valid]+pred[valid],minlength=81)
    return metrics_from_hist(hist.cpu().reshape(9,9))

def train(a):
    torch.set_num_threads(4);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark=True
    if not torch.cuda.is_bf16_supported():raise RuntimeError('BF16 CUDA GPU required')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if (out/'history.jsonl').exists():raise ValueError('Use a fresh run directory')
    training,validation,split=split_pairs(paired_paths(a),a.split,.1,a.seed)
    (out/'config.json').write_text(json.dumps(vars(a),indent=2));(out/'split.json').write_text(json.dumps(split))
    weights=torch.tensor(json.loads(Path(a.class_balance).read_text())['weights'],device='cuda')
    g=torch.Generator().manual_seed(a.seed)
    loader=DataLoader(CropDataset(training,a.size,True,sampling='resize'),batch_size=a.batch,shuffle=True,generator=g,num_workers=a.workers,pin_memory=True,worker_init_fn=seed_worker,persistent_workers=a.workers>0)
    val=DataLoader(ValidationDataset(validation,a.size),batch_size=2,num_workers=a.workers,pin_memory=True,persistent_workers=a.workers>0)
    model=DinoSegmenter(a.source,config_only=bool(a.init)).cuda()
    if a.init:model.load_state_dict(torch.load(a.init,map_location='cpu',weights_only=False)['model'])
    model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    ema=copy.deepcopy(model).eval().requires_grad_(False)
    backbone=list(model.backbone.parameters());ids={id(p) for p in backbone}
    decoder=[p for p in model.parameters() if id(p) not in ids]
    optimizer=torch.optim.AdamW([{'params':backbone,'lr':a.lr},{'params':decoder,'lr':a.head_lr}],weight_decay=.01,foreach=False)
    best=-1.;updates=0;step=0;total=a.epochs*len(loader);start=time.time()
    for epoch in range(1,a.epochs+1):
        frozen=epoch<=a.freeze_epochs;model.freeze_backbone(frozen);model.train();optimizer.zero_grad(set_to_none=True);loss_sum=0.
        for batch,(x,y) in enumerate(loader,1):
            factor=min(1.,(step+1)/100)*(.05+.95*(1-step/total)**.9)
            optimizer.param_groups[0]['lr']=0 if frozen else a.lr*factor
            optimizer.param_groups[1]['lr']=a.head_lr*factor
            x,y=x.cuda(non_blocking=True),y.cuda(non_blocking=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits=F.interpolate(model(x),size=y.shape[-2:],mode='bilinear',align_corners=False)
                loss=segmentation_loss(logits,y,.5,weights)+.3*lovasz_softmax(logits,y)
            if not torch.isfinite(loss):raise RuntimeError('Non-finite loss')
            start_group=((batch-1)//a.accum)*a.accum;divisor=min(a.accum,len(loader)-start_group)
            (loss/divisor).backward()
            if batch%a.accum==0 or batch==len(loader):
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not torch.isfinite(norm):raise RuntimeError('Non-finite gradients')
                optimizer.step();optimizer.zero_grad(set_to_none=True);updates+=1;update_ema(ema,model,updates)
            loss_sum+=loss.item();step+=1
            if batch==1 or batch%100==0:print(f'epoch={epoch}/{a.epochs} frozen={frozen} batch={batch}/{len(loader)} loss={loss_sum/batch:.4f} elapsed={time.time()-start:.0f}s peakGB={torch.cuda.max_memory_allocated()/1e9:.2f}',flush=True)
            if a.smoke and batch==2:
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):assert torch.isfinite(ema(x)).all()
                print('SMOKE_OK',flush=True);return
        metrics=evaluate(ema,val)
        row={'epoch':epoch,'frozen_backbone':frozen,'loss':loss_sum/len(loader),'elapsed_seconds':time.time()-start,**metrics}
        with (out/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
        state={'model':ema.state_dict(),'epoch':epoch,'metrics':metrics,'config':vars(a),'split':split,'ema_updates':updates,'architecture':'dinov3_pyramid128'}
        if metrics['mIoU_present_nonignored']>best:best=metrics['mIoU_present_nonignored'];torch.save(state,out/'best.pth')
        torch.save({**state,'student':model.state_dict(),'optimizer':optimizer.state_dict()},out/'last.pth')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['images','masks','source','out','split','class-balance']:p.add_argument('--'+name,required=True)
    p.add_argument('--init');p.add_argument('--epochs',type=int,default=8);p.add_argument('--freeze-epochs',type=int,default=1)
    p.add_argument('--size',type=int,default=512);p.add_argument('--batch',type=int,default=4);p.add_argument('--accum',type=int,default=2)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--seed',type=int,default=20260927)
    p.add_argument('--lr',type=float,default=1e-5);p.add_argument('--head-lr',type=float,default=3e-4)
    p.add_argument('--smoke',action='store_true');train(p.parse_args())
