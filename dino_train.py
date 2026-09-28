"""Train a local DINOv3 encoder and pyramid decoder on the fixed UAV split."""
import argparse,copy,json,random,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from dinoseg import DinoSegmenter,load_dino_weights
from improvedseg import CropDataset,paired_paths,split_pairs,seed_worker,segmentation_loss,metrics_from_hist
from round2seg import ValidationDataset
from round4_train import lovasz_softmax,update_ema,RobustDataset

class MixedScaleDataset(torch.utils.data.Dataset):
    """Keep scene context in half the samples and native-scale detail in the rest."""
    def __init__(self,pairs,size):
        self.full=CropDataset(pairs,size,True,sampling='resize')
        self.crop=CropDataset(pairs,size,True,sampling='crop')
    def __len__(self):return len(self.full)
    def __getitem__(self,index):
        return (self.crop if random.random()<.5 else self.full)[index]

def boundary_loss(logits,target):
    valid=target!=0
    edge=torch.zeros_like(valid)
    dh=(target[:,1:]!=target[:,:-1])&valid[:,1:]&valid[:,:-1]
    dw=(target[:,:,1:]!=target[:,:,:-1])&valid[:,:,1:]&valid[:,:,:-1]
    edge[:,1:]|=dh;edge[:,:-1]|=dh;edge[:,:,1:]|=dw;edge[:,:,:-1]|=dw
    edge=F.max_pool2d(edge[:,None].float(),3,1,1)[:,0]
    # Exclude ignored pixels and their neighbourhood, including the image border.
    invalid=F.pad((~valid)[:,None].float(),(1,1,1,1),value=1)
    keep=F.max_pool2d(invalid,3,1)[:,0]==0
    prediction=F.interpolate(logits.float(),size=target.shape[-2:],mode='bilinear',align_corners=False)[:,0]
    if not keep.any():return prediction.sum()*0
    labels=edge[keep];positive=labels.sum()
    weight=((labels.numel()-positive)/(positive+1)).clamp(1,10).detach()
    return F.binary_cross_entropy_with_logits(prediction[keep],labels,pos_weight=weight)

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
    if a.full_data:
        training=training+validation;validation=[]
        split={'train':[p.name for p,_ in training],'val':[],'mode':'full_data_no_holdout'}
    (out/'config.json').write_text(json.dumps(vars(a),indent=2));(out/'split.json').write_text(json.dumps(split))
    weights=torch.tensor(json.loads(Path(a.class_balance).read_text())['weights'],device='cuda')
    g=torch.Generator().manual_seed(a.seed)
    if a.augmentation=='robust':dataset=RobustDataset(training,a.size)
    elif a.augmentation=='mixed':dataset=MixedScaleDataset(training,a.size)
    else:dataset=CropDataset(training,a.size,True,sampling='resize')
    if a.hard_manifest:
        from barren_hard import HardDataset
        dataset=HardDataset(training,a.size,a.hard_manifest)
    sampler=None
    if a.barren_sampling>1:
        from PIL import Image
        fractions=[]
        for _,mask in training:
            with Image.open(mask) as im:fractions.append(float((np.asarray(im)==5).mean()))
        sample_weights=[a.barren_sampling if fraction>=.01 else 1. for fraction in fractions]
        sampler=torch.utils.data.WeightedRandomSampler(sample_weights,len(training),replacement=True,generator=g)
        (out/'barren_sampling.json').write_text(json.dumps({'threshold':.01,'multiplier':a.barren_sampling,'eligible':sum(f>=.01 for f in fractions),'total':len(training)},indent=2))
    loader=DataLoader(dataset,batch_size=a.batch,shuffle=sampler is None,sampler=sampler,generator=g,num_workers=a.workers,pin_memory=True,worker_init_fn=seed_worker,persistent_workers=a.workers>0)
    pseudo_loader=None
    if a.pseudo_manifest:
        from dino_pseudo import load_pseudo_pairs,digest
        pseudo_pairs,manifest=load_pseudo_pairs(a.pseudo_manifest,validation)
        if not a.init or digest(a.init)!=manifest['teacher_sha256']:raise ValueError('Student must initialize from the pseudo teacher checkpoint')
        labeled={p.resolve() for p,_ in training}
        if any(p.resolve() in labeled for p,_ in pseudo_pairs):raise ValueError('Labeled image repeated as pseudo data')
        pseudo_loader=DataLoader(CropDataset(pseudo_pairs,a.size,True,sampling='resize'),batch_size=a.pseudo_batch,
            shuffle=True,generator=torch.Generator().manual_seed(a.seed+1),num_workers=a.workers,pin_memory=True,
            worker_init_fn=seed_worker,persistent_workers=a.workers>0)
        pseudo_iterator=iter(pseudo_loader)
        (out/'pseudo_summary.json').write_text(json.dumps({'images':len(pseudo_pairs),'teacher_sha256':manifest['teacher_sha256'],
            'confidence':manifest['confidence'],'agreement':manifest['agreement'],'pseudo_weight':a.pseudo_weight},indent=2))
    val=DataLoader(ValidationDataset(validation,a.size),batch_size=2,num_workers=a.workers,pin_memory=True,persistent_workers=a.workers>0)
    model=DinoSegmenter(a.source,config_only=bool(a.init),head_variant=a.head_variant).cuda()
    if a.init:load_dino_weights(model,torch.load(a.init,map_location='cpu',weights_only=False),allow_head_upgrade=True)
    model.mixstyle_enabled=a.mixstyle
    model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    ema=copy.deepcopy(model).eval().requires_grad_(False)
    backbone=list(model.backbone.parameters());ids={id(p) for p in backbone}
    added=[p for name,p in model.named_parameters() if name.startswith(('context_adapter.','detail_adapter.','boundary_head.'))]
    added_ids={id(p) for p in added}
    decoder=[p for p in model.parameters() if id(p) not in ids and id(p) not in added_ids]
    groups=[{'params':backbone,'lr':a.lr},{'params':decoder,'lr':a.head_lr}]
    if added:groups.append({'params':added,'lr':a.new_head_lr})
    optimizer=torch.optim.AdamW(groups,weight_decay=.01,foreach=False)
    best=-1.;updates=0;step=0;total=a.epochs*len(loader);start=time.time()
    for epoch in range(1,a.epochs+1):
        frozen=epoch<=a.freeze_epochs;model.freeze_backbone(frozen);model.train();optimizer.zero_grad(set_to_none=True);loss_sum=0.;pseudo_sum=0.
        for batch,(x,y) in enumerate(loader,1):
            factor=min(1.,(step+1)/100)*(.05+.95*(1-step/total)**.9)
            optimizer.param_groups[0]['lr']=0 if frozen else a.lr*factor
            optimizer.param_groups[1]['lr']=a.head_lr*factor
            if added:optimizer.param_groups[2]['lr']=a.new_head_lr*factor
            x,y=x.cuda(non_blocking=True),y.cuda(non_blocking=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                if a.head_variant=='context_boundary':raw,edges=model(x,return_aux=True)
                else:raw=model(x)
                logits=F.interpolate(raw,size=y.shape[-2:],mode='bilinear',align_corners=False)
                loss=segmentation_loss(logits,y,.5,weights)+.3*lovasz_softmax(logits,y)
                if a.head_variant=='context_boundary':loss=loss+a.boundary_weight*boundary_loss(edges,y)
            if not torch.isfinite(loss):raise RuntimeError('Non-finite loss')
            start_group=((batch-1)//a.accum)*a.accum;divisor=min(a.accum,len(loader)-start_group)
            (loss/divisor).backward()
            if pseudo_loader is not None:
                try:px,py=next(pseudo_iterator)
                except StopIteration:
                    pseudo_iterator=iter(pseudo_loader);px,py=next(pseudo_iterator)
                px,py=px.cuda(non_blocking=True),py.cuda(non_blocking=True)
                if a.classmix:
                    from dino_classmix import classmix
                    count=min(len(x),len(px))
                    px,py,_=classmix(x[:count].detach(),y[:count],px[:count],py[:count])
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    plogits=F.interpolate(model(px),size=py.shape[-2:],mode='bilinear',align_corners=False)
                    ploss=segmentation_loss(plogits,py,0.,weights)
                if not torch.isfinite(ploss):raise RuntimeError('Non-finite pseudo loss')
                ramp=min(1.,(step+1)/len(loader))
                (a.pseudo_weight*ramp*ploss/divisor).backward();pseudo_sum+=ploss.item()
            if batch%a.accum==0 or batch==len(loader):
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not torch.isfinite(norm):raise RuntimeError('Non-finite gradients')
                optimizer.step();optimizer.zero_grad(set_to_none=True);updates+=1;update_ema(ema,model,updates)
            loss_sum+=loss.item();step+=1
            if batch==1 or batch%100==0:print(f'epoch={epoch}/{a.epochs} frozen={frozen} batch={batch}/{len(loader)} loss={loss_sum/batch:.4f} elapsed={time.time()-start:.0f}s peakGB={torch.cuda.max_memory_allocated()/1e9:.2f}',flush=True)
            if a.smoke and batch==2:
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):assert torch.isfinite(ema(x)).all()
                print('SMOKE_OK',flush=True);return
        metrics={} if a.full_data else evaluate(ema,val)
        row={'epoch':epoch,'frozen_backbone':frozen,'loss':loss_sum/len(loader),'pseudo_loss':pseudo_sum/len(loader),'elapsed_seconds':time.time()-start,**metrics}
        with (out/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
        state={'model':ema.state_dict(),'epoch':epoch,'metrics':metrics,'config':vars(a),'split':split,'ema_updates':updates,'architecture':'dinov3_'+a.head_variant}
        if a.full_data:
            if epoch==a.epochs:torch.save(state,out/'final.pth')
        elif metrics['mIoU_present_nonignored']>best:best=metrics['mIoU_present_nonignored'];torch.save(state,out/'best.pth')
        torch.save({**state,'student':model.state_dict(),'optimizer':optimizer.state_dict()},out/'last.pth')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['images','masks','source','out','split','class-balance']:p.add_argument('--'+name,required=True)
    p.add_argument('--init');p.add_argument('--epochs',type=int,default=8);p.add_argument('--freeze-epochs',type=int,default=1)
    p.add_argument('--size',type=int,default=512);p.add_argument('--batch',type=int,default=4);p.add_argument('--accum',type=int,default=2)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--seed',type=int,default=20260927)
    p.add_argument('--lr',type=float,default=1e-5);p.add_argument('--head-lr',type=float,default=3e-4)
    p.add_argument('--mixstyle',action='store_true')
    p.add_argument('--augmentation',choices=['mild','robust','mixed'],default='mild')
    p.add_argument('--head-variant',choices=['pyramid','context_boundary'],default='pyramid')
    p.add_argument('--new-head-lr',type=float,default=2e-4)
    p.add_argument('--boundary-weight',type=float,default=.1)
    p.add_argument('--pseudo-manifest');p.add_argument('--pseudo-batch',type=int,default=2)
    p.add_argument('--pseudo-weight',type=float,default=.25)
    p.add_argument('--classmix',action='store_true')
    p.add_argument('--barren-sampling',type=float,default=1.)
    p.add_argument('--hard-manifest')
    p.add_argument('--full-data',action='store_true')
    p.add_argument('--smoke',action='store_true');train(p.parse_args())
