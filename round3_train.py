"""Higher-resolution supervised fine tuning with photometric augmentation and EMA."""
import argparse,copy,json,random,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageEnhance,ImageFilter
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from improvedseg import read_pair,tensor_image,paired_paths,split_pairs,seed_worker,segmentation_loss
from round2seg import Segmenter,ValidationDataset,evaluate

class RobustDataset(Dataset):
    def __init__(self,pairs,size): self.pairs,self.size=pairs,size
    def __len__(self): return len(self.pairs)
    def __getitem__(self,i):
        image,mask=read_pair(*self.pairs[i])
        if random.random()<.5:
            scale=random.uniform(.75,1.)
            w,h=round(image.width*scale),round(image.height*scale)
            x,y=random.randint(0,image.width-w),random.randint(0,image.height-h)
            box=(x,y,x+w,y+h);image,mask=image.crop(box),mask.crop(box)
        image=image.resize((self.size,self.size),Image.Resampling.BILINEAR)
        mask=mask.resize((self.size,self.size),Image.Resampling.NEAREST)
        for flip in (Image.Transpose.FLIP_LEFT_RIGHT,Image.Transpose.FLIP_TOP_BOTTOM):
            if random.random()<.5: image,mask=image.transpose(flip),mask.transpose(flip)
        rotation=random.choice([None,Image.Transpose.ROTATE_90,Image.Transpose.ROTATE_180,Image.Transpose.ROTATE_270])
        if rotation is not None: image,mask=image.transpose(rotation),mask.transpose(rotation)
        for enhancer,lo,hi in [(ImageEnhance.Brightness,.7,1.3),(ImageEnhance.Contrast,.7,1.3),(ImageEnhance.Color,.6,1.4)]:
            image=enhancer(image).enhance(random.uniform(lo,hi))
        if random.random()<.1: image=image.filter(ImageFilter.GaussianBlur(random.uniform(.2,.8)))
        x=tensor_image(image)
        if random.random()<.5: x=x.pow(random.uniform(.75,1.35))
        return x,torch.from_numpy(np.asarray(mask,dtype=np.int64).copy())

@torch.no_grad()
def update_ema(ema,student,updates):
    decay=min(.999,(1+updates)/(10+updates))
    for e,s in zip(ema.parameters(),student.parameters()): e.lerp_(s,1-decay)
    for e,s in zip(ema.buffers(),student.buffers()): e.copy_(s)

def train(a):
    torch.set_num_threads(4)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark=True
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if (out/'history.jsonl').exists(): raise ValueError('Use a fresh experiment directory')
    training,validation,split=split_pairs(paired_paths(a),a.split,.1,a.seed)
    (out/'split.json').write_text(json.dumps(split),encoding='utf-8')
    (out/'config.json').write_text(json.dumps(vars(a),indent=2),encoding='utf-8')
    weights=torch.tensor(json.loads(Path(a.class_balance).read_text())['weights'],device='cuda')
    g=torch.Generator().manual_seed(a.seed)
    loader=DataLoader(RobustDataset(training,a.size),batch_size=a.batch,shuffle=True,generator=g,
        num_workers=a.workers,pin_memory=True,worker_init_fn=seed_worker,persistent_workers=a.workers>0)
    val=DataLoader(ValidationDataset(validation,a.size),batch_size=2,num_workers=a.workers,
        pin_memory=True,persistent_workers=a.workers>0)
    student=Segmenter(a.source,config_only=True).cuda()
    student.load_state_dict(torch.load(a.init,map_location='cpu',weights_only=False)['model'])
    ema=copy.deepcopy(student).eval().requires_grad_(False)
    optimizer=torch.optim.AdamW([{'params':student.net.segformer.parameters(),'lr':a.lr},
        {'params':student.net.decode_head.parameters(),'lr':a.lr*10}],weight_decay=.02)
    scaler=torch.amp.GradScaler('cuda');best=-1.;step=0;updates=0
    started=time.time();total=a.epochs*len(loader)
    for epoch in range(1,a.epochs+1):
        student.train();optimizer.zero_grad(set_to_none=True);sum_loss=0.
        for batch,(x,y) in enumerate(loader,1):
            factor=min(1.,(step+1)/100)*(.05+.95*(1-step/total)**.9)
            for group,base in zip(optimizer.param_groups,[a.lr,a.lr*10]): group['lr']=base*factor
            x,y=x.cuda(non_blocking=True),y.cuda(non_blocking=True)
            with torch.autocast('cuda',dtype=torch.float16):
                logits=F.interpolate(student(x),size=y.shape[-2:],mode='bilinear',align_corners=False)
                loss=segmentation_loss(logits,y,.5,weights)
            if not torch.isfinite(loss): raise RuntimeError('Non-finite training loss')
            group_start=((batch-1)//a.accum)*a.accum
            divisor=min(a.accum,len(loader)-group_start)
            scaler.scale(loss/divisor).backward()
            if batch%a.accum==0 or batch==len(loader):
                scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(student.parameters(),1)
                old_scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scaler.get_scale()>=old_scale:
                    updates+=1;update_ema(ema,student,updates)
            sum_loss+=loss.item();step+=1
            if batch==1 or batch%100==0:
                print(f'epoch={epoch}/{a.epochs} batch={batch}/{len(loader)} loss={sum_loss/batch:.4f} elapsed={time.time()-started:.0f}s peakGB={torch.cuda.max_memory_allocated()/1e9:.2f}',flush=True)
            if a.smoke and batch==2:
                with torch.no_grad():
                    assert torch.isfinite(ema(x)).all()
                print('SMOKE_OK',flush=True);return
        metrics=evaluate(ema,val)
        row={'epoch':epoch,'loss':sum_loss/len(loader),'elapsed_seconds':time.time()-started,**metrics}
        with (out/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
        state={'model':ema.state_dict(),'epoch':epoch,'metrics':metrics,'config':vars(a),'split':split,'ema_updates':updates}
        if metrics['mIoU_present_nonignored']>best:
            best=metrics['mIoU_present_nonignored'];torch.save(state,out/'best.pth')
        torch.save({**state,'student':student.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict()},out/'last.pth')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('images','masks','split','class-balance','source','init','out'):p.add_argument('--'+name,required=True)
    p.add_argument('--epochs',type=int,default=6);p.add_argument('--size',type=int,default=640)
    p.add_argument('--batch',type=int,default=4);p.add_argument('--accum',type=int,default=2)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--seed',type=int,default=20260923)
    p.add_argument('--lr',type=float,default=1.5e-5);p.add_argument('--smoke',action='store_true')
    train(p.parse_args())
