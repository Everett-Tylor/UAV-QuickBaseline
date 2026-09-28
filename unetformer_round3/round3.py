"""Round 3: controlled UNetFormer continuation, rare-image sampling and hard pixels."""
import argparse,copy,hashlib,json,math,random,time,zipfile
from pathlib import Path
import numpy as np
from PIL import Image,ImageEnhance
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,WeightedRandomSampler
from concurrent.futures import ThreadPoolExecutor
from unetformer_pipeline import (Pairs,make_model,forward,loss_fn,loader,metrics,
                                evaluate,save_json,tensor,LABELS)

def sampling_weights(names,masks):
    def one(name):
        with Image.open(Path(masks)/name) as im:y=np.asarray(im)
        barren=float((y==5).mean());vehicle=float((y==8).mean())
        return 1.+1.5*(barren>.01)+.5*(vehicle>.005)
    with ThreadPoolExecutor(max_workers=6) as pool:return list(pool.map(one,names))

def hard_pixel_loss(logits,y,weights):
    if isinstance(logits,tuple):
        return hard_pixel_loss(logits[0],y,weights)+.4*hard_pixel_loss(logits[1],y,weights)
    z=F.interpolate(logits.float(),size=y.shape[-2:],mode='bilinear',align_corners=False)
    valid=y!=0
    if not valid.any():return z.sum()*0
    per_pixel=F.cross_entropy(z,y,weight=weights,ignore_index=0,reduction='none')[valid]
    # Blend hard-pixel supervision with ordinary CE+Dice to retain easy context.
    count=max(1,per_pixel.numel()//2)
    hard=per_pixel.topk(count,sorted=False).values.mean()
    return .75*loss_fn(logits,y,weights)+.25*hard

@torch.inference_mode()
def evaluate_hv(model,val):
    model.eval();hist=torch.zeros(81,dtype=torch.int64,device='cuda')
    for x,y in val:
        x,y=x.cuda(non_blocking=True),y.cuda(non_blocking=True)
        total=None
        for dims in [[],[-1],[-2],[-1,-2]]:
            with torch.autocast('cuda',dtype=torch.float16):z=forward(model,x.flip(dims) if dims else x).float()
            if dims:z=z.flip(dims)
            total=z if total is None else total+z
        pred=F.interpolate(total/4,size=y.shape[-2:],mode='bilinear',align_corners=False).argmax(1)
        valid=y!=0;hist+=torch.bincount(9*y[valid]+pred[valid],minlength=81)
    return metrics(hist.cpu().reshape(9,9))

class AugmentedPairs(Pairs):
    def __init__(self,*args,strength=.15,**kwargs):
        super().__init__(*args,**kwargs);self.strength=strength
    def __getitem__(self,i):
        name=self.names[i]
        with Image.open(self.images/name) as im:x=im.convert('RGB')
        with Image.open(self.masks/name) as im:y=im.copy()
        x=x.resize((self.size,self.size),Image.Resampling.BILINEAR)
        y=y.resize((self.size,self.size),Image.Resampling.NEAREST)
        for tr in (Image.Transpose.FLIP_LEFT_RIGHT,Image.Transpose.FLIP_TOP_BOTTOM):
            if random.random()<.5:x,y=x.transpose(tr),y.transpose(tr)
        k=random.randrange(4)
        if k:x,y=x.rotate(90*k),y.rotate(90*k)
        for cls in (ImageEnhance.Brightness,ImageEnhance.Contrast,ImageEnhance.Color):
            x=cls(x).enhance(random.uniform(1-self.strength,1+self.strength))
        return tensor(x),torch.from_numpy(np.asarray(y,dtype=np.int64).copy())

def freeze_bn(model):
    for m in model.modules():
        if isinstance(m,nn.modules.batchnorm._BatchNorm):m.eval()

@torch.no_grad()
def update_ema(ema,model,decay=.995):
    for e,m in zip(ema.parameters(),model.parameters()):e.lerp_(m,1-decay)
    for e,m in zip(ema.buffers(),model.buffers()):e.copy_(m)

def run_trial(a,recipe,split,weights,baseline_score,sample_weights):
    out=Path(a.out)/recipe['name'];out.mkdir()
    save_json(out/'recipe.json',recipe)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    model=make_model().cuda()
    model.load_state_dict(torch.load(a.init,map_location='cpu',weights_only=True)['model'])
    ema=copy.deepcopy(model).eval()
    for p in ema.parameters():p.requires_grad_(False)
    # Reducing the exponent reduces excess weighting of rare classes.
    w=torch.tensor(weights,device='cuda',dtype=torch.float32)
    if recipe['balanced']:
        w[1:]=w[1:].sqrt();w[1:]/=w[1:].mean()
    save_json(out/'weights.json',w.tolist())
    opt=torch.optim.AdamW([{'params':model.backbone.parameters(),'lr':recipe['lr']},
                          {'params':model.decoder.parameters(),'lr':recipe['lr']*5}],weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda');best=baseline_score;chosen=None;step=0;start=time.time()
    ds=AugmentedPairs(split['train'],Path(a.cache)/'images',Path(a.cache)/'masks',512,
                      strength=recipe['strength'])
    val=loader(Pairs(split['val'],a.images,a.masks,512),2,a.workers)
    batches=math.ceil(len(ds)/4);total=recipe['epochs']*batches
    for epoch in range(1,recipe['epochs']+1):
        ds.size=recipe['sizes'][(epoch-1)%len(recipe['sizes'])]
        generator=torch.Generator().manual_seed(a.seed+epoch)
        if recipe['sampling']:
            sampler=WeightedRandomSampler(sample_weights,len(ds),replacement=True,generator=generator)
            from unetformer_pipeline import seed_worker
            train_loader=DataLoader(ds,batch_size=4,sampler=sampler,num_workers=a.workers,
                pin_memory=True,worker_init_fn=seed_worker,generator=generator)
        else:
            train_loader=loader(ds,4,a.workers,True)
            train_loader.generator.manual_seed(a.seed+epoch)
        model.train();freeze_bn(model);opt.zero_grad(set_to_none=True);total_loss=0
        for i,(x,y) in enumerate(train_loader):
            factor=min(1.,(step+1)/50)*(.1+.9*(1-step/total)**.9)
            for g,lr in zip(opt.param_groups,[recipe['lr'],recipe['lr']*5]):g['lr']=lr*factor
            x,y=x.cuda(non_blocking=True),y.cuda(non_blocking=True)
            objective=hard_pixel_loss if recipe['hard_pixels'] else loss_fn
            with torch.autocast('cuda',dtype=torch.float16):loss=objective(forward(model,x),y,w)
            if not torch.isfinite(loss):raise RuntimeError('Non-finite loss')
            group=min(2,len(train_loader)-(i//2)*2)
            scaler.scale(loss/group).backward()
            if (i+1)%2==0 or i+1==len(train_loader):
                scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True);update_ema(ema,model)
            total_loss+=loss.item();step+=1
            if i%250==0:print(f"{recipe['name']} epoch={epoch} size={ds.size} batch={i+1}/{len(train_loader)} loss={total_loss/(i+1):.4f}",flush=True)
        del train_loader
        candidates=[('raw',model),('ema',ema)]
        for variant,net in candidates:
            result=evaluate_hv(net,val)
            row={'trial':recipe['name'],'epoch':epoch,'variant':variant,'train_size':ds.size,
                 'loss':total_loss/batches,'elapsed_seconds':time.time()-start,**result}
            with (out/'history.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
            print('validation '+json.dumps({k:row[k] for k in ['trial','epoch','variant','mIoU']}),flush=True)
            if result['mIoU']>best:
                best=result['mIoU'];chosen={'trial':recipe['name'],'epoch':epoch,'variant':variant,**result}
                torch.save({'model':net.state_dict(),'epoch':epoch,'size':512,'metrics':result,'recipe':recipe,'variant':variant},out/'best.pth')
        # Save restart evidence as well as the last inference checkpoint.
        torch.save({'model':model.state_dict(),'ema':ema.state_dict(),'optimizer':opt.state_dict(),
                    'scaler':scaler.state_dict(),'epoch':epoch,'recipe':recipe},out/'last_training.pth')
        save_json(Path(a.out)/'status.json',{'stage':'training','trial':recipe['name'],'epoch':epoch,'best_mIoU':best})
    save_json(out/'selection.json',{'selected':chosen,'baseline_mIoU':baseline_score})
    del model,ema,opt,val;torch.cuda.empty_cache()
    return (out/'best.pth',chosen) if chosen else (None,None)

MODES={
    '512_hvflip':{'sizes':[512],'flips':[(False,False),(True,False),(False,True),(True,True)]},
    '480_512_544_hvflip':{'sizes':[480,512,544],'flips':[(False,False),(True,False),(False,True),(True,True)]},
    '512_dihedral8':{'sizes':[512],'flips':[(False,False),(True,False),(False,True),(True,True)],'transpose':True},
}

@torch.inference_mode()
def infer(model,im,mode):
    result=None
    for size in mode['sizes']:
        x=tensor(im.resize((size,size),Image.Resampling.BILINEAR))[None].cuda()
        subtotal=None
        transposes=[False,True] if mode.get('transpose') else [False]
        for transpose in transposes:
            base=x.transpose(-1,-2) if transpose else x
            for h,v in mode['flips']:
                dims=([-1] if h else [])+([-2] if v else [])
                inp=base.flip(dims) if dims else base
                with torch.autocast('cuda',dtype=torch.float16):z=forward(model,inp).float()
                if dims:z=z.flip(dims)
                if transpose:z=z.transpose(-1,-2)
                subtotal=z if subtotal is None else subtotal+z
        z=F.interpolate(subtotal/(len(mode['flips'])*len(transposes)),size=(im.height,im.width),mode='bilinear',align_corners=False)
        result=z if result is None else result+z
    return (result/len(mode['sizes'])).argmax(1)[0]

@torch.inference_mode()
def eval_mode(model,a,names,mode):
    hist=torch.zeros(81,dtype=torch.int64,device='cuda')
    for i,name in enumerate(names,1):
        with Image.open(Path(a.images)/name) as im:pred=infer(model,im.convert('RGB'),mode)
        with Image.open(Path(a.masks)/name) as im:y=torch.tensor(np.asarray(im,dtype=np.int64),device='cuda')
        valid=y!=0;hist+=torch.bincount(9*y[valid]+pred[valid],minlength=81)
        if i%200==0:print('inference validation',i,len(names),flush=True)
    return metrics(hist.cpu().reshape(9,9))

def export(model,images,out,mode):
    images=Path(images);out=Path(out);dest=out/'predictions';dest.mkdir()
    paths=sorted(images.glob('*.png'))
    if {p.name for p in paths}!={f'test2_{i}.png' for i in range(1,1301)}:raise ValueError('Incomplete test2')
    for i,p in enumerate(paths,1):
        with Image.open(p) as im:pred=infer(model,im.convert('RGB'),mode).byte().cpu().numpy()
        Image.fromarray(pred).save(dest/p.name)
        if i%100==0:print('prediction',i,1300,flush=True)
    archive=out/'submission_unetformer_round3_test2.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in paths:z.write(dest/p.name,p.name)
    with zipfile.ZipFile(archive) as z:
        assert len(z.namelist())==1300 and z.testzip() is None
    with archive.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    save_json(out/'submission_manifest.json',{'count':1300,'file':archive.name,'sha256':digest,'inference':mode})

def main():
    p=argparse.ArgumentParser()
    for n in ['images','masks','cache','split','init','audit','test-images','out']:p.add_argument('--'+n,required=True)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--seed',type=int,default=20260930)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
    split=json.loads(Path(a.split).read_text(encoding='utf-8'))
    if len(set(split['train']+split['val']))!=len(split['train'])+len(split['val']):raise ValueError('Split overlap')
    audit=json.loads(Path(a.audit).read_text(encoding='utf-8'))
    if hashlib.sha256(Path(a.split).read_bytes()).hexdigest()!=audit['split_sha256']:raise ValueError('Class weights/split mismatch')
    save_json(out/'config.json',vars(a));save_json(out/'split.json',split)
    model=make_model().cuda();model.load_state_dict(torch.load(a.init,map_location='cpu',weights_only=True)['model'])
    val=loader(Pairs(split['val'],a.images,a.masks,512),2,a.workers)
    baseline=evaluate_hv(model,val);save_json(out/'baseline.json',baseline);del model,val
    print('baseline',baseline['mIoU'],flush=True)
    recipes=[
        {'name':'continuation','epochs':6,'lr':3e-6,'balanced':True,'strength':.1,'sizes':[512,448,576],'sampling':False,'hard_pixels':False},
        {'name':'rare_images','epochs':8,'lr':3e-6,'balanced':True,'strength':.1,'sizes':[512,448,576],'sampling':True,'hard_pixels':False},
        {'name':'hard_pixels','epochs':8,'lr':3e-6,'balanced':True,'strength':.1,'sizes':[512,448,576],'sampling':False,'hard_pixels':True},
    ]
    save_json(out/'recipes.json',recipes)
    sample_weights=sampling_weights(split['train'],Path(a.cache)/'masks')
    save_json(out/'sampling_audit.json',{'training_only':True,'count':len(sample_weights),
        'weight_counts':{str(w):sample_weights.count(w) for w in sorted(set(sample_weights))},
        'rule':'1 + 1.5*(barren_fraction>0.01) + 0.5*(vehicle_fraction>0.005)',
        'split_sha256':audit['split_sha256']})
    best_path=Path(a.init);best={'trial':'baseline',**baseline};trials=[]
    for recipe in recipes:
        ck,record=run_trial(a,recipe,split,audit['weights'],baseline['mIoU'],sample_weights)
        trials.append({'name':recipe['name'],'best':record})
        if record and record['mIoU']>best['mIoU']:best_path=ck;best=record
    state=torch.load(best_path,map_location='cpu',weights_only=True)
    # Weight-space averaging keeps a single inference model; compare on the same holdout.
    prior=torch.load(a.init,map_location='cpu',weights_only=True)['model']
    current=copy.deepcopy(state['model']);averages=[]
    model=make_model().cuda();val=loader(Pairs(split['val'],a.images,a.masks,512),2,a.workers)
    for old_fraction in [.25,.5]:
        mixed={k:(v*(1-old_fraction)+prior[k]*old_fraction if v.is_floating_point() else v.clone()) for k,v in current.items()}
        model.load_state_dict(mixed);result=evaluate_hv(model,val)
        averages.append({'prior_weight':old_fraction,**result})
        if result['mIoU']>best['mIoU']:
            state={**state,'model':mixed,'averaging_prior_fraction':old_fraction};best={**best,**result,'averaging_prior_fraction':old_fraction}
    save_json(out/'weight_averaging.json',averages);del model,val
    torch.save(state,out/'best.pth')
    model=make_model().cuda();model.load_state_dict(state['model']);model.eval()
    candidates=[]
    for name,mode in MODES.items():
        result=eval_mode(model,a,split['val'],mode)
        candidates.append({'name':name,'mode':mode,**result});print('mode',name,result['mIoU'],flush=True)
    selected=max(candidates,key=lambda x:x['mIoU'])
    summary={'baseline':baseline,'trials':trials,'checkpoint_selection':best,
             'inference_candidates':candidates,'selected':selected,
             'delta_percentage_points':100*(selected['mIoU']-baseline['mIoU']),'official_score':None}
    save_json(out/'selection.json',summary)
    export(model,a.test_images,out,selected['mode'])
    save_json(out/'status.json',{'stage':'complete','mIoU':selected['mIoU'],
             'delta_percentage_points':summary['delta_percentage_points'],'official_score':None})

if __name__=='__main__':main()
