"""Generate filtered labels with ONE fixed teacher on explicitly supplied unlabeled data."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from dinoseg import DinoSegmenter
from dino_predict import Inputs
from quickseg import images

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def pixel_digest(path):
    with Image.open(path) as im:return hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()

def filter_labels(mean_probs,view_labels,confidence=.95,agreement=.75):
    score,label=mean_probs.max(1)
    votes=(view_labels==label.unsqueeze(0)).float().mean(0)
    keep=(score>=confidence)&(votes>=agreement)&(label!=0)
    return torch.where(keep,label,torch.zeros_like(label)),keep

def class_thresholds(hist,retain=.6,floor=.92,ceiling=.99):
    if hist.shape!=(9,1001) or not 0<retain<=1 or not 0<floor<=ceiling<=1:raise ValueError('Invalid calibration configuration')
    thresholds=np.full(9,ceiling,dtype=np.float64);thresholds[0]=1.
    for c in range(1,9):
        total=int(hist[c].sum())
        if total:
            count=max(1,int(np.ceil(total*retain)))
            offset=int(np.searchsorted(np.cumsum(hist[c,::-1]),count,side='left'))
            thresholds[c]=np.clip((1000-offset)/1000.,floor,ceiling)
    return thresholds

def apply_thresholds(labels,confidence_codes,agreed,thresholds):
    required=np.ceil(np.asarray(thresholds)*65535).astype(np.int64)
    keep=agreed&(labels!=0)&(confidence_codes.astype(np.int64)>=required[labels])
    return np.where(keep,labels,0).astype(np.uint8),keep

def load_pseudo_pairs(manifest_path,validation):
    manifest=json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    if manifest.get('complete') is not True:raise ValueError('Incomplete pseudo-label set')
    forbidden={pixel_digest(p) for p,_ in validation}
    result=[];seen=set()
    for row in manifest['images']:
        image,mask=Path(row['image']),Path(row['mask'])
        current=pixel_digest(image)
        if current in forbidden:raise ValueError(f'Validation image in pseudo training: {image}')
        if current in seen:raise ValueError(f'Duplicate pseudo image: {image}')
        seen.add(current)
        if current!=row['pixel_sha256'] or digest(mask)!=row['mask_sha256']:
            raise ValueError(f'Pseudo data changed: {image}')
        if row['coverage']>=manifest['min_coverage']:result.append((image,mask))
    if not result:raise ValueError('No pseudo images pass coverage threshold')
    return result,manifest

@torch.inference_mode()
def run(a):
    if not 0<a.confidence<=1 or not 0<a.agreement<=1:raise ValueError('Invalid thresholds')
    out=Path(a.out).resolve()
    if out.exists():raise ValueError('Use a fresh pseudo-label directory')
    paths=images(a.images)
    if not paths or len({p.stem for p in paths})!=len(paths):raise ValueError('Empty or duplicate input names')
    profiles=json.loads(Path(a.profiles).read_text(encoding='utf-8'))
    forbidden={r['sha256_pixels'] for r in profiles if r['group'] in ('train','val')}
    hashes={};seen=set()
    for p in paths:
        value=pixel_digest(p)
        if value in forbidden:raise ValueError(f'Labeled image supplied as unlabeled: {p}')
        if value in seen:raise ValueError(f'Duplicate pixels in unlabeled data: {p}')
        hashes[p.name]=value;seen.add(value)
    out.mkdir(parents=True)
    hist=np.zeros((9,1001),dtype=np.int64)
    if a.class_keep:
        if not 0<a.class_keep<=1 or not 0<a.confidence_floor<=a.confidence_ceiling<=1:raise ValueError('Invalid class threshold settings')
        (out/'confidence_cache').mkdir()
    torch.set_num_threads(4)
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    model=DinoSegmenter(a.source,config_only=True,head_variant=state.get('config',{}).get('head_variant','pyramid')).cuda().eval()
    model.load_state_dict(state['model'],strict=True)
    loader=DataLoader(Inputs([(p,None) for p in paths],a.sizes,.35,1.25),batch_size=1,num_workers=a.workers,pin_memory=True)
    rows=[];retained=np.zeros(9,dtype=np.int64);predicted=np.zeros(9,dtype=np.int64)
    for i,(xs,_,names,shape) in enumerate(loader,1):
        h,w=map(int,shape[0]);total=None;votes=[]
        for x in xs:
            x=x.cuda(non_blocking=True)
            for flip in (False,True):
                with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(x.flip(-1) if flip else x)
                if flip:logits=logits.flip(-1)
                probs=F.interpolate(logits.float(),size=(h,w),mode='bilinear',align_corners=False).softmax(1)
                total=probs if total is None else total+probs
                votes.append(probs.argmax(1).to(torch.uint8))
        mean=total/len(votes)
        if a.class_keep:
            score,raw=mean.max(1)
            agree=(torch.stack(votes)==raw.unsqueeze(0)).float().mean(0)>=a.agreement
            raw=raw[0].cpu().numpy().astype(np.uint8)
            codes=torch.floor(score[0]*65535).cpu().numpy().astype(np.uint16)
            agree=agree[0].cpu().numpy()
            predicted+=np.bincount(raw.reshape(-1),minlength=9)
            eligible=agree&(raw!=0)
            bins=codes[eligible].astype(np.int64)*1000//65535
            hist+=np.bincount(raw[eligible].astype(np.int64)*1001+bins,minlength=9009).reshape(9,1001)
            np.savez_compressed(out/'confidence_cache'/(paths[i-1].stem+'.npz'),labels=raw,confidence=codes,agreed=agree)
            if i%100==0 or i==len(paths):print(f'calibration={i}/{len(paths)}',flush=True)
            continue
        label,keep=filter_labels(mean,torch.stack(votes),a.confidence,a.agreement)
        ids=label[0].cpu().numpy().astype(np.uint8)
        raw=mean.argmax(1)[0].cpu().numpy()
        predicted+=np.bincount(raw.reshape(-1),minlength=9)
        retained+=np.bincount(ids[ids!=0],minlength=9)
        dest=out/(paths[i-1].stem+'.png');Image.fromarray(ids).save(dest)
        rows.append({'image':str(paths[i-1].resolve()),'mask':str(dest),'pixel_sha256':hashes[names[0]],
                     'mask_sha256':digest(dest),'coverage':float(keep.float().mean())})
        if i%100==0 or i==len(paths):print(f'pseudo={i}/{len(paths)}',flush=True)
    thresholds=None
    if a.class_keep:
        thresholds=class_thresholds(hist,a.class_keep,a.confidence_floor,a.confidence_ceiling)
        print(json.dumps({'class_thresholds':thresholds.tolist()}),flush=True)
        for p in paths:
            with np.load(out/'confidence_cache'/(p.stem+'.npz')) as cached:
                ids,keep=apply_thresholds(cached['labels'],cached['confidence'],cached['agreed'],thresholds)
            retained+=np.bincount(ids[ids!=0],minlength=9)
            dest=out/(p.stem+'.png');Image.fromarray(ids).save(dest)
            rows.append({'image':str(p.resolve()),'mask':str(dest),'pixel_sha256':hashes[p.name],
                         'mask_sha256':digest(dest),'coverage':float(keep.mean())})
    manifest={'complete':True,'teacher':str(Path(a.checkpoint).resolve()),'teacher_sha256':digest(a.checkpoint),
              'confidence':thresholds.tolist() if thresholds is not None else a.confidence,
              'filter_strategy':'class_adaptive' if a.class_keep else 'fixed',
              'class_keep':a.class_keep,'confidence_floor':a.confidence_floor,'confidence_ceiling':a.confidence_ceiling,
              'agreement':a.agreement,'min_coverage':a.min_coverage,
              'sizes':a.sizes,'hflip':True,'brightness_floor':.35,'max_brightening':1.25,
              'predicted_class_pixels':predicted.tolist(),'retained_class_pixels':retained.tolist(),'images':rows}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({'images':len(rows),'usable_images':sum(r['coverage']>=a.min_coverage for r in rows),
                      'mean_coverage':float(np.mean([r['coverage'] for r in rows])),
                      'retained_class_pixels':retained.tolist()}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('source','checkpoint','images','profiles','out'):p.add_argument('--'+key,required=True)
    p.add_argument('--sizes',type=int,nargs='+',default=[512,640,768,896])
    p.add_argument('--confidence',type=float,default=.95);p.add_argument('--agreement',type=float,default=.75)
    p.add_argument('--min-coverage',type=float,default=.02);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--class-keep',type=float,default=0.)
    p.add_argument('--confidence-floor',type=float,default=.92);p.add_argument('--confidence-ceiling',type=float,default=.99)
    run(p.parse_args())

