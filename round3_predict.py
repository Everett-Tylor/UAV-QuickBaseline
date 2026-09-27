"""Multi-scale/flip evaluation and reproducible 1300-image submission export."""
import argparse,json,zipfile
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from round2seg import Segmenter
from improvedseg import tensor_image,metrics_from_hist,paired_paths,split_pairs
from quickseg import images

class Inputs(Dataset):
    def __init__(self,pairs,sizes):self.pairs,self.sizes=pairs,sizes
    def __len__(self):return len(self.pairs)
    def __getitem__(self,i):
        path,mask=self.pairs[i]
        with Image.open(path) as im:
            rgb=im.convert('RGB');shape=(rgb.height,rgb.width)
            xs=[tensor_image(rgb.resize((s,s),Image.Resampling.BILINEAR)) for s in self.sizes]
        if mask is not None:
            with Image.open(mask) as im:y=torch.from_numpy(np.asarray(im,dtype=np.int64).copy())
        else:y=torch.empty(0,dtype=torch.long)
        return xs,y,path.name,torch.tensor(shape)

@torch.inference_mode()
def run(a):
    torch.set_num_threads(4)
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    model=Segmenter(a.source,config_only=True).cuda().eval();model.load_state_dict(state['model'])
    if a.masks:
        _,pairs,_=split_pairs(paired_paths(a),a.split,.1,2026)
    else:pairs=[(p,None) for p in images(a.images)]
    if len({p.name for p,_ in pairs})!=len(pairs):raise ValueError('Duplicate image names')
    loader=DataLoader(Inputs(pairs,a.sizes),batch_size=1,num_workers=a.workers,pin_memory=True)
    groups={};thresholds={}
    if a.profiles:
        rows=json.loads(Path(a.profiles).read_text())
        thresholds={k:float(np.quantile([r[k] for r in rows if r['group']=='train'],.25)) for k in ['brightness','contrast']}
        groups={r['name']:[k for k,v in thresholds.items() if r[k]<=v] for r in rows if r['group']=='val'}
    hist={k:torch.zeros(81,dtype=torch.int64,device='cuda') for k in ['all','brightness','contrast']}
    count={k:0 for k in hist}
    out=Path(a.out)
    if not a.masks:out.mkdir(parents=True,exist_ok=True)
    for i,(xs,y,names,shape) in enumerate(loader,1):
        name=names[0];h,w=map(int,shape[0]);total=None
        for x in xs:
            x=x.cuda(non_blocking=True)
            for flip in ([False,True] if a.hflip else [False]):
                with torch.autocast('cuda',dtype=torch.float16):logits=model(x.flip(-1) if flip else x)
                if flip:logits=logits.flip(-1)
                # Average probabilities after mapping every view into original coordinates.
                probs=F.interpolate(logits.float(),size=(h,w),mode='bilinear',align_corners=False).softmax(1)
                total=probs if total is None else total+probs
        pred=total.argmax(1)
        if a.masks:
            y=y.cuda(non_blocking=True);valid=y!=0
            matrix=torch.bincount(9*y[valid]+pred[valid],minlength=81)
            for group in ['all',*groups.get(name,[])]:hist[group]+=matrix;count[group]+=1
        else:Image.fromarray(pred[0].cpu().numpy().astype(np.uint8)).save(out/(Path(name).stem+'.png'))
        if i%100==0:print(f'images={i}/{len(pairs)}',flush=True)
    if a.masks:
        report={'checkpoint':a.checkpoint,'sizes':a.sizes,'hflip':a.hflip,'group_thresholds_from_train':thresholds,
            'groups':{k:{'images':count[k],**metrics_from_hist(v.cpu().reshape(9,9))} for k,v in hist.items() if count[k]}}
        out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps({k:round(v['mIoU_present_nonignored']*100,4) for k,v in report['groups'].items()}),flush=True)
    else:
        expected={p.stem+'.png' for p,_ in pairs}
        if {p.name for p in out.glob('*.png')}!=expected:raise ValueError('Filename mismatch')
        with zipfile.ZipFile(out.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED) as z:
            for path,_ in pairs:
                p=out/(path.stem+'.png')
                with Image.open(p) as im,Image.open(path) as src:
                    ids=np.asarray(im)
                    if im.mode!='L' or im.size!=src.size or ids.max()>8:raise ValueError(str(p))
                z.write(p,p.name)
        with zipfile.ZipFile(out.with_suffix('.zip')) as z:assert z.testzip() is None
        print(f'VALIDATED {len(pairs)} PNGs: {out.with_suffix(".zip")}',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('checkpoint','source','images','out'):p.add_argument('--'+name,required=True)
    p.add_argument('--sizes',type=int,nargs='+',default=[640]);p.add_argument('--hflip',action='store_true')
    p.add_argument('--masks');p.add_argument('--split');p.add_argument('--profiles')
    p.add_argument('--workers',type=int,default=4)
    run(p.parse_args())
