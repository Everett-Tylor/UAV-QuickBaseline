"""Mine true-label training errors; never mine the validation split."""
import argparse,json,random
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from improvedseg import paired_paths,split_pairs,CropDataset
from dino_predict import Inputs
from dinoseg import DinoSegmenter
from dino_pseudo import digest

def crop_box(error,side=768):
    h,w=error.shape
    if error.sum()<1024:return None
    # Highest-error 256px grid region, surrounded by native-resolution context.
    cells=[(int(error[t:t+256,l:l+256].sum()),t,l) for t in range(0,h,256) for l in range(0,w,256)]
    count,t,l=max(cells)
    if count<512:return None
    width,height=min(side,w),min(side,h)
    left=max(0,min(w-width,l+128-width//2));top=max(0,min(h-height,t+128-height//2))
    return left,top,left+width,top+height

class HardDataset(torch.utils.data.Dataset):
    def __init__(self,pairs,size,manifest):
        data=json.loads(Path(manifest).read_text());allowed={p.name for p,_ in pairs}
        if not data['complete'] or any(r['source'] not in allowed for r in data['crops']):raise ValueError('Hard crops outside training split')
        self.full=CropDataset(pairs,size,True,sampling='resize')
        self.hard={kind:CropDataset([(Path(r['image']),Path(r['mask'])) for r in data['crops'] if r['kind']==kind],size,True,sampling='resize') for kind in ['fp','fn']}
        if any(len(v)==0 for v in self.hard.values()):raise ValueError('Both FP and FN crops required')
    def __len__(self):return len(self.full)
    def __getitem__(self,index):
        if random.random()>=.2:return self.full[index]
        ds=self.hard['fp' if random.random()<2/3 else 'fn']
        return ds[random.randrange(len(ds))]

@torch.inference_mode()
def mine(a):
    torch.set_num_threads(4)
    train,val,_=split_pairs(paired_paths(a),a.split,.1,2026)
    assert not ({p.name for p,_ in train}&{p.name for p,_ in val})
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    model=DinoSegmenter(a.source,config_only=True).cuda().eval();model.load_state_dict(state['model'])
    loader=DataLoader(Inputs(train,[640],.35,1.25),batch_size=1,num_workers=4,pin_memory=True)
    sources={p.name:(p,m) for p,m in train};rows=[]
    for i,(xs,y,names,shape) in enumerate(loader,1):
        with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(xs[0].cuda())
        pred=F.interpolate(logits.float(),size=tuple(map(int,shape[0])),mode='bilinear',align_corners=False).argmax(1)[0].cpu().numpy()
        truth=y[0].numpy();path,mask=sources[names[0]]
        errors={'fp':(pred==5)&(truth!=5)&(truth!=0),'fn':(truth==5)&(pred!=5)}
        for kind,error in errors.items():
            box=crop_box(error)
            if box is None:continue
            ip=out/(path.stem+'_'+kind+'_rgb.png');mp=out/(path.stem+'_'+kind+'_mask.png')
            with Image.open(path) as im:im.convert('RGB').crop(box).save(ip)
            with Image.open(mask) as im:im.crop(box).save(mp)
            rows.append({'source':path.name,'kind':kind,'box':box,'image':str(ip.resolve()),'mask':str(mp.resolve()),'errors':int(error.sum())})
        if i%100==0:print(f'mined={i}/{len(train)} crops={len(rows)}',flush=True)
    (out/'manifest.json').write_text(json.dumps({'complete':True,'teacher_sha256':digest(a.checkpoint),'training_images':len(train),'crops':rows},indent=2))
    print('MINING_COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ['images','masks','split','source','checkpoint','out']:p.add_argument('--'+n,required=True)
    mine(p.parse_args())
