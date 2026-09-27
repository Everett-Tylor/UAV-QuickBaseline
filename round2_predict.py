"""Native-resolution validation or submission with a trained SegFormer."""
import argparse
import json
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from round2seg import Segmenter, ValidationDataset, evaluate
from improvedseg import paired_paths, split_pairs, tensor_image
from quickseg import images


def main(a):
    torch.set_num_threads(4)
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    size=a.size or checkpoint['config']['size']
    model=Segmenter(a.source,config_only=True).cuda()
    model.load_state_dict(checkpoint['model']); model.eval()
    if a.masks:
        pairs=paired_paths(a)
        _,val,_=split_pairs(pairs,a.split,.1,2026)
        metrics=evaluate(model,DataLoader(ValidationDataset(val,size),batch_size=2,num_workers=4))
        Path(a.out).write_text(json.dumps(metrics,indent=2),encoding='utf-8')
        print(json.dumps(metrics)); return
    paths=images(a.images)
    if len({p.name for p in paths})!=len(paths): raise ValueError('Duplicate filenames')
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    with torch.inference_mode():
        for i,path in enumerate(paths,1):
            with Image.open(path) as im:
                shape=(im.height,im.width)
                x=tensor_image(im.convert('RGB').resize((size,size),Image.Resampling.BILINEAR))[None].cuda()
            with torch.autocast('cuda',dtype=torch.float16): logits=model(x)
            pred=F.interpolate(logits.float(),size=shape,mode='bilinear',align_corners=False).argmax(1)[0].cpu().numpy().astype(np.uint8)
            Image.fromarray(pred).save(out/(path.stem+'.png'))
            if i%100==0: print(f'prediction={i}/{len(paths)}',flush=True)
    expected={p.stem+'.png' for p in paths}
    if {p.name for p in out.glob('*.png')}!=expected: raise ValueError('Output filename mismatch')
    archive=out.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for path in paths:
            prediction=out/(path.stem+'.png')
            with Image.open(prediction) as im,Image.open(path) as src:
                ids=np.asarray(im)
                if im.mode!='L' or im.size!=src.size or ids.max()>8: raise ValueError(str(prediction))
            z.write(prediction,prediction.name)
    print(f'VALIDATED {len(paths)} predictions: {archive}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('checkpoint','source','images','out'): p.add_argument('--'+name,required=True)
    p.add_argument('--size',type=int); p.add_argument('--masks'); p.add_argument('--split')
    main(p.parse_args())
