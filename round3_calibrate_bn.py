"""Re-estimate EMA BatchNorm buffers on a fixed clean training-only subset."""
import argparse,json,random
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from round2seg import Segmenter
from round3_predict import Inputs

def main(a):
    torch.set_num_threads(4)
    split=json.loads(Path(a.split).read_text())
    if set(split['train'])&set(split['val']):raise ValueError('Overlapping split')
    names=list(split['train']);random.Random(20260923).shuffle(names);names=names[:a.samples]
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    if state['split']!=split:raise ValueError('Checkpoint split mismatch')
    model=Segmenter(a.source,config_only=True).cuda().eval();model.load_state_dict(state['model'])
    bns=[m for m in model.modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)]
    if not bns:raise ValueError('No BatchNorm modules to calibrate')
    for m in bns:m.reset_running_stats();m.momentum=None;m.train()
    loader=DataLoader(Inputs([(Path(a.images)/name,None) for name in names],[a.size]),batch_size=8,num_workers=4,pin_memory=True)
    with torch.inference_mode():
        for i,(xs,_,_,_) in enumerate(loader,1):
            with torch.autocast('cuda',dtype=torch.float16):model(xs[0].cuda(non_blocking=True))
            if i%25==0:print(f'calibration_batch={i}/{len(loader)}',flush=True)
    model.eval();state['model']=model.state_dict()
    if 'metrics' in state:state['precalibration_metrics']=state.pop('metrics')
    state['bn_calibration']={'images':names,'source_group':'train','size':a.size,'labels_used':False}
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);torch.save(state,out)
    print(f'Calibrated {len(bns)} BN modules with {len(names)} training images',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['images','split','source','checkpoint','out']:p.add_argument('--'+name,required=True)
    p.add_argument('--samples',type=int,default=1024);p.add_argument('--size',type=int,default=640)
    main(p.parse_args())
