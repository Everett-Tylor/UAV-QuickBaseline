"""Export test2 using the selected tuned UNetFormer and exact inference recipe."""
import argparse,json
from pathlib import Path
import torch
from tune import make_model,export

def main():
    p=argparse.ArgumentParser()
    for n in ('run','images','out'):p.add_argument('--'+n,required=True)
    a=p.parse_args();run=Path(a.run);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
    model=make_model();model.load_state_dict(torch.load(run/'best.pth',map_location='cpu',weights_only=True)['model'])
    mode=json.loads((run/'selection.json').read_text(encoding='utf-8'))['selected']['mode']
    export(model.cuda().eval(),a.images,out,mode)

if __name__=='__main__':main()
