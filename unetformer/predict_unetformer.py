"""Re-export test2 from the selected checkpoint and validation settings."""
import argparse,json
from pathlib import Path
import torch
from unetformer_pipeline import make_model,predict

def main():
    p=argparse.ArgumentParser()
    for name in ('run','images','out'):p.add_argument('--'+name,required=True)
    a=p.parse_args();run=Path(a.run);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
    model=make_model()
    state=torch.load(run/'best.pth',map_location='cpu',weights_only=True)
    model.load_state_dict(state['model'])
    config=json.loads((run/'validation.json').read_text(encoding='utf-8'))['selected']
    predict(model.cuda(),a.images,out,config['size'],config['hflip'])

if __name__=='__main__':main()
