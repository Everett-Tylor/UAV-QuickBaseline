"""Frozen experiment launcher; reads only mounted official data."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def run(args):
    subprocess.run([sys.executable,*map(str,args)],check=True)
def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['check','train','ablation','eval','predict'])
    p.add_argument('--data',type=Path,default=Path('/data'))
    p.add_argument('--out',type=Path,default=Path('/out'))
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    manifest=json.loads((ROOT/'MANIFEST.json').read_text(encoding='utf-8'))
    for path,sha in manifest['runtime_sha256'].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=sha:
            raise RuntimeError('File checksum mismatch: '+path)
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU unavailable; use --gpus all and a compatible NVIDIA driver')
    if a.command=='check':
        sys.path.insert(0,str(ROOT/'src'))
        from round2seg import Segmenter
        model=Segmenter(str(ROOT/'weights/final'),config_only=True).cuda().eval()
        state=torch.load(ROOT/'weights/final/best.pth',map_location='cpu',weights_only=False)
        model.load_state_dict(state['model'])
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            logits=model(torch.zeros(1,3,512,512,device='cuda'))
        assert logits.shape[1]==9 and torch.isfinite(logits).all()
        print(json.dumps({'check':'passed','torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),'logits_shape':list(logits.shape)})); return
    a.out.mkdir(parents=True,exist_ok=True)
    split=ROOT/'metadata/split.json'; initial=ROOT/'weights/initial'
    if a.command in ('train','ablation'):
        common=[ROOT/'src/round2seg.py','--images',a.data/'train_images','--masks',a.data/'train_masks','--split',split,'--class-balance',ROOT/'metadata/class_balance.json','--source',initial,'--batch','8','--accum','1','--workers',a.workers]
        base=a.out/'base'; control=a.out/'control'; mix=a.out/'mixstyle'
        if a.command=='train':
            if any((d/'history.jsonl').exists() for d in [base,control]): raise RuntimeError('Use an empty output directory for a new training run')
            cmd=common+['--out',base,'--epochs','6','--lr','0.00006','--seed','2026']
            if a.smoke: cmd+=['--smoke']
            run(cmd)
            if a.smoke: return
            run(common+['--out',control,'--init',base/'best.pth','--epochs','2','--lr','0.00002','--seed','2027','--mixstyle','0'])
        else:
            if not (base/'best.pth').exists(): raise RuntimeError('Run train first, or supply an output directory with base/best.pth')
            if (mix/'history.jsonl').exists(): raise RuntimeError('MixStyle output already exists')
            run(common+['--out',mix,'--init',base/'best.pth','--epochs','2','--lr','0.00002','--seed','2027','--mixstyle','0.2'])
        return
    checkpoint=a.checkpoint or ROOT/'weights/final/best.pth'
    cmd=[ROOT/'src/round2_predict.py','--checkpoint',checkpoint,'--source',ROOT/'weights/final','--size','512']
    if a.command=='eval':
        cmd+=['--images',a.data/'train_images','--masks',a.data/'train_masks','--split',split,'--out',a.out/'validation_512.json']
    else:
        test=a.data/'test2_images'
        if len(list(test.glob('*.png')))!=1300: raise RuntimeError('Expected 1300 PNG images directly under data/test2_images')
        cmd+=['--images',test,'--out',a.out/'submission_round2_1300']
    run(cmd)
if __name__=='__main__': main()
