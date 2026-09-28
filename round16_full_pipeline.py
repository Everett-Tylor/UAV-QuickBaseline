"""Fixed-epoch full-labeled-data fine-tune; no held-out metrics or best selection."""
import json,subprocess,sys,time,hashlib,zipfile
from pathlib import Path

def main():
    root=Path.cwd();code=root/'outputs/UAV-QuickBaseline';out=root/'outputs/round16_full';out.mkdir(exist_ok=True)
    def status(stage,**kw):
        value={'stage':stage,'time':time.strftime('%Y-%m-%d %H:%M:%S'),'parent_official_score':68.75,'independent_validation':False,**kw}
        (out/'status.json').write_text(json.dumps(value,indent=2));print(json.dumps(value),flush=True)
    def run(script,args,log):
        with (out/log).open('w') as f:subprocess.run([sys.executable,'-u',str(code/script),*map(str,args)],stdout=f,stderr=subprocess.STDOUT,check=True)
    try:
        cfg=json.loads((root/'outputs/runs/round15_hard640/config.json').read_text())
        teacher=root/'outputs/runs/round15_hard640/best.pth';dest=root/'outputs/runs/round16_full640';smoke=root/'outputs/runs/round16_full640_smoke'
        if dest.exists() or smoke.exists():raise RuntimeError('Existing run directories; inspect before restart')
        args=sum((['--'+k.replace('_','-'),cfg[k]] for k in ['images','masks','source','split','class_balance','hard_manifest']),[])
        args+=['--init',teacher,'--full-data','--epochs',3,'--freeze-epochs',0,'--size',640,'--batch',4,'--accum',2,'--lr',5e-7,'--head-lr',5e-6,'--seed',20261007,'--mixstyle']
        (out/'protocol.json').write_text(json.dumps({'epochs':3,'initial_checkpoint':str(teacher),'initial_sha256':hashlib.sha256(teacher.read_bytes()).hexdigest(),'selection':'final epoch EMA only','training_images':6996,'heldout_images':0,'hard_crops':'reuse round15 training-only crops','official_score':None},indent=2))
        status('smoke');run('dino_train.py',args+['--out',smoke,'--smoke'],'smoke.log')
        split=json.loads((smoke/'split.json').read_text());assert len(split['train'])==6996 and len(set(split['train']))==6996 and not split['val']
        status('training',epochs=3,training_images=6996);run('dino_train.py',args+['--out',dest],'training.log')
        output=root/'outputs/submission_round16_full_single';status('predicting')
        run('dino_predict.py',['--source',cfg['source'],'--checkpoint',dest/'final.pth','--sizes',512,640,768,896,'--hflip','--brightness-floor',.35,'--max-brightening',1.25,'--images',root/'work/round2_data/images','--out',output],'prediction.log')
        archive=output.with_suffix('.zip')
        with zipfile.ZipFile(archive) as z:assert len(z.namelist())==1300 and z.testzip() is None
        status('complete',zip=str(archive),sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),official_score=None)
    except Exception as e:status('failed',error=str(e));raise

if __name__=='__main__':main()
