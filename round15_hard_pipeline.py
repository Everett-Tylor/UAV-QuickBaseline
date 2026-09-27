"""Mine training-only barren confusions, fine-tune one checkpoint, validate/export."""
import json,subprocess,sys,time,hashlib
from pathlib import Path

def main():
    root=Path.cwd();code=root/'outputs/UAV-QuickBaseline';out=root/'outputs/round15_hard';out.mkdir(exist_ok=True)
    def status(stage,**kw):
        data={'stage':stage,'time':time.strftime('%Y-%m-%d %H:%M:%S'),**kw}
        (out/'status.json').write_text(json.dumps(data,indent=2));print(json.dumps(data),flush=True)
    def run(script,args,log):
        with (out/log).open('w') as f:subprocess.run([sys.executable,'-u',str(code/script),*map(str,args)],stdout=f,stderr=subprocess.STDOUT,check=True)
    try:
        cfg=json.loads((root/'outputs/runs/round8_mixstyle640/config.json').read_text())
        teacher=root/'outputs/runs/round12_pseudo640/best.pth';runout=root/'outputs/runs/round15_hard640'
        crops=out/'crops';smoke=root/'outputs/runs/round15_hard640_smoke'
        if any(p.exists() for p in [crops,runout,smoke]):raise RuntimeError('Existing outputs; inspect before restarting')
        common=sum(([ '--'+k.replace('_','-'),cfg[k]] for k in ['images','masks','source','split','class_balance']),[])
        mining=sum((['--'+k,cfg[k]] for k in ['images','masks','source','split']),[])
        status('mining_training_errors',teacher=str(teacher))
        run('barren_hard.py',mining+['--checkpoint',teacher,'--out',crops],'mining.log')
        args=common+['--init',teacher,'--hard-manifest',crops/'manifest.json','--epochs',3,'--freeze-epochs',0,'--size',640,'--batch',4,'--accum',2,'--lr',5e-7,'--head-lr',5e-6,'--seed',20261006,'--mixstyle']
        status('smoke');run('dino_train.py',args+['--out',smoke,'--smoke'],'smoke.log')
        status('training');run('dino_train.py',args+['--out',runout],'training.log')
        infer=['--source',cfg['source'],'--checkpoint',runout/'best.pth','--sizes',512,640,768,896,'--hflip','--brightness-floor',.35,'--max-brightening',1.25]
        status('validating');report=out/'validation.json'
        run('dino_predict.py',infer+['--images',cfg['images'],'--masks',cfg['masks'],'--split',cfg['split'],'--profiles',root/'outputs/round3/image_profiles.json','--out',report],'validation.log')
        current=json.loads(report.read_text());base=json.loads((root/'outputs/round12_pseudo/validation.json').read_text())
        scores={k:v['mIoU_present_nonignored'] for k,v in current['groups'].items()}
        delta={k:scores[k]-base['groups'][k]['mIoU_present_nonignored'] for k in scores}
        barren=current['groups']['all']['per_class_IoU']['5'];summary={'scores':scores,'delta_vs_round12':delta,'barren_IoU':barren,'official_score':None}
        (out/'selection.json').write_text(json.dumps(summary,indent=2))
        if delta['all']<0 or barren<base['groups']['all']['per_class_IoU']['5'] or min(delta.values())<-.002:
            status('no_validated_improvement',**summary);return
        status('predicting',**summary);dest=root/'outputs/submission_round15_hard_single'
        run('dino_predict.py',infer+['--images',root/'work/round2_data/images','--out',dest],'prediction.log')
        archive=dest.with_suffix('.zip');summary.update(zip=str(archive),sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
        (out/'selection.json').write_text(json.dumps(summary,indent=2));status('complete',**summary)
    except Exception as e:status('failed',error=str(e));raise

if __name__=='__main__':main()
