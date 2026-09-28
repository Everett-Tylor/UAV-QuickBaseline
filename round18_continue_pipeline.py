"""Continue round17 control on unchanged holdout; export only after comparison."""
import json,subprocess,sys,time,hashlib
from pathlib import Path

def main():
    root=Path.cwd();code=root/'outputs/UAV-QuickBaseline';out=root/'outputs/round18_continue';out.mkdir(exist_ok=True)
    def status(stage,**kw):
        d={'stage':stage,'time':time.strftime('%Y-%m-%d %H:%M:%S'),**kw}
        (out/'status.json').write_text(json.dumps(d,indent=2));print(json.dumps(d),flush=True)
    def run(script,args,log):
        with (out/log).open('w') as f:subprocess.run([sys.executable,'-u',str(code/script),*map(str,args)],stdout=f,stderr=subprocess.STDOUT,check=True)
    try:
        cfg=json.loads((root/'outputs/runs/round17_control640/config.json').read_text())
        teacher=root/'outputs/runs/round17_control640/best.pth';dest=root/'outputs/runs/round18_continue640';smoke=root/'outputs/runs/round18_continue_smoke'
        if dest.exists() or smoke.exists():raise RuntimeError('Fresh run directories required')
        args=sum((['--'+k.replace('_','-'),cfg[k]] for k in ['images','masks','source','split','class_balance','hard_manifest']),[])
        args+=['--init',teacher,'--epochs',3,'--freeze-epochs',0,'--size',640,'--batch',4,'--accum',2,'--lr',5e-7,'--head-lr',5e-6,'--seed',20261009,'--mixstyle','--contrast-weight',0]
        (out/'protocol.json').write_text(json.dumps({'initial_checkpoint':str(teacher),'sha256':hashlib.sha256(teacher.read_bytes()).hexdigest(),'epochs':3,'training_images':6296,'validation_images':700,'contrast_weight':0,'official_score':None},indent=2))
        status('smoke');run('dino_train.py',args+['--out',smoke,'--smoke'],'smoke.log')
        status('training');run('dino_train.py',args+['--out',dest],'training.log')
        infer=['--source',cfg['source'],'--checkpoint',dest/'best.pth','--sizes',512,640,768,896,'--hflip','--brightness-floor',.35,'--max-brightening',1.25]
        status('validating');report=out/'validation.json'
        run('dino_predict.py',infer+['--images',cfg['images'],'--masks',cfg['masks'],'--split',cfg['split'],'--profiles',root/'outputs/round3/image_profiles.json','--out',report],'validation.log')
        cur=json.loads(report.read_text());base=json.loads((root/'outputs/round17_contrast/control_validation.json').read_text())
        for key in ['sizes','hflip','brightness_floor','max_brightening']:assert cur[key]==base[key]
        scores={k:v['mIoU_present_nonignored'] for k,v in cur['groups'].items()}
        delta={k:scores[k]-base['groups'][k]['mIoU_present_nonignored'] for k in scores}
        barren=cur['groups']['all']['per_class_IoU']['5']
        summary={'scores':scores,'delta_vs_round17_control':delta,'barren_IoU':barren,'official_score':None}
        (out/'selection.json').write_text(json.dumps(summary,indent=2))
        if delta['all']<=0 or barren<base['groups']['all']['per_class_IoU']['5'] or min(delta.values())<-.002:
            status('no_validated_improvement',**summary);return
        status('predicting',**summary);output=root/'outputs/submission_round18_continue_single'
        run('dino_predict.py',infer+['--images',root/'work/round2_data/images','--out',output],'prediction.log')
        archive=output.with_suffix('.zip');summary.update(zip=str(archive),sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
        (out/'selection.json').write_text(json.dumps(summary,indent=2));status('complete',**summary)
    except Exception as e:status('failed',error=str(e));raise

if __name__=='__main__':main()
