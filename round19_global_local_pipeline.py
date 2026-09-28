"""One shared DINOv3:640 context and four native512 tiles, learned feature fusion."""
import json,subprocess,sys,time,hashlib
from pathlib import Path

def main():
    root=Path.cwd();code=root/'outputs/UAV-QuickBaseline';out=root/'outputs/round19_global_local';out.mkdir(exist_ok=True)
    def status(stage,**kw):
        d={'stage':stage,'time':time.strftime('%Y-%m-%d %H:%M:%S'),**kw}
        (out/'status.json').write_text(json.dumps(d,indent=2));print(json.dumps(d),flush=True)
    def run(script,args,log):
        with (out/log).open('w') as f:subprocess.run([sys.executable,'-u',str(code/script),*map(str,args)],stdout=f,stderr=subprocess.STDOUT,check=True)
    try:
        cfg=json.loads((root/'outputs/runs/round17_control640/config.json').read_text())
        teacher=root/'outputs/runs/round17_control640/best.pth';dest=root/'outputs/runs/round19_global_local';smoke=root/'outputs/runs/round19_global_local_smoke'
        if dest.exists() or smoke.exists():raise RuntimeError('Fresh run directories required')
        args=sum((['--'+k.replace('_','-'),cfg[k]] for k in ['images','masks','source','split','class_balance']),[])
        args+=['--init',teacher,'--epochs',3,'--freeze-epochs',0,'--size',1024,'--batch',1,'--accum',8,'--lr',5e-7,'--head-lr',5e-6,'--new-head-lr',5e-5,'--head-variant','global_local','--seed',20261010]
        status('smoke');run('dino_train.py',args+['--out',smoke,'--smoke'],'smoke.log')
        status('training');run('dino_train.py',args+['--out',dest],'training.log')
        infer=['--source',cfg['source'],'--checkpoint',dest/'best.pth','--sizes',1024,'--hflip','--brightness-floor',.35,'--max-brightening',1.25]
        status('validating');report=out/'validation.json'
        run('dino_predict.py',infer+['--images',cfg['images'],'--masks',cfg['masks'],'--split',cfg['split'],'--profiles',root/'outputs/round3/image_profiles.json','--out',report],'validation.log')
        cur=json.loads(report.read_text())['groups'];base=json.loads((root/'outputs/round17_contrast/control_validation.json').read_text())['groups']
        summary={'scores':{k:v['mIoU_present_nonignored'] for k,v in cur.items()},'barren_IoU':cur['all']['per_class_IoU']['5'],'official_score':None,'baseline_inference':'round17 four-scale flip','candidate_inference':'one shared model global640+native512tiles, flip'}
        (out/'selection.json').write_text(json.dumps(summary,indent=2))
        if cur['all']['mIoU_present_nonignored']<=base['all']['mIoU_present_nonignored'] or cur['all']['per_class_IoU']['5']<base['all']['per_class_IoU']['5'] or any(cur[k]['mIoU_present_nonignored']<base[k]['mIoU_present_nonignored']-.002 for k in ['brightness','contrast']):
            status('no_validated_improvement',**summary);return
        status('predicting',**summary);output=root/'outputs/submission_round19_global_local_single'
        run('dino_predict.py',infer+['--images',root/'work/round2_data/images','--out',output],'prediction.log')
        archive=output.with_suffix('.zip');status('complete',zip=str(archive),sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),**summary)
    except Exception as e:status('failed',error=str(e));raise

if __name__=='__main__':main()
