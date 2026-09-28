"""Matched continuation control and regional contrast; single-checkpoint export."""
import json,subprocess,sys,time,hashlib
from pathlib import Path

def main():
    root=Path.cwd();code=root/'outputs/UAV-QuickBaseline';out=root/'outputs/round17_contrast';out.mkdir(exist_ok=True)
    def status(stage,**kw):
        d={'stage':stage,'time':time.strftime('%Y-%m-%d %H:%M:%S'),**kw}
        (out/'status.json').write_text(json.dumps(d,indent=2));print(json.dumps(d),flush=True)
    def run(script,args,log):
        with (out/log).open('w') as f:subprocess.run([sys.executable,'-u',str(code/script),*map(str,args)],stdout=f,stderr=subprocess.STDOUT,check=True)
    try:
        cfg=json.loads((root/'outputs/runs/round15_hard640/config.json').read_text())
        teacher=root/'outputs/runs/round15_hard640/best.pth'
        args=sum((['--'+k.replace('_','-'),cfg[k]] for k in ['images','masks','source','split','class_balance','hard_manifest']),[])
        args+=['--init',teacher,'--epochs',3,'--freeze-epochs',0,'--size',640,'--batch',4,'--accum',2,'--lr',5e-7,'--head-lr',5e-6,'--seed',20261008,'--mixstyle']
        smoke=root/'outputs/runs/round17_contrast_smoke'
        paths={n:root/('outputs/runs/round17_'+n+'640') for n in ['control','contrast']}
        if any(p.exists() for p in [smoke,*paths.values()]):raise RuntimeError('Fresh directories required')
        status('smoke');run('dino_train.py',args+['--contrast-weight',.03,'--out',smoke,'--smoke'],'smoke.log')
        reports={}
        for name,weight in [('contrast',.03),('control',0.)]:
            status('training',experiment=name);run('dino_train.py',args+['--contrast-weight',weight,'--out',paths[name]],name+'_training.log')
            infer=['--source',cfg['source'],'--checkpoint',paths[name]/'best.pth','--sizes',512,640,768,896,'--hflip','--brightness-floor',.35,'--max-brightening',1.25]
            status('validating',experiment=name);report=out/(name+'_validation.json')
            run('dino_predict.py',infer+['--images',cfg['images'],'--masks',cfg['masks'],'--split',cfg['split'],'--profiles',root/'outputs/round3/image_profiles.json','--out',report],name+'_validation.log')
            reports[name]=json.loads(report.read_text())['groups']
        reports['round15']=json.loads((root/'outputs/round15_hard/validation.json').read_text())['groups']
        (out/'comparison.json').write_text(json.dumps(reports,indent=2))
        c=reports['contrast'];valid=all(c['all']['mIoU_present_nonignored']>=reports[b]['all']['mIoU_present_nonignored'] and c['all']['per_class_IoU']['5']>reports[b]['all']['per_class_IoU']['5'] for b in ['control','round15'])
        valid=valid and all(c[g]['mIoU_present_nonignored']>=reports['round15'][g]['mIoU_present_nonignored']-.002 for g in ['brightness','contrast'])
        if not valid:status('no_validated_improvement');return
        status('predicting');dest=root/'outputs/submission_round17_contrast_single'
        run('dino_predict.py',['--source',cfg['source'],'--checkpoint',paths['contrast']/'best.pth','--sizes',512,640,768,896,'--hflip','--brightness-floor',.35,'--max-brightening',1.25,'--images',root/'work/round2_data/images','--out',dest],'prediction.log')
        archive=dest.with_suffix('.zip');status('complete',zip=str(archive),sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),official_score=None)
    except Exception as e:status('failed',error=str(e));raise

if __name__=='__main__':main()
