"""Train and evaluate one context/boundary DINOv3 head, without ensembling."""
import argparse,json,subprocess,sys,time,zipfile,hashlib
from pathlib import Path

def main(a):
    root=Path(a.workspace).resolve();code=root/'outputs/UAV-QuickBaseline'
    results=root/'outputs/round11_head';results.mkdir(parents=True,exist_ok=True)
    status=results/('smoke_status.json' if a.smoke else 'status.json')
    def record(stage,**kw):
        data={'stage':stage,'time':time.strftime('%Y-%m-%d %H:%M:%S'),**kw}
        temp=status.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2),encoding='utf-8');temp.replace(status)
        print(json.dumps(data),flush=True)
    def run(script,args,log):
        with (results/log).open('w',encoding='utf-8') as f:
            subprocess.run([sys.executable,'-u',str(code/script),*map(str,args)],cwd=root,stdout=f,stderr=subprocess.STDOUT,check=True)
    try:
        cfg=json.loads((root/'outputs/runs/round8_mixstyle640/config.json').read_text())
        out=root/('outputs/runs/round11_context_boundary640'+('_smoke' if a.smoke else ''))
        if out.exists():raise RuntimeError(f'Inspect existing run before repeating: {out}')
        args=[]
        for key in ['images','masks','source','split','class_balance']:args+=['--'+key.replace('_','-'),cfg[key]]
        args+=['--init',root/'outputs/runs/round8_mixstyle640/best.pth','--out',out,
               '--head-variant','context_boundary','--epochs',6,'--freeze-epochs',0 if a.smoke else 1,
               '--size',640,'--batch',4,'--accum',2,'--lr',1e-6,'--head-lr',2e-5,
               '--new-head-lr',2e-4,'--boundary-weight',.1,'--mixstyle','--augmentation','mild','--seed',20261004]
        if a.smoke:args+=['--smoke']
        record('smoke' if a.smoke else 'training')
        run('dino_train.py',args,'smoke.log' if a.smoke else 'training.log')
        if a.smoke:record('smoke_complete');return
        infer=['--source',cfg['source'],'--checkpoint',out/'best.pth','--sizes',512,640,768,896,
               '--hflip','--brightness-floor',.35,'--max-brightening',1.25]
        report=results/'validation.json';record('validating')
        run('dino_predict.py',infer+['--images',cfg['images'],'--masks',cfg['masks'],'--split',cfg['split'],
            '--profiles',root/'outputs/round3/image_profiles.json','--out',report],'validation.log')
        current=json.loads(report.read_text());base=json.loads((root/'outputs/round8_mixstyle_mid_bright_tta.json').read_text())
        for k in ['sizes','hflip','brightness_floor','max_brightening']:
            if current[k]!=base[k]:raise RuntimeError('Baseline inference settings differ')
        if current['ensemble_checkpoint'] is not None:raise RuntimeError('Unexpected ensemble')
        baseline={k:v['mIoU_present_nonignored'] for k,v in base['groups'].items()}
        scores={k:v['mIoU_present_nonignored'] for k,v in current['groups'].items()}
        summary={'baseline':baseline,'scores':scores,'official_baseline':68.45,
                 'official_round10_robust640':68.36,'official_current':None,'checkpoint':str(out/'best.pth'),
                 'head_variant':'context_boundary','single_checkpoint':True}
        if scores['all']<=baseline['all'] or any(scores[k]<baseline[k]-.001 for k in ['brightness','contrast']):
            (results/'selection.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
            record('no_validated_improvement',**summary);return
        output=root/'outputs/submission_round11_head_single';record('predicting',**summary)
        run('dino_predict.py',infer+['--images',root/'work/round2_data/images','--out',output],'prediction.log')
        archive=output.with_suffix('.zip')
        with zipfile.ZipFile(archive) as z:assert len(z.namelist())==1300 and z.testzip() is None
        summary.update(zip=str(archive),sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
        (results/'selection.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');record('complete',**summary)
    except Exception as error:
        record('failed',error=str(error));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workspace',default='.')
    p.add_argument('--smoke',action='store_true');main(p.parse_args())

