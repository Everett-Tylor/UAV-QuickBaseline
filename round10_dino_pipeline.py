"""Controlled single-checkpoint DINOv3 experiments; never ensemble candidates."""
import argparse,hashlib,json,subprocess,sys,time,zipfile
from pathlib import Path

def main(a):
    root=Path(a.workspace).resolve()
    code=root/'outputs/UAV-QuickBaseline'
    results=root/'outputs/round10_dino';results.mkdir(parents=True,exist_ok=True)
    config=json.loads((root/'outputs/runs/round8_mixstyle640/config.json').read_text())
    baseline=root/'outputs/runs/round8_mixstyle640/best.pth'
    status=results/('smoke_status.json' if a.smoke else 'status.json')
    def record(stage,**kw):
        data={'stage':stage,'time':time.strftime('%Y-%m-%d %H:%M:%S'),**kw}
        temporary=status.with_suffix('.tmp')
        temporary.write_text(json.dumps(data,indent=2),encoding='utf-8');temporary.replace(status)
        print(json.dumps(data),flush=True)
    def execute(script,args,log):
        with (results/log).open('w',encoding='utf-8') as f:
            subprocess.run([sys.executable,'-u',str(code/script),*map(str,args)],cwd=root,stdout=f,stderr=subprocess.STDOUT,check=True)
    def scores(report):
        return {k:v['mIoU_present_nonignored'] for k,v in report['groups'].items()}
    try:
        base_report=json.loads((root/'outputs/round8_mixstyle_mid_bright_tta.json').read_text())
        assert base_report['sizes']==[512,640,768,896] and base_report['hflip']
        assert base_report['brightness_floor']==.35 and base_report['max_brightening']==1.25
        assert base_report['ensemble_checkpoint'] is None
        baseline_scores=scores(base_report)
        inference=['--source',config['source'],'--sizes',512,640,768,896,'--hflip',
                   '--brightness-floor',.35,'--max-brightening',1.25]
        candidates=[]
        for name,augmentation,size,batch,accum in [
            ('robust640','robust',640,8,1),('mixed768','mixed',768,4,2)]:
            out=root/('outputs/runs/round10_dino_'+name+('_smoke' if a.smoke else ''))
            if out.exists():raise RuntimeError(f'Existing run must be inspected before restart: {out}')
            record('smoke' if a.smoke else 'training',candidate=name)
            args=[]
            for key in ['images','masks','source','split','class_balance']:
                args+=['--'+key.replace('_','-'),config[key]]
            args+=['--init',baseline,'--out',out,'--epochs',4,'--freeze-epochs',0,
                   '--size',size,'--batch',batch,'--accum',accum,'--seed',20261003,
                   '--lr',2e-6,'--head-lr',2e-5,'--mixstyle','--augmentation',augmentation]
            if a.smoke:args+=['--smoke']
            execute('dino_train.py',args,name+('_smoke.log' if a.smoke else '_train.log'))
            if a.smoke:continue
            record('validating',candidate=name)
            report=results/(name+'_validation.json')
            execute('dino_predict.py',inference+['--checkpoint',out/'best.pth',
                '--images',config['images'],'--masks',config['masks'],'--split',config['split'],
                '--profiles',root/'outputs/round3/image_profiles.json','--out',report],name+'_validation.log')
            data=json.loads(report.read_text())
            if data['ensemble_checkpoint'] is not None:raise RuntimeError('Unexpected ensemble')
            candidates.append({'name':name,'checkpoint':str(out/'best.pth'),'scores':scores(data)})
        if a.smoke:
            record('smoke_complete');return
        # Predeclared gate: preserve overall accuracy and avoid material dark/contrast regression.
        def objective(s):return .5*s['all']+.25*s['brightness']+.25*s['contrast']
        eligible=[c for c in candidates if c['scores']['all']>=baseline_scores['all']
                  and all(c['scores'][k]>=baseline_scores[k]-.001 for k in ['brightness','contrast'])
                  and objective(c['scores'])>objective(baseline_scores)]
        summary={'baseline':baseline_scores,'candidates':candidates,
                 'official_previous':68.45,'official_current':None,'single_checkpoint':True}
        if not eligible:
            record('no_validated_improvement',**summary)
            (results/'selection.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');return
        chosen=max(eligible,key=lambda c:objective(c['scores']))
        record('predicting',selected=chosen,**summary)
        output=root/'outputs/submission_round10_dino_single'
        execute('dino_predict.py',inference+['--checkpoint',chosen['checkpoint'],
            '--images',root/'work/round2_data/images','--out',output],'prediction.log')
        archive=output.with_suffix('.zip')
        with zipfile.ZipFile(archive) as z:
            assert len(z.namelist())==1300 and z.testzip() is None
        summary.update(selected=chosen,zip=str(archive),sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
        (results/'selection.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        record('complete',**summary)
    except Exception as error:
        record('failed',error=str(error));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workspace',default='.')
    p.add_argument('--smoke',action='store_true');main(p.parse_args())

