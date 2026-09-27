"""Test-set self training, authorized by the user's explicit competition-rule confirmation."""
import argparse,hashlib,json,subprocess,sys,time,zipfile
from pathlib import Path

def main(a):
    root=Path(a.workspace).resolve();code=root/'outputs/UAV-QuickBaseline'
    balanced=getattr(a,'class_balanced',False)
    experiment='round13_classbalanced' if balanced else 'round12_pseudo'
    results=root/('outputs/'+experiment);results.mkdir(parents=True,exist_ok=True)
    status=results/'status.json'
    def record(stage,**kw):
        value={'stage':stage,'time':time.strftime('%Y-%m-%d %H:%M:%S'),**kw}
        temp=status.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2),encoding='utf-8');temp.replace(status)
        print(json.dumps(value),flush=True)
    def run(script,args,log):
        with (results/log).open('w',encoding='utf-8') as f:
            subprocess.run([sys.executable,'-u',str(code/script),*map(str,args)],cwd=root,stdout=f,stderr=subprocess.STDOUT,check=True)
    try:
        cfg=json.loads((root/'outputs/runs/round8_mixstyle640/config.json').read_text())
        teacher=root/'outputs/runs/round8_mixstyle640/best.pth'
        pseudo=results/'masks'
        train_out=root/('outputs/runs/'+experiment+'640')
        smoke_out=root/('outputs/runs/'+experiment+'640_smoke')
        if any(p.exists() for p in [pseudo,train_out,smoke_out]):raise RuntimeError('Existing outputs require inspection before restarting')
        record('generating_pseudo_labels',teacher=str(teacher),rule_confirmation='User confirmed test_2 pseudo-label training is allowed')
        generation=['--source',cfg['source'],'--checkpoint',teacher,'--images',root/'work/round2_data/images',
            '--profiles',root/'outputs/round3/image_profiles.json','--out',pseudo]
        if balanced:generation+=['--class-keep',.6,'--confidence-floor',.92,'--confidence-ceiling',.99]
        run('dino_pseudo.py',generation,'generation.log')
        manifest=json.loads((pseudo/'manifest.json').read_text())
        usable=sum(r['coverage']>=manifest['min_coverage'] for r in manifest['images'])
        if len(manifest['images'])!=1300 or usable<100:raise RuntimeError('Insufficient pseudo coverage or unexpected image count')
        summary={'teacher_official_score':68.45,'official_current':None,'confidence':manifest['confidence'],
                 'agreement':manifest['agreement'],'usable_images':usable,'mean_coverage':sum(r['coverage'] for r in manifest['images'])/1300,
                 'retained_class_pixels':manifest['retained_class_pixels'],'single_checkpoint':True,
                 'test_set_self_training':True,'rule_confirmation':'User explicitly confirmed allowed',
                 'filter_strategy':manifest.get('filter_strategy','fixed')}
        args=[]
        for key in ['images','masks','source','split','class_balance']:args+=['--'+key.replace('_','-'),cfg[key]]
        args+=['--init',teacher,'--pseudo-manifest',pseudo/'manifest.json','--pseudo-weight',.25,'--pseudo-batch',2,
               '--epochs',3,'--freeze-epochs',0,'--size',640,'--batch',4,'--accum',2,
               '--lr',1e-6,'--head-lr',1e-5,'--seed',20261005,'--mixstyle','--augmentation','mild']
        record('smoke',**summary)
        run('dino_train.py',args+['--out',smoke_out,'--smoke'],'smoke.log')
        record('training',**summary)
        run('dino_train.py',args+['--out',train_out],'training.log')
        infer=['--source',cfg['source'],'--checkpoint',train_out/'best.pth','--sizes',512,640,768,896,
               '--hflip','--brightness-floor',.35,'--max-brightening',1.25]
        record('validating',**summary)
        report=results/'validation.json'
        run('dino_predict.py',infer+['--images',cfg['images'],'--masks',cfg['masks'],'--split',cfg['split'],
            '--profiles',root/'outputs/round3/image_profiles.json','--out',report],'validation.log')
        data=json.loads(report.read_text());base=json.loads((root/'outputs/round8_mixstyle_mid_bright_tta.json').read_text())
        for key in ['sizes','hflip','brightness_floor','max_brightening']:
            if data[key]!=base[key]:raise RuntimeError('Mismatched baseline inference settings')
        if data['ensemble_checkpoint'] is not None:raise RuntimeError('Unexpected ensemble')
        scores={k:v['mIoU_present_nonignored'] for k,v in data['groups'].items()}
        baseline={k:v['mIoU_present_nonignored'] for k,v in base['groups'].items()}
        summary.update(scores=scores,baseline=baseline,checkpoint=str(train_out/'best.pth'))
        if balanced:
            previous=json.loads((root/'outputs/round12_pseudo/selection.json').read_text())
            summary['round12_scores']=previous['scores']
            summary['delta_vs_round12']={k:scores[k]-previous['scores'][k] for k in scores}
        # This is target-domain self training. Export is a stability check, not a claimed gain.
        if scores['all']<baseline['all']-.003 or any(scores[k]<baseline[k]-.005 for k in ['brightness','contrast']):
            (results/'selection.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
            record('validation_regression',**summary);return
        record('predicting',**summary)
        output=root/('outputs/submission_'+experiment+'_single')
        run('dino_predict.py',infer+['--images',root/'work/round2_data/images','--out',output],'prediction.log')
        archive=output.with_suffix('.zip')
        with zipfile.ZipFile(archive) as z:assert len(z.namelist())==1300 and z.testzip() is None
        summary.update(zip=str(archive),sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
        (results/'selection.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');record('complete',**summary)
    except Exception as error:
        record('failed',error=str(error));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workspace',default='.')
    p.add_argument('--class-balanced',action='store_true')
    main(p.parse_args())

