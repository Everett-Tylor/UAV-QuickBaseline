"""Single DINOv3 teacher, conservative test2 pseudo labels, bare-ground-focused student."""
import argparse
import hashlib
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def better(student,teacher):
    old=teacher['groups'];new=student['groups']
    overall=new['all']['mIoU_present_nonignored']
    bare=new['all']['per_class_IoU']['5']
    return (overall>max(old['all']['mIoU_present_nonignored'],.7841157214158992)
            and bare>old['all']['per_class_IoU']['5']
            and all(new[g]['mIoU_present_nonignored']>=old[g]['mIoU_present_nonignored']-.005
                    for g in ('brightness','contrast')))


def main(a):
    code=Path(__file__).resolve().parent
    out=Path(a.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a fresh output directory')
    out.mkdir(parents=True,exist_ok=True)

    def record(stage,**details):
        payload={'stage':stage,'time':time.strftime('%Y-%m-%d %H:%M:%S'),**details}
        temporary=out/'status.tmp'
        temporary.write_text(json.dumps(payload,indent=2),encoding='utf-8')
        temporary.replace(out/'status.json')
        print(json.dumps(payload),flush=True)

    def run(script,args,name):
        with (out/name).open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-u',str(code/script),*map(str,args)],
                           stdout=log,stderr=subprocess.STDOUT,check=True)

    try:
        paths={key:Path(getattr(a,key)).resolve() for key in
               ('teacher','source','images','masks','test_images','split','class_balance','profiles')}
        missing=[f'{key}: {path}' for key,path in paths.items() if not path.exists()]
        if missing:
            record('awaiting_required_files',missing=missing)
            return
        teacher=paths['teacher'];source=paths['source']
        if not (source/'config.json').exists():
            raise FileNotFoundError(f'DINOv3 model config missing: {source}')
        profile=json.loads(paths['profiles'].read_text(encoding='utf-8'))
        from quickseg import images
        test=images(paths['test_images'])
        if len(test)!=1300 or len(profile)!=8296:
            raise ValueError('Expected audited 6996 labeled and 1300 test2 images')
        if sha256(teacher)=='46182c05c2526bf1c0f360ebd6f3716e2c0491ef17b5f85ed5d1bf9eb0ce444c':
            record('teacher_matches_round8',checkpoint_sha256=sha256(teacher))
        else:
            record('teacher_differs_from_round8',checkpoint_sha256=sha256(teacher),
                   note='Will evaluate supplied teacher before training')
        shared=['--source',source,'--sizes',512,640,768,896,'--hflip',
                '--brightness-floor',.35,'--max-brightening',1.25]
        record('evaluating_teacher')
        teacher_report=out/'teacher_validation.json'
        run('dino_predict.py',shared+['--checkpoint',teacher,'--images',paths['images'],
            '--masks',paths['masks'],'--split',paths['split'],'--profiles',paths['profiles'],
            '--out',teacher_report],'teacher_validation.log')
        teacher_result=json.loads(teacher_report.read_text(encoding='utf-8'))
        if teacher_result['groups']['all']['mIoU_present_nonignored']<.70:
            raise ValueError('Supplied teacher below 70% validation mIoU; inspect checkpoint/config')
        pseudo=out/'pseudo_masks'
        record('generating_pseudo_labels',teacher_miou=teacher_result['groups']['all']['mIoU_present_nonignored'])
        run('dino_pseudo.py',['--source',source,'--checkpoint',teacher,'--images',paths['test_images'],
            '--profiles',paths['profiles'],'--out',pseudo,'--confidence',.95,'--agreement',.75,
            '--bare-confidence',.98,'--bare-agreement',.875,'--workers',2], 'pseudo_generation.log')
        manifest=json.loads((pseudo/'manifest.json').read_text(encoding='utf-8'))
        usable=[r for r in manifest['images'] if r['coverage']>=manifest['min_coverage']]
        bare_pixels=sum(r['retained_bare_pixels'] for r in usable)
        if len(usable)<100 or bare_pixels<10000:
            record('insufficient_pseudo_labels',usable=len(usable),retained_bare_pixels=bare_pixels)
            return
        shared_train=['--images',paths['images'],'--masks',paths['masks'],'--split',paths['split'],
            '--class-balance',paths['class_balance'],'--source',source,'--init',teacher,
            '--pseudo-manifest',pseudo/'manifest.json','--pseudo-batch',1,'--pseudo-weight',.2,
            '--epochs',3,'--freeze-epochs',0,'--size',640,'--batch',1,'--accum',8,
            '--workers',2,'--lr',1e-6,'--head-lr',1e-5,'--seed',20260928,
            '--augmentation','robust','--focus-bare']
        record('smoke_testing',usable=len(usable),retained_bare_pixels=bare_pixels)
        run('dino_train.py',shared_train+['--out',out/'smoke','--smoke'],'smoke.log')
        record('training',epochs=3,usable=len(usable),retained_bare_pixels=bare_pixels)
        run('dino_train.py',shared_train+['--out',out/'student'],'train.log')
        record('validating')
        student_report=out/'student_validation.json'
        run('dino_predict.py',shared+['--checkpoint',out/'student/best.pth',
            '--images',paths['images'],'--masks',paths['masks'],'--split',paths['split'],
            '--profiles',paths['profiles'],'--out',student_report],'student_validation.log')
        student_result=json.loads(student_report.read_text(encoding='utf-8'))
        selection={'teacher':teacher_result['groups'],'student':student_result['groups'],
                   'retained_bare_pixels':bare_pixels,'usable_pseudo_images':len(usable),
                   'single_model':True,'test2_used_as_unlabeled_pseudo_data':True,
                   'improved':better(student_result,teacher_result),'official_score':None}
        (out/'selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
        if not selection['improved']:
            record('validation_not_improved',teacher_miou=teacher_result['groups']['all']['mIoU_present_nonignored'],
                   student_miou=student_result['groups']['all']['mIoU_present_nonignored'],
                   teacher_bare_iou=teacher_result['groups']['all']['per_class_IoU']['5'],
                   student_bare_iou=student_result['groups']['all']['per_class_IoU']['5'])
            return
        record('predicting_test2')
        prediction=out/'submission_dino_bare'
        run('dino_predict.py',shared+['--checkpoint',out/'student/best.pth',
            '--images',paths['test_images'],'--out',prediction],'prediction.log')
        archive=prediction.with_suffix('.zip')
        with zipfile.ZipFile(archive) as z:
            if len(z.namelist())!=1300 or z.testzip() is not None:
                raise ValueError('Prediction archive failed integrity check')
        record('complete',archive=str(archive),sha256=sha256(archive),
               student_miou=student_result['groups']['all']['mIoU_present_nonignored'],
               student_bare_iou=student_result['groups']['all']['per_class_IoU']['5'],
               official_score=None)
    except Exception as error:
        record('failed',error=str(error))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('teacher','source','images','masks','test-images','split','class-balance','profiles','out'):
        parser.add_argument('--'+key,required=True)
    main(parser.parse_args())
