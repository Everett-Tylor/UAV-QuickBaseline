"""Resume round-three orchestration after a saved continuation trial."""
import argparse,copy,json,hashlib
from pathlib import Path
import torch

from round3 import (Pairs,MODES,eval_mode,evaluate_hv,export,loader,make_model,
                    run_trial,sampling_weights,save_json)


def best_history(path, baseline):
    rows=[json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]
    eligible=[r for r in rows if r['mIoU']>baseline]
    return max(eligible,key=lambda r:r['mIoU']) if eligible else None


def main():
    p=argparse.ArgumentParser()
    for n in ['images','masks','cache','split','init','audit','test-images','out']:
        p.add_argument('--'+n,required=True)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--seed',type=int,default=20260930)
    a=p.parse_args();out=Path(a.out)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
    split=json.loads(Path(a.split).read_text(encoding='utf-8'))
    audit=json.loads(Path(a.audit).read_text(encoding='utf-8'))
    if hashlib.sha256(Path(a.split).read_bytes()).hexdigest()!=audit['split_sha256']:
        raise ValueError('Class weights/split mismatch')
    baseline=json.loads((out/'baseline.json').read_text(encoding='utf-8'))
    continuation=best_history(out/'continuation/history.jsonl',baseline['mIoU'])
    save_json(out/'continuation/selection.json',{'selected':continuation,
              'baseline_mIoU':baseline['mIoU'],
              'note':'Epoch 6 raw validation completed before interruption; last complete training state is epoch 5.'})
    sample_weights=sampling_weights(split['train'],Path(a.cache)/'masks')
    recipes=[
      {'name':'rare_images','epochs':8,'lr':3e-6,'balanced':True,'strength':.1,
       'sizes':[512,448,576],'sampling':True,'hard_pixels':False},
      {'name':'hard_pixels','epochs':8,'lr':3e-6,'balanced':True,'strength':.1,
       'sizes':[512,448,576],'sampling':False,'hard_pixels':True},
    ]
    best_path=out/'continuation/best.pth' if continuation else Path(a.init)
    best={'trial':'continuation',**continuation} if continuation else {'trial':'baseline',**baseline}
    trials=[{'name':'continuation','best':continuation}]
    for recipe in recipes:
        trial_dir=out/recipe['name']
        if trial_dir.exists():raise FileExistsError(f'Expected absent trial directory: {trial_dir}')
        ck,record=run_trial(a,recipe,split,audit['weights'],baseline['mIoU'],sample_weights)
        trials.append({'name':recipe['name'],'best':record})
        if record and record['mIoU']>best['mIoU']:
            best_path=ck;best=record
    state=torch.load(best_path,map_location='cpu',weights_only=True)
    prior=torch.load(a.init,map_location='cpu',weights_only=True)['model']
    current=copy.deepcopy(state['model']);averages=[]
    model=make_model().cuda()
    val=loader(Pairs(split['val'],a.images,a.masks,512),2,a.workers)
    for old_fraction in [.25,.5]:
        mixed={k:(v*(1-old_fraction)+prior[k]*old_fraction if v.is_floating_point() else v.clone())
               for k,v in current.items()}
        model.load_state_dict(mixed);result=evaluate_hv(model,val)
        averages.append({'prior_weight':old_fraction,**result})
        if result['mIoU']>best['mIoU']:
            state={**state,'model':mixed,'metrics':result,'averaging_prior_fraction':old_fraction}
            best={**best,**result,'averaging_prior_fraction':old_fraction}
    save_json(out/'weight_averaging.json',averages);del model,val
    torch.save(state,out/'best.pth')
    model=make_model().cuda();model.load_state_dict(state['model']);model.eval()
    candidates=[]
    for name,mode in MODES.items():
        result=eval_mode(model,a,split['val'],mode)
        candidates.append({'name':name,'mode':mode,**result})
        print('mode',name,result['mIoU'],flush=True)
    selected=max(candidates,key=lambda x:x['mIoU'])
    summary={'baseline':baseline,'trials':trials,'checkpoint_selection':best,
             'inference_candidates':candidates,'selected':selected,
             'delta_percentage_points':100*(selected['mIoU']-baseline['mIoU']),
             'official_score':None,
             'interruption_note':'Continuation epoch 6 EMA evaluation was interrupted; epoch 5 was the last complete saved state.'}
    save_json(out/'selection.json',summary)
    export(model,a.test_images,out,selected['mode'])
    save_json(out/'status.json',{'stage':'complete','mIoU':selected['mIoU'],
              'delta_percentage_points':summary['delta_percentage_points'],
              'official_score':None})


if __name__=='__main__':main()

