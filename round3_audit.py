"""Image-only distribution and exact pixel duplicate audit; no test labels used."""
import argparse,hashlib,json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image

def inspect(item):
    group,path=item
    with Image.open(path) as im:
        rgb=im.convert('RGB');digest=hashlib.sha256(rgb.tobytes()).hexdigest()
        x=np.asarray(rgb.resize((64,64)),dtype=np.float32)/255
    gray=x.mean(2)
    return {'group':group,'name':path.name,'sha256_pixels':digest,
        'brightness':float(gray.mean()),'contrast':float(gray.std()),
        'saturation':float((x.max(2)-x.min(2)).mean()),
        'edge':float(np.abs(np.diff(gray,axis=0)).mean()+np.abs(np.diff(gray,axis=1)).mean()),
        'rgb':x.mean((0,1)).tolist()}

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['images','test-images','split','out']:p.add_argument('--'+name,required=True)
    a=p.parse_args();split=json.loads(Path(a.split).read_text());val=set(split['val'])
    paths=sorted(Path(a.images).glob('*.png'))
    if set(split['train'])&val:raise ValueError('Train/val overlap')
    if {x.name for x in paths}!=set(split['train']+split['val']):raise ValueError('Split coverage mismatch')
    jobs=[('val' if x.name in val else 'train',x) for x in paths]
    jobs += [('test2',x) for x in sorted(Path(a.test_images).glob('*.png'))]
    with ThreadPoolExecutor(6) as pool:rows=list(pool.map(inspect,jobs))
    hashes={};duplicates=[]
    for r in rows:
        previous=hashes.get(r['sha256_pixels'])
        if previous and previous['group']!=r['group']:
            duplicates.append([previous['group'],previous['name'],r['group'],r['name']])
        if not previous:hashes[r['sha256_pixels']]=r
    summary={g:{'count':sum(r['group']==g for r in rows),**{k:np.percentile([r[k] for r in rows if r['group']==g],[10,50,90]).tolist() for k in ['brightness','contrast','saturation','edge']}} for g in ['train','val','test2']}
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    (out/'image_profiles.json').write_text(json.dumps(rows),encoding='utf-8')
    (out/'data_audit.json').write_text(json.dumps({'summary':summary,'cross_group_exact_duplicates':duplicates,'scene_separation':'not established from numeric filenames'},indent=2),encoding='utf-8')
    print(json.dumps({'summary':summary,'cross_group_exact_duplicates':len(duplicates)}))
