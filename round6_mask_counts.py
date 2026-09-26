import argparse,json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from PIL import Image

def count(path):
    with Image.open(path) as im:a=np.asarray(im)
    assert a.ndim==2 and a.max()<=8,path
    return Path(path).name,np.bincount(a.reshape(-1),minlength=9).tolist()

def main():
    p=argparse.ArgumentParser();p.add_argument('--masks',required=True);p.add_argument('--split',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    split=json.loads(Path(a.split).read_text());print('split keys',list(split),flush=True)
    names=split['train'];paths=[str(Path(a.masks)/(Path(n).stem+'.png')) for n in names]
    result={}
    with ProcessPoolExecutor(max_workers=4) as ex:
        for i,(name,counts) in enumerate(ex.map(count,paths),1):
            result[name]=counts
            if i%1000==0:print(i,flush=True)
    Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(json.dumps(result));print('counted',len(result),flush=True)
if __name__=='__main__':main()
