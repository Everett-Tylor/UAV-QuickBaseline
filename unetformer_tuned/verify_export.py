"""Independently validate every PNG inside a competition submission ZIP."""
import argparse,hashlib,io,json,zipfile
from pathlib import Path
import numpy as np
from PIL import Image

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--zip',required=True)
    parser.add_argument('--report',required=True)
    a=parser.parse_args();path=Path(a.zip)
    expected={f'test2_{i}.png' for i in range(1,1301)}
    counts=np.zeros(9,dtype=np.int64)
    with zipfile.ZipFile(path) as z:
        if len(z.namelist())!=1300 or set(z.namelist())!=expected:raise ValueError('Incorrect filenames')
        for name in z.namelist():
            with Image.open(io.BytesIO(z.read(name))) as im:
                if im.mode!='L' or im.size!=(1024,1024):raise ValueError(f'Invalid PNG: {name}')
                values=np.asarray(im)
                if values.max()>8:raise ValueError(f'Invalid class IDs: {name}')
                counts+=np.bincount(values.ravel(),minlength=9)
    with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    report={'png_count':1300,'dimensions':[1024,1024],'mode':'L',
            'all_member_crc_checks_passed':True,'class_pixel_counts':counts.tolist(),
            'zip_bytes':path.stat().st_size,'sha256':digest}
    Path(a.report).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
