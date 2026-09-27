"""Read-only validation diagnostic, never export validation examples as training data."""
import json,subprocess,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw

def main():
    root=Path.cwd();out=root/'outputs/barren_error_audit';out.mkdir(exist_ok=True)
    cfg=json.loads((root/'outputs/runs/round8_mixstyle640/config.json').read_text())
    records={}
    for tag,path in [('round8','outputs/round8_mixstyle_mid_bright_tta.json'),('round12','outputs/round12_pseudo/validation.json'),('round14','outputs/round14_classmix_barren/validation.json')]:
        h=np.array(json.loads((root/path).read_text())['groups']['all']['confusion_matrix'])
        tp=h[5,5];fn=h[5].sum()-tp;fp=h[:,5].sum()-tp
        records[tag]={'precision':float(tp/h[:,5].sum()),'recall':float(tp/h[5].sum()),'fp':int(fp),'fn':int(fn),
            'miss_to':{str(c):int(h[5,c]) for c in range(1,9) if c!=5},'false_from':{str(c):int(h[c,5]) for c in range(1,9) if c!=5}}
    (out/'comparison.json').write_text(json.dumps(records,indent=2))
    audit=out/'round12_640'
    if audit.exists():raise RuntimeError('Audit directory exists; inspect before rerunning')
    args=[sys.executable,'-u','outputs/UAV-QuickBaseline/dino_predict.py','--checkpoint','outputs/runs/round12_pseudo640/best.pth',
          '--source',cfg['source'],'--images',cfg['images'],'--masks',cfg['masks'],'--split',cfg['split'],
          '--sizes','640','--brightness-floor','.35','--max-brightening','1.25','--audit-dir',str(audit),'--out',str(out/'round12_640.json')]
    subprocess.run(args,check=True)
    rows=[json.loads(line) for line in (audit/'images.jsonl').read_text().splitlines()]
    palette=np.array([[0,0,0],[100,100,100],[220,80,80],[220,220,220],[40,100,230],[200,150,60],[30,150,40],[150,210,50],[240,80,220]],dtype=np.uint8)
    ranked={}
    for mode in ['false_positive','false_negative']:
        top=sorted(rows,key=lambda r:r[mode],reverse=True)[:8];ranked[mode]=top
        sheet=Image.new('RGB',(4*256,8*280),'white');draw=ImageDraw.Draw(sheet)
        for i,r in enumerate(top):
            with Image.open(Path(cfg['images'])/r['name']) as im:rgb=im.convert('RGB')
            with Image.open(Path(cfg['masks'])/r['name']) as im:y=np.asarray(im).copy()
            with Image.open(audit/r['name']) as im:p=np.asarray(im).copy()
            error=np.zeros((*y.shape,3),dtype=np.uint8)
            error[(p==5)&(y!=5)&(y!=0)]=[255,60,60]
            error[(y==5)&(p!=5)]=[40,150,255]
            for j,tile in enumerate([rgb,Image.fromarray(palette[y]),Image.fromarray(palette[p]),Image.fromarray(error)]):
                sheet.paste(tile.resize((256,256)),(j*256,i*280+24))
            draw.text((0,i*280),f'{i+1}: {r["name"][:28]} FP={r["false_positive"]} FN={r["false_negative"]}',fill='black')
        sheet.save(out/(mode+'.jpg'))
    (out/'ranked.json').write_text(json.dumps(ranked,indent=2))
    print('AUDIT_COMPLETE: RGB / ground truth / prediction / red FP blue FN',flush=True)

if __name__=='__main__':main()
