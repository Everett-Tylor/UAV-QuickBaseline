"""Check that sampling metadata cannot include validation masks."""
import json,tempfile
from pathlib import Path
from round6_train import sampling_weights

def main():
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder);counts=root/'counts.json';profiles=root/'profiles.json'
        train=[(Path('a.png'),Path('a.png')),(Path('b.png'),Path('b.png'))]
        counts.write_text(json.dumps({'a.png':[0,90,0,0,0,10,0,0,0],'b.png':[0,99,0,0,0,0,0,0,1]}))
        profiles.write_text(json.dumps([{'name':'a.png','group':'train','brightness':.2,'contrast':.05},{'name':'b.png','group':'train','brightness':.5,'contrast':.2},{'name':'val.png','group':'val','brightness':0,'contrast':0}]))
        weights,report=sampling_weights(train,counts,profiles)
        assert len(weights)==2 and (weights>0).all() and weights.max()<=2.8
        d=json.loads(counts.read_text());d['val.png']=[0,100,0,0,0,0,0,0,0];counts.write_text(json.dumps(d))
        try:sampling_weights(train,counts,profiles)
        except ValueError:pass
        else:raise AssertionError('Validation metadata was not rejected')
    print('PASS: valid bounded weights; validation-mask metadata rejected')
if __name__=='__main__':main()
