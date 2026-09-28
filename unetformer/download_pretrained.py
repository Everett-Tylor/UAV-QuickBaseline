"""Fetch the exact upstream ResNet18 SWSL weights and verify SHA-256."""
import argparse,hashlib,json,urllib.request
from pathlib import Path

URL='https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnet18-118f1556.pth'
SHA256='118f1556c5734939bce4171a5223f613f7ff344080967bb054db2d41613bbb3d'

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True)
    out=Path(p.parse_args().out);out.mkdir(parents=True,exist_ok=True)
    path=out/'resnet18_swsl.pth'
    with urllib.request.urlopen(URL,timeout=120) as src,path.open('wb') as dst:
        while block:=src.read(1024**2):dst.write(block)
    with path.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
    if actual!=SHA256:raise ValueError('Pretrained SHA-256 mismatch')
    (out/'provenance.json').write_text(json.dumps({'url':URL,'sha256':actual,'license':'CC-BY-NC-4.0'},indent=2))

if __name__=='__main__':main()
