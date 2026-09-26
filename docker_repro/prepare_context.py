"""Restore exact weights from GitHub artifact parts before docker build."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
PACKAGES = [
    ('round2_model.zip', 'b2c5316650c7811b42c79190a55f54dd2f79223e25668c7c65bb2ad7f71c0763',
     'round2_model', 'final', ['best.pth', 'config.json', 'inference.json', 'pretrained_provenance.json']),
    ('round2_initial.zip', 'fae328e4f82e5ea4ab00a0d0845b092fcefa15b74105c5eb96322a3c77ef6d01',
     'initial', 'initial', ['config.json', 'model.safetensors']),
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifacts', type=Path, default=ROOT.parent/'artifacts/round2')
    parser.add_argument('--destination', type=Path, default=ROOT/'weights')
    args = parser.parse_args()
    cache = args.destination/'.archives'
    cache.mkdir(parents=True, exist_ok=True)
    for filename, expected, archive_dir, target_dir, names in PACKAGES:
        archive = cache/filename
        digest = hashlib.sha256()
        with archive.open('wb') as output:
            count = 7 if filename == 'round2_model.zip' else 11
            for i in range(1, count + 1):
                part = args.artifacts/f'{filename}.part{i:02}'
                with part.open('rb') as source:
                    while chunk := source.read(1024*1024):
                        output.write(chunk)
                        digest.update(chunk)
        if digest.hexdigest() != expected:
            raise RuntimeError(f'Checksum mismatch: {filename}')
        target = args.destination/target_dir
        target.mkdir(exist_ok=True)
        with zipfile.ZipFile(archive) as zipped:
            for name in names:
                # Fixed member list; no archive-controlled extraction paths.
                (target/name).write_bytes(zipped.read(f'{archive_dir}/{name}'))
        print('Restored:', filename)
    manifest=json.loads((ROOT/'MANIFEST.json').read_text(encoding='utf-8'))
    for name, expected in manifest['runtime_sha256'].items():
        if name.startswith('weights/'):
            actual=hashlib.sha256((args.destination/name[len('weights/'):]).read_bytes()).hexdigest()
            if actual != expected: raise RuntimeError('Restored weight mismatch: '+name)
    print('All restored weights match the original experiment.')

if __name__=='__main__':
    main()
