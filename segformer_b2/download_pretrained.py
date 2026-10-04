"""Download the pinned NVIDIA MiT-B2 encoder and record SHA-256 hashes."""
import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

MODEL = 'nvidia/mit-b2'
REVISION = '3bb39e8739149c3777d0325349b2a6c32c6413db'
FILES = ('config.json', 'preprocessor_config.json', 'pytorch_model.bin')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    out = Path(parser.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in FILES:
        path = Path(hf_hub_download(repo_id=MODEL, revision=REVISION,
                                    filename=name, local_dir=out))
        with path.open('rb') as stream:
            hashes[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
        print(f'{name}: {path.stat().st_size} bytes {hashes[name]}', flush=True)
    (out / 'provenance.json').write_text(json.dumps({
        'model': MODEL, 'revision': REVISION, 'sha256': hashes}, indent=2),
        encoding='utf-8')


if __name__ == '__main__':
    main()
