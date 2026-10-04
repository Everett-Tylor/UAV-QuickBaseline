"""Convert a verified Meta DINOv3 ViT-B checkpoint to local Transformers format."""
import argparse
import hashlib
import json
import re
from pathlib import Path

import torch
from transformers import DINOv3ViTConfig, DINOv3ViTModel

OFFICIAL_SHA256 = '73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c'


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def convert(meta):
    result = {}
    for key, value in meta.items():
        if key.startswith('rope_embed.') or key.endswith('.attn.qkv.bias_mask'):
            continue
        if key in ('cls_token', 'mask_token', 'storage_tokens'):
            name = {'cls_token': 'cls_token', 'mask_token': 'mask_token',
                    'storage_tokens': 'register_tokens'}[key]
            result['embeddings.' + name] = value.unsqueeze(1) if key == 'mask_token' else value
            continue
        if key.startswith('patch_embed.proj.'):
            result[key.replace('patch_embed.proj.', 'embeddings.patch_embeddings.')] = value
            continue
        if key.startswith('norm.'):
            result[key] = value
            continue
        match = re.fullmatch(r'blocks\.(\d+)\.(.+)', key)
        if not match:
            raise ValueError(f'Unexpected source tensor: {key}')
        layer, suffix = match.groups()
        base = f'layer.{layer}.'
        if suffix.startswith('attn.qkv.'):
            q, k, v = value.chunk(3, dim=0)
            ending = suffix.rsplit('.', 1)[1]
            result[base + 'attention.q_proj.' + ending] = q.contiguous()
            if ending != 'bias':
                result[base + 'attention.k_proj.' + ending] = k.contiguous()
            result[base + 'attention.v_proj.' + ending] = v.contiguous()
        else:
            suffix = suffix.replace('attn.proj.', 'attention.o_proj.')
            suffix = suffix.replace('mlp.fc1.', 'mlp.up_proj.')
            suffix = suffix.replace('mlp.fc2.', 'mlp.down_proj.')
            suffix = re.sub(r'ls([12])\.gamma', r'layer_scale\1.lambda1', suffix)
            result[base + suffix] = value
    return result


def main(args):
    source, output = Path(args.checkpoint).resolve(), Path(args.out).resolve()
    if file_hash(source) != OFFICIAL_SHA256:
        raise ValueError('Source checkpoint does not match official DINOv3 ViT-B SHA-256')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a fresh output directory')
    config = DINOv3ViTConfig.from_pretrained(args.config, local_files_only=True)
    if (config.hidden_size, config.num_hidden_layers, config.num_attention_heads) != (768, 12, 12):
        raise ValueError('Expected DINOv3 ViT-B/16 config')
    raw = torch.load(source, map_location='cpu', weights_only=True, mmap=True)
    converted = convert(raw)
    model = DINOv3ViTModel(config)
    model.load_state_dict(converted, strict=True)
    output.mkdir(parents=True)
    model.save_pretrained(output, safe_serialization=True)
    check = DINOv3ViTModel.from_pretrained(output, local_files_only=True)
    if any(not torch.equal(a, b) for a, b in zip(model.state_dict().values(), check.state_dict().values())):
        raise RuntimeError('Saved model failed exact round-trip check')
    provenance = {'upstream_model': 'facebook/dinov3-vitb16-pretrain-lvd1689m',
                  'source_checkpoint_sha256': OFFICIAL_SHA256,
                  'conversion': 'Meta PyTorch tensor names to Transformers 4.57.1'}
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    print(f'VERIFIED {output} tensors={len(converted)}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', required=True)
    main(parser.parse_args())
