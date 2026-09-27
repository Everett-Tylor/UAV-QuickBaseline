"""Synthetic correctness checks only; these are not competition results."""
import argparse
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from transformers import SegformerConfig, SegformerForSemanticSegmentation

from b2_model import Segmenter
from b2_predict import run, validate_archive


class B2Tests(unittest.TestCase):
    def test_rejects_non_b2(self):
        with tempfile.TemporaryDirectory() as temp:
            SegformerConfig().save_pretrained(temp)
            with self.assertRaisesRegex(ValueError, 'B2'):
                Segmenter(temp, config_only=True)

    def test_archive_validates_pixels_and_names(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / 'sample.png'
            archive = root / 'prediction.zip'
            for value, valid in [(8, True), (9, False)]:
                Image.fromarray(np.full((1024, 1024), value, np.uint8)).save(path)
                with zipfile.ZipFile(archive, 'w') as z:
                    z.write(path, 'sample.png')
                if valid:
                    self.assertEqual(validate_archive(archive, [path])['images'], 1)
                else:
                    with self.assertRaises(ValueError):
                        validate_archive(archive, [path])
            with self.assertRaises(ValueError):
                validate_archive(archive, [path, root / 'sample.jpg'])

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), 'BF16 CUDA required')
    def test_real_b2_training_checkpoint_and_export(self):
        torch.set_num_threads(4)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ('images', 'masks', 'test', 'pretrained'):
                (root / name).mkdir()
            rng = np.random.default_rng(2026)
            for i in range(3):
                Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(root / 'images' / f'{i}.png')
                Image.fromarray(rng.integers(0, 9, (64, 64), dtype=np.uint8)).save(root / 'masks' / f'{i}.png')
            Image.fromarray(rng.integers(0, 256, (1024, 1024, 3), dtype=np.uint8)).save(root / 'test' / 'test.png')
            split = {'train': ['0.png', '1.png'], 'val': ['2.png']}
            (root / 'split.json').write_text(json.dumps(split), encoding='utf-8')
            (root / 'weights.json').write_text(json.dumps({'weights': [0.] + [1.] * 8}), encoding='utf-8')
            config = SegformerConfig(depths=[3, 4, 6, 3], hidden_sizes=[64, 128, 320, 512],
                                     decoder_hidden_size=768, num_labels=9)
            network = SegformerForSemanticSegmentation(config)
            network.save_pretrained(root / 'pretrained')
            del network
            source = Path(__file__).resolve().parent
            subprocess.run([sys.executable, str(source / 'b2_train.py'), '--images', str(root / 'images'),
                            '--masks', str(root / 'masks'), '--split', str(root / 'split.json'),
                            '--class-balance', str(root / 'weights.json'), '--source', str(root / 'pretrained'),
                            '--out', str(root / 'run'), '--epochs', '1', '--size', '64', '--batch', '1',
                            '--accum', '2', '--workers', '0'], check=True)
            self.assertTrue((root / 'run/best.pth').exists())
            args = argparse.Namespace(checkpoint=str(root / 'run/best.pth'), source=str(root / 'run/model_config'),
                                      images=str(root / 'test'), masks=None, split=None,
                                      out=str(root / 'prediction'), sizes=[64], hflip=True)
            report = run(args)
            self.assertEqual(report['images'], 1)
            self.assertEqual(len(report['sha256']), 64)
            args.images, args.masks = str(root / 'images'), str(root / 'masks')
            args.split, args.out = str(root / 'split.json'), str(root / 'validation.json')
            metrics = run(args)
            self.assertEqual(metrics['images'], 1)
            self.assertIsNotNone(metrics['mIoU_present_nonignored'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
