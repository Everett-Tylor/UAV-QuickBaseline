import json
import tempfile
import unittest
from argparse import Namespace
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn

from improvedseg import (CropDataset, make_model, segmentation_loss, predict_logits,
                         metrics_from_hist, split_pairs, read_pair)

torch.set_num_threads(2)


class PixelModel(nn.Module):
    def forward(self, x):
        return torch.cat([x[:, :1] * (i+1) for i in range(9)], dim=1)


class SegmentationTests(unittest.TestCase):
    def test_ignored_pixels_have_zero_gradient(self):
        logits = torch.randn(2, 9, 8, 8, requires_grad=True)
        target = torch.randint(1, 9, (2, 8, 8))
        target[:, :3] = 0
        loss = segmentation_loss(logits, target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(logits.grad[:, :, :3].abs().sum()), 0.)

    def test_all_ignored_is_finite_differentiable_zero(self):
        logits = torch.randn(1, 9, 8, 8, requires_grad=True)
        loss = segmentation_loss(logits, torch.zeros(1, 8, 8, dtype=torch.long))
        loss.backward()
        self.assertEqual(float(loss.detach()), 0.)
        self.assertEqual(float(logits.grad.abs().sum()), 0.)

    def test_perfect_prediction_loss_is_small(self):
        target = torch.randint(1, 9, (1, 8, 8))
        logits = torch.full((1, 9, 8, 8), -10.).scatter_(1, target[:, None], 10.)
        self.assertLess(float(segmentation_loss(logits, target)), 1e-5)

    def test_class_weights_use_only_supplied_training_masks(self):
        from improvedseg import training_class_weights
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'train.png'
            mask = np.ones((10, 10), dtype=np.uint8)
            mask[0, 0] = 5
            Image.fromarray(mask).save(path)
            Image.fromarray(np.full((10, 10), 8, dtype=np.uint8)).save(Path(folder)/'excluded.png')
            weights, counts = training_class_weights([(Path('unused.png'), path)])
            self.assertEqual(counts[1], 99)
            self.assertEqual(counts[5], 1)
            self.assertEqual(counts[8], 0)
            self.assertEqual(float(weights[0]), 0.)
            self.assertGreater(float(weights[5]), float(weights[1]))
            self.assertLessEqual(float(weights.max()), 4.)
            logits = torch.randn(1, 9, 10, 10, requires_grad=True)
            target = torch.from_numpy(mask.astype(np.int64))[None]
            actual = segmentation_loss(logits, target, 0, weights)
            expected = torch.nn.functional.cross_entropy(logits, target, weight=weights, ignore_index=0)
            torch.testing.assert_close(actual, expected)

    def test_tile_blending_covers_edges_and_small_images(self):
        for shape in [(31, 47), (7, 9), (32, 32)]:
            x = torch.rand(1, 3, *shape)
            actual = predict_logits(PixelModel(), x, 'cpu', 16, overlap=.25)
            torch.testing.assert_close(actual, PixelModel()(x))

    def test_legacy_prediction_matches_original_resize(self):
        from improvedseg import tensor_image
        rgb = Image.fromarray(np.random.default_rng(42).integers(0, 256, (31, 47, 3), dtype=np.uint8))
        expected = PixelModel()(tensor_image(rgb.resize((16, 16), Image.Resampling.BILINEAR))[None])
        expected = torch.nn.functional.interpolate(expected, size=(31, 47), mode='bilinear', align_corners=False)
        actual = predict_logits(PixelModel(), tensor_image(rgb)[None], 'cpu', 16, mode='resize')
        torch.testing.assert_close(actual, expected)

    def test_context_model_backward_and_legacy_compatibility(self):
        for architecture in ('unet', 'context'):
            model = make_model(4, architecture)
            output = model(torch.rand(2, 3, 32, 40))
            self.assertEqual(tuple(output.shape), (2, 9, 32, 40))
            segmentation_loss(output, torch.randint(0, 9, (2, 32, 40))).backward()
            model2 = make_model(4, architecture)
            model2.load_state_dict(model.state_dict())

    def test_metric_false_positives_and_ignored_prediction(self):
        hist = torch.zeros(9, 9, dtype=torch.long)
        hist[1, 1] = 2
        hist[1, 0] = 1
        hist[1, 2] = 1
        metrics = metrics_from_hist(hist)
        self.assertEqual(metrics['per_class_IoU']['1'], .5)
        self.assertEqual(metrics['per_class_IoU']['2'], 0.)
        self.assertIsNone(metrics['per_class_IoU']['3'])
        self.assertEqual(metrics['mIoU_present_nonignored'], .25)

    def test_split_rejects_overlap_and_reuses_names(self):
        pairs = [(Path(f'{i}.png'), Path(f'mask{i}.png')) for i in range(3)]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'split.json'
            path.write_text(json.dumps({'train': ['2.png', '0.png'], 'val': ['1.png']}))
            train, val, _ = split_pairs(pairs, path, .1, 42)
            self.assertEqual([x[0].name for x in train], ['2.png', '0.png'])
            self.assertEqual(val[0][0].name, '1.png')
            path.write_text(json.dumps({'train': ['2.png', '0.png'], 'val': ['0.png', '1.png']}))
            with self.assertRaises(ValueError):
                split_pairs(pairs, path, .1, 42)

    def test_mask_ids_palette_and_dimensions(self):
        with tempfile.TemporaryDirectory() as folder:
            im, mask = Path(folder)/'im.png', Path(folder)/'mask.png'
            Image.fromarray(np.full((32, 40, 3), 128, dtype=np.uint8)).save(im)
            pal = Image.fromarray(np.ones((32, 40), dtype=np.uint8)).convert('P')
            pal.save(mask)
            rgb, label = CropDataset([(im, mask)], size=16, augment=True)[0]
            self.assertEqual(tuple(rgb.shape), (3, 16, 16))
            self.assertEqual(tuple(label.shape), (16, 16))
            self.assertTrue(set(label.unique().tolist()).issubset({0, 1}))
            Image.fromarray(np.full((32, 40), 255, dtype=np.uint8)).save(mask)
            with self.assertRaises(ValueError):
                read_pair(im, mask)
            Image.fromarray(np.ones((16, 16), dtype=np.uint8)).save(mask)
            with self.assertRaises(ValueError):
                read_pair(im, mask)

    def test_train_eval_infer_pack_roundtrip(self):
        from improvedseg import train, validate, infer, pack
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ('images', 'masks', 'test'):
                (root/name).mkdir()
            for i in range(3):
                rgb = np.zeros((64, 64, 3), dtype=np.uint8)
                rgb[:, :32, i] = 255
                label = np.ones((64, 64), dtype=np.uint8)
                label[:, 32:] = 2
                Image.fromarray(rgb).save(root/'images'/f'{i}.png')
                Image.fromarray(label).save(root/'masks'/f'{i}.png')
            args = Namespace(images=str(root/'images'), masks=str(root/'masks'),
                             output=str(root/'run'), seed=42, split=None, val_fraction=.3,
                             size=128, crops_per_image=1, batch=2, workers=0, cpu=True,
                             width=2, architecture='context', lr=.001, init_weights=None,
                             epochs=1, warmup_epochs=0, dice_weight=.5, overlap=.25)
            train(args)
            args.weights = str(root/'run/best.pth')
            args.split = str(root/'run/split.json')
            args.report = str(root/'report.json')
            validate(args)
            report = json.loads(Path(args.report).read_text())
            self.assertTrue(0 <= report['metrics']['mIoU_present_nonignored'] <= 1)
            Image.fromarray(np.zeros((1024, 1024, 3), dtype=np.uint8)).save(root/'test'/'test.png')
            args.images, args.output = str(root/'test'), str(root/'predictions')
            infer(args)
            args.predictions, args.zip = args.output, str(root/'submission.zip')
            pack(args)
            with zipfile.ZipFile(args.zip) as archive:
                self.assertEqual(archive.namelist(), ['test.png'])


if __name__ == '__main__':
    unittest.main()
