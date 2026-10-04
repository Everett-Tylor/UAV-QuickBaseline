import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from dinoseg import feature_block_indices
from dino_vith_bare_pipeline import improved


class ViTHConfigurationTests(unittest.TestCase):
    def test_feature_levels_follow_backbone_depth(self):
        self.assertEqual(feature_block_indices(SimpleNamespace(num_hidden_layers=12)), [3, 6, 9, 12])
        self.assertEqual(feature_block_indices(SimpleNamespace(num_hidden_layers=32)), [8, 16, 24, 32])

    def test_bundled_config_is_vith_plus(self):
        source = Path(__file__).parent / 'model_configs' / 'dinov3-vith16plus' / 'config.json'
        config = json.loads(source.read_text(encoding='utf-8'))
        self.assertEqual((config['hidden_size'], config['num_hidden_layers'],
                          config['num_attention_heads'], config['use_gated_mlp']),
                         (1280, 32, 20, True))

    def test_selection_requires_both_metrics_to_improve(self):
        def report(miou, bare):
            return {'groups': {'all': {'mIoU_present_nonignored': miou,
                                       'per_class_IoU': {'5': bare}}}}
        teacher = report(.78, .53)
        self.assertTrue(improved(report(.79, .54), teacher))
        self.assertFalse(improved(report(.79, .52), teacher))
        self.assertFalse(improved(report(.77, .54), teacher))


if __name__ == '__main__':
    unittest.main()
