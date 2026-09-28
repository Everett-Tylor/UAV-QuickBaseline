import unittest
from dino_bare_pipeline import better


def result(all_score,bare,dark=.72,contrast=.745):
    return {'groups':{'all':{'mIoU_present_nonignored':all_score,'per_class_IoU':{'5':bare}},
                      'brightness':{'mIoU_present_nonignored':dark},
                      'contrast':{'mIoU_present_nonignored':contrast}}}


class BarePipelineTests(unittest.TestCase):
    def test_bare_gain_without_overall_gain_is_rejected(self):
        self.assertFalse(better(result(.783,.55),result(.78384,.5308)))

    def test_overall_gain_without_bare_gain_is_rejected(self):
        self.assertFalse(better(result(.785,.52),result(.78384,.5308)))

    def test_both_gains_and_stable_subgroups_are_required(self):
        teacher=result(.78384,.5308)
        self.assertTrue(better(result(.785,.55),teacher))
        self.assertFalse(better(result(.785,.55,dark=.71),teacher))


if __name__=='__main__':unittest.main(verbosity=2)
