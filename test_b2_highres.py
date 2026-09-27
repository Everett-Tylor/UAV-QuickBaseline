import unittest
from b2_highres_pipeline import select_confirmed


def candidate(calibration, confirmation, overall):
    return {'candidate': {key: {'mIoU_present_nonignored': value} for key, value in
                          [('calibration', calibration), ('confirmation', confirmation), ('all', overall)]}}


class HighResolutionTests(unittest.TestCase):
    def test_selects_higher_calibration_score_then_confirms(self):
        base = candidate(.755, .755, .755)
        new = candidate(.76, .76, .76)
        chosen, improved = select_confirmed(base, [new])
        self.assertIs(chosen, new)
        self.assertTrue(improved)

    def test_failed_confirmation_preserves_previous_archive(self):
        base = candidate(.755, .755, .755)
        bad = candidate(.77, .75, .76)
        other = candidate(.76, .77, .765)
        chosen, improved = select_confirmed(base, [bad, other])
        self.assertIs(chosen, base)
        self.assertFalse(improved)

    def test_tie_does_not_claim_gain(self):
        base = candidate(.755, .755, .755)
        chosen, improved = select_confirmed(base, [candidate(.755, .755, .755)])
        self.assertIs(chosen, base)
        self.assertFalse(improved)


if __name__ == '__main__':
    unittest.main(verbosity=2)
