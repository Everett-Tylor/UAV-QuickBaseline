import unittest
from b2_calibrate import choose_candidate


def candidate(calibration, confirmation, overall):
    return {name: {'mIoU_present_nonignored': score}
            for name, score in [('calibration', calibration), ('confirmation', confirmation), ('all', overall)]}


class CalibrationTests(unittest.TestCase):
    def test_rejects_tuning_gain_that_fails_confirmation(self):
        base, tuned = candidate(.74, .75, .745), candidate(.77, .74, .755)
        selected, approved = choose_candidate([base, tuned])
        self.assertIs(selected, base)
        self.assertFalse(approved)

    def test_accepts_gain_in_both_subsets(self):
        base, tuned = candidate(.74, .75, .745), candidate(.75, .76, .755)
        selected, approved = choose_candidate([base, tuned])
        self.assertIs(selected, tuned)
        self.assertTrue(approved)

    def test_does_not_search_confirmation_for_alternative(self):
        base = candidate(.74, .75, .745)
        chosen_on_calibration = candidate(.78, .74, .76)
        alternative = candidate(.76, .78, .77)
        selected, approved = choose_candidate([base, chosen_on_calibration, alternative])
        self.assertIs(selected, base)
        self.assertFalse(approved)


if __name__ == '__main__':
    unittest.main(verbosity=2)
